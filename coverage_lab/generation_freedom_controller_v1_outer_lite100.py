#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import MODEL_WINDOW, POOL_K, main_numbers, portfolio_bundle
from coverage_lab.generation_freedom_controller_v1_inner_validation import (
    prep,
    select_a_centroid,
    select_b_core_static,
    select_c_dynamic,
    select_d_controller_v1,
)

WORLD_COUNT = 100
DRAWS_PER_WORLD = 696
BASE_SEED = 20261004
NAMESPACE = "loto7-world-c-lite100-v0"
OUT_DIR = Path("results/controller_v1_outer_lite100")


def world_seed(world_id: int) -> int:
    raw = f"{BASE_SEED}:{NAMESPACE}:{world_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def make_world(world_id: int) -> pd.DataFrame:
    rng = random.Random(world_seed(world_id))
    rows = []
    for rnd in range(1, DRAWS_PER_WORLD + 1):
        nums = sorted(rng.sample(range(1, 38), 7))
        rows.append({"round": rnd, **{f"n{i+1}": n for i, n in enumerate(nums)}})
    return pd.DataFrame(rows)


def success5(tickets, actual) -> bool:
    return any(len(set(t) & actual) >= 5 for t in tickets)


def evaluate_world(world_id: int) -> dict:
    df = make_world(world_id)
    names = ["A_centroid", "B_core_static", "C_dynamic", "D_controller_v1"]
    success_sets = {n: set() for n in names}
    success6_sets = {n: set() for n in names}
    fidelity_mismatch = []

    for idx in range(MODEL_WINDOW, len(df)):
        hist = df.iloc[idx - MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(hist)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))
        pool = ranked[:POOL_K]
        seed = rnd * 100_003 + 20261002

        core, combos, z, m, median, high, d, core_min = prep(pool, scores, seed + 2)
        A = select_a_centroid(high, d)
        B = select_b_core_static(high, core_min)
        C = select_c_dynamic(high, core, z)
        D = select_d_controller_v1(high, core, z, m)

        _, trace = portfolio_bundle(pool, scores, seed + 2)
        current_d = [tuple(x) for x in trace["diversify"]]
        if D != current_d:
            fidelity_mismatch.append(rnd)

        variants = {
            "A_centroid": A,
            "B_core_static": B,
            "C_dynamic": C,
            "D_controller_v1": D,
        }
        for name, six in variants.items():
            if success5(six, actual):
                success6_sets[name].add(rnd)
            if success5(core + six, actual):
                success_sets[name].add(rnd)

    if fidelity_mismatch:
        raise RuntimeError(f"world {world_id} fidelity mismatch: {fidelity_mismatch[:10]}")

    pairwise = {}
    for other in ("A_centroid", "B_core_static", "C_dynamic"):
        pairwise[f"D_vs_{other}"] = {
            "D_only": len(success_sets["D_controller_v1"] - success_sets[other]),
            "other_only": len(success_sets[other] - success_sets["D_controller_v1"]),
            "both": len(success_sets["D_controller_v1"] & success_sets[other]),
            "union": len(success_sets["D_controller_v1"] | success_sets[other]),
        }

    return {
        "world_id": world_id,
        "world_seed": world_seed(world_id),
        "evaluation_rounds": DRAWS_PER_WORLD - MODEL_WINDOW,
        "success16_counts": {n: len(success_sets[n]) for n in names},
        "success6_counts": {n: len(success6_sets[n]) for n in names},
        "pairwise": pairwise,
        "fidelity_ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("invalid shard")

    ids = [i for i in range(1, WORLD_COUNT + 1) if (i - 1) % args.num_shards == args.shard_id]
    worlds = []
    for wid in ids:
        print(f"WORLD {wid}/{WORLD_COUNT}", flush=True)
        worlds.append(evaluate_world(wid))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"shard_{args.shard_id:03d}.json"
    payload = {
        "experiment": "generation_freedom_controller_v1_outer_lite100",
        "world_source": {
            "namespace": NAMESPACE,
            "base_seed": BASE_SEED,
            "world_count": WORLD_COUNT,
            "draws_per_world": DRAWS_PER_WORLD,
            "evaluation_rounds_per_world": DRAWS_PER_WORLD - MODEL_WINDOW,
            "note": "Reuses the pre-existing World C Lite100 deterministic world family; no world is generated from Controller v1 outcomes.",
        },
        "variants": {
            "A_centroid": "m>=median; fixed centroid distance top6",
            "B_core_static": "m>=median; fixed min distance to Core10 top6",
            "C_dynamic": "m>=median; iterative min_dist to selected set; deterministic tie only",
            "D_controller_v1": "m>=median; iterative min_dist; overlap/m/deterministic existing tie-break",
        },
        "boundary": [
            "These 100 worlds were defined by the earlier World C Lite100 seed family, before Controller v1.",
            "Each world has 100 warmup + 596 evaluation draws.",
            "No parameter, threshold, seed, or tie-break is tuned from these outer-world results.",
            "Primary comparison is equal-budget Core10+Diversify6 (16 tickets).",
            "This is external structural/null-world validation, not evidence of predictive skill in the real lottery.",
            "D must reproduce current Portfolio Diversify in every evaluated round or the shard fails.",
        ],
        "shard": {"id": args.shard_id, "num_shards": args.num_shards},
        "worlds": worlds,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
