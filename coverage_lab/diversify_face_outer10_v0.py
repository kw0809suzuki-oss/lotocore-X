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
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW,
    POOL_K,
    avg_candidate_score,
    euclid,
    main_numbers,
    portfolio_bundle,
    sampled_combinations,
    standardized_vectors,
)

DRAWS_PER_WORLD = 696
BASE_SEED = 20261004
NAMESPACE = "loto7-world-c-lite100-v0"
OUT_DIR = Path("results/diversify_face_outer10_v0")
THRESHOLDS = (3, 4, 5)

# Frozen from World A (596 rounds) before this outer test.
# ticket_id is 1-based Portfolio20 position; Diversify = 11..16.
WORLD_A_SLOT_BASELINE = {
    11: {"distance_mean": 5.740442, "max_overlap_mean": 4.944631},
    12: {"distance_mean": 5.148121, "max_overlap_mean": 4.642617},
    13: {"distance_mean": 4.831775, "max_overlap_mean": 4.620805},
    14: {"distance_mean": 4.707827, "max_overlap_mean": 4.624161},
    15: {"distance_mean": 4.557645, "max_overlap_mean": 4.629195},
    16: {"distance_mean": 4.424164, "max_overlap_mean": 4.640940},
}


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


def overlap(a, b) -> int:
    return len(set(a) & set(b))


def evaluate_world(world_id: int) -> dict:
    df = make_world(world_id)
    counts = {
        str(th): {
            "face_tickets": 0,
            "face_successes": 0,
            "nonface_tickets": 0,
            "nonface_successes": 0,
        }
        for th in THRESHOLDS
    }
    face_ticket_count = 0
    nonface_ticket_count = 0

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx - MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        pool = sorted(range(1, 38), key=lambda n: (ranks[n], n))[:POOL_K]

        seed = rnd * 100_003 + 20261002
        selected, trace = portfolio_bundle(pool, scores, seed + 2)
        selected = [tuple(t) for t in selected]
        core = [tuple(t) for t in trace["core"]]

        combos = sampled_combinations(pool, seed + 2 + 5000, core)
        z = standardized_vectors(combos)
        core_centroid = tuple(
            sum(z[t][i] for t in core) / len(core)
            for i in range(len(z[core[0]]))
        )

        for ticket_id in range(11, 17):
            t = selected[ticket_id - 1]
            d = euclid(z[t], core_centroid)
            max_ov = max(overlap(t, u) for j, u in enumerate(selected, start=1) if j != ticket_id)
            base = WORLD_A_SLOT_BASELINE[ticket_id]
            face = d < base["distance_mean"] and max_ov < base["max_overlap_mean"]

            if face:
                face_ticket_count += 1
            else:
                nonface_ticket_count += 1

            hits = overlap(t, actual)
            for th in THRESHOLDS:
                bucket = counts[str(th)]
                if face:
                    bucket["face_tickets"] += 1
                    bucket["face_successes"] += int(hits >= th)
                else:
                    bucket["nonface_tickets"] += 1
                    bucket["nonface_successes"] += int(hits >= th)

    return {
        "world_id": world_id,
        "world_seed": world_seed(world_id),
        "evaluation_rounds": DRAWS_PER_WORLD - MODEL_WINDOW,
        "diversify_tickets": (DRAWS_PER_WORLD - MODEL_WINDOW) * 6,
        "face_ticket_count": face_ticket_count,
        "nonface_ticket_count": nonface_ticket_count,
        "thresholds": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-id", type=int, required=True)
    args = ap.parse_args()
    if not 1 <= args.world_id <= 10:
        raise ValueError("outer10 v0 fixes world-id to 1..10")

    payload = {
        "experiment": "diversify_face_outer10_v0",
        "hypothesis_frozen": {
            "role": "diversify",
            "face": "distance_from_core_centroid < frozen World-A same-slot mean AND max_overlap_20 < frozen World-A same-slot mean",
            "world_a_slot_baseline": WORLD_A_SLOT_BASELINE,
            "no_outer_recalibration": True,
        },
        "world_source": {
            "namespace": NAMESPACE,
            "base_seed": BASE_SEED,
            "pre_existing_family": "World C Lite100",
        },
        "boundary": [
            "World IDs 1..10 are fixed before reading this probe's outcomes.",
            "No feature, threshold, slot baseline, seed, or generator is tuned from outer results.",
            "World-A same-slot means are frozen constants; outer worlds do not recompute normalization.",
            "These are independent null/random worlds, not future real-lottery OOS.",
            "This probe tests whether the World-A face enrichment survives outside World A; it cannot establish real-lottery predictive skill.",
        ],
        "world": evaluate_world(args.world_id),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"world_{args.world_id:03d}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
