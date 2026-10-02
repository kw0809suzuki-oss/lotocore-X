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

from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW,
    POOL_K,
    A_TICKETS,
    main_numbers,
    unique_structured_bundle,
    portfolio_bundle,
    bundle_metrics,
)
from coverage_lab.loto7_hybrid_ratio_sweep_v0 import RATIOS, select, merge_unique
import lotocore

WORLD_COUNT = 500
DRAWS_PER_WORLD = 696
EVAL_ROUNDS = 596
CONFIRM_BASE_SEED = 20261003
CONFIRM_NAMESPACE = "loto7-world-c-fixed-confirm-v0"


def world_seed(world_id: int) -> int:
    raw = f"{CONFIRM_BASE_SEED}:{CONFIRM_NAMESPACE}:{world_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def make_world(world_id: int) -> pd.DataFrame:
    rng = random.Random(world_seed(world_id))
    rows = []
    for rnd in range(1, DRAWS_PER_WORLD + 1):
        nums = sorted(rng.sample(range(1, 38), 7))
        rows.append({"round": rnd, **{f"n{i+1}": n for i, n in enumerate(nums)}})
    return pd.DataFrame(rows)


def evaluate_world(world_id: int) -> dict:
    df = make_world(world_id)
    structured = 0
    portfolio = 0
    hybrids = {f"{a}+{b}": 0 for a, b in RATIOS}

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx - MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        pool = sorted(range(1, 38), key=lambda n: (ranks[n], n))[:POOL_K]

        # Frozen World A ticket-generation rule.
        seed = rnd * 100_003 + 20261002
        a20, _ = unique_structured_bundle(pool, A_TICKETS, seed + 1)
        b20, _ = portfolio_bundle(pool, scores, seed + 2)

        structured += int(bundle_metrics(a20, actual)["tickets_ge5"] > 0)
        portfolio += int(bundle_metrics(b20, actual)["tickets_ge5"] > 0)

        for a_n, b_n in RATIOS:
            key = f"{a_n}+{b_n}"
            h20, _ = merge_unique(select(a20, a_n), select(b20, b_n), b20)
            hybrids[key] += int(bundle_metrics(h20, actual)["tickets_ge5"] > 0)

    if len(df) - MODEL_WINDOW != EVAL_ROUNDS:
        raise RuntimeError("evaluation length mismatch")

    best_single = max(structured, portfolio)
    deltas = {k: v - best_single for k, v in hybrids.items()}
    n_plus = sum(v > 0 for v in deltas.values())
    delta_min = min(deltas.values())
    fixed_condition = (n_plus >= 4 and delta_min >= 0)

    return {
        "world_id": world_id,
        "world_seed": world_seed(world_id),
        "n_plus": n_plus,
        "delta_min": delta_min,
        "fixed_condition": fixed_condition,
        "deltas": deltas,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/world_c_confirm"))
    args = p.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("invalid shard")
    ids = [i for i in range(1, WORLD_COUNT + 1) if (i - 1) % args.num_shards == args.shard_id]

    worlds = [evaluate_world(i) for i in ids]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"world_c_shard_{args.shard_id:03d}.json"
    payload = {
        "experiment": CONFIRM_NAMESPACE,
        "pre_fixed_condition": "N_plus >= 4 AND delta_min >= 0",
        "world_definition": {
            "count": WORLD_COUNT,
            "draws_per_world": DRAWS_PER_WORLD,
            "warmup": MODEL_WINDOW,
            "evaluation_rounds": EVAL_ROUNDS,
            "distribution": "independent Uniform 7-of-37 without replacement per draw",
            "confirm_base_seed": CONFIRM_BASE_SEED,
            "namespace": CONFIRM_NAMESPACE,
        },
        "frozen_from_world_a": {
            "pool_k": POOL_K,
            "structured_tickets": A_TICKETS,
            "portfolio_roles": [10, 6, 4],
            "hybrid_ratios": [list(x) for x in RATIOS],
            "ticket_seed_rule": "round*100003+20261002; structured +1; portfolio +2",
        },
        "worlds": worlds,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
