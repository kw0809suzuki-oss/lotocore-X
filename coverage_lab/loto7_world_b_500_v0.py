#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

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
WORLD_BASE_SEED = 20261002


def world_seed(world_id: int) -> int:
    raw = f"{WORLD_BASE_SEED}:loto7-world-b:{world_id}".encode()
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
    base_s = 0
    base_p = 0
    s_only_total = 0
    p_only_total = 0
    ratios = {
        f"{a}+{b}": {
            "fiveplus_rounds": 0,
            "exclusive_recovered": 0,
            "structured_only_recovered": 0,
            "portfolio_only_recovered": 0,
            "collision_repairs": 0,
        }
        for a, b in RATIOS
    }

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx - MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        pool = sorted(range(1, 38), key=lambda n: (ranks[n], n))[:POOL_K]

        # Frozen World A ticket-generation seed rule.
        seed = rnd * 100_003 + 20261002
        a20, _ = unique_structured_bundle(pool, A_TICKETS, seed + 1)
        b20, _ = portfolio_bundle(pool, scores, seed + 2)

        ma = bundle_metrics(a20, actual)
        mb = bundle_metrics(b20, actual)
        a5 = ma["tickets_ge5"] > 0
        b5 = mb["tickets_ge5"] > 0
        base_s += int(a5)
        base_p += int(b5)

        s_only = a5 and not b5
        p_only = b5 and not a5
        s_only_total += int(s_only)
        p_only_total += int(p_only)

        for a_n, b_n in RATIOS:
            key = f"{a_n}+{b_n}"
            h20, rep = merge_unique(select(a20, a_n), select(b20, b_n), b20)
            mh = bundle_metrics(h20, actual)
            h5 = mh["tickets_ge5"] > 0
            ratios[key]["fiveplus_rounds"] += int(h5)
            ratios[key]["collision_repairs"] += rep
            if s_only and h5:
                ratios[key]["structured_only_recovered"] += 1
                ratios[key]["exclusive_recovered"] += 1
            if p_only and h5:
                ratios[key]["portfolio_only_recovered"] += 1
                ratios[key]["exclusive_recovered"] += 1

    if len(df) - MODEL_WINDOW != EVAL_ROUNDS:
        raise RuntimeError("evaluation length mismatch")

    best_single = max(base_s, base_p)
    for m in ratios.values():
        m["delta_vs_best_single"] = m["fiveplus_rounds"] - best_single
        m["expanded_success_set"] = m["fiveplus_rounds"] > best_single

    return {
        "world_id": world_id,
        "world_seed": world_seed(world_id),
        "warmup": MODEL_WINDOW,
        "evaluation_rounds": EVAL_ROUNDS,
        "structured20_5plus_rounds": base_s,
        "portfolio20_5plus_rounds": base_p,
        "structured_only_total": s_only_total,
        "portfolio_only_total": p_only_total,
        "exclusive_total": s_only_total + p_only_total,
        "best_single_5plus_rounds": best_single,
        "ratios": ratios,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/world_b"))
    args = p.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("invalid shard")
    ids = [i for i in range(1, WORLD_COUNT + 1) if (i - 1) % args.num_shards == args.shard_id]

    records = []
    for world_id in ids:
        print(f"WORLD {world_id}/{WORLD_COUNT}", flush=True)
        records.append(evaluate_world(world_id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"world_b_shard_{args.shard_id:03d}.json"
    payload = {
        "experiment": "loto7_world_b_500_v0",
        "world_definition": {
            "count": WORLD_COUNT,
            "draws_per_world": DRAWS_PER_WORLD,
            "warmup": MODEL_WINDOW,
            "evaluation_rounds": EVAL_ROUNDS,
            "distribution": "independent Uniform 7-of-37 without replacement per draw",
            "world_base_seed": WORLD_BASE_SEED,
            "world_seed_rule": "uint64_be(first8bytes(SHA256('20261002:loto7-world-b:<world_id>')))",
        },
        "frozen_from_world_a": {
            "pool_k": POOL_K,
            "structured_tickets": A_TICKETS,
            "portfolio_roles": [10, 6, 4],
            "hybrid_ratios": [list(x) for x in RATIOS],
            "ticket_seed_rule": "round*100003+20261002; structured +1; portfolio +2",
            "generator_source": "coverage_lab/loto7_portfolio20_v0.py + loto7_hybrid_ratio_sweep_v0.py",
        },
        "shard": {"id": args.shard_id, "num_shards": args.num_shards},
        "worlds": records,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
