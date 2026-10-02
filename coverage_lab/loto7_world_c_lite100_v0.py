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
    MODEL_WINDOW, POOL_K, A_TICKETS, main_numbers,
    unique_structured_bundle, portfolio_bundle, bundle_metrics,
)
from coverage_lab.loto7_hybrid_ratio_sweep_v0 import RATIOS, select, merge_unique
import lotocore

WORLD_COUNT = 100
DRAWS_PER_WORLD = 696
BASE_SEED = 20261004
NAMESPACE = "loto7-world-c-lite100-v0"

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

def evaluate_world(world_id: int) -> dict:
    df = make_world(world_id)
    s = 0
    p = 0
    h = {f"{a}+{b}": 0 for a,b in RATIOS}
    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx-MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k,v in snap["ranks"].items()}
        scores = {int(k): float(v) for k,v in snap["scores"].items()}
        pool = sorted(range(1,38), key=lambda n:(ranks[n], n))[:POOL_K]

        seed = rnd * 100_003 + 20261002
        a20, _ = unique_structured_bundle(pool, A_TICKETS, seed + 1)
        b20, _ = portfolio_bundle(pool, scores, seed + 2)

        s += int(bundle_metrics(a20, actual)["tickets_ge5"] > 0)
        p += int(bundle_metrics(b20, actual)["tickets_ge5"] > 0)

        for a_n,b_n in RATIOS:
            key=f"{a_n}+{b_n}"
            h20,_=merge_unique(select(a20,a_n), select(b20,b_n), b20)
            h[key] += int(bundle_metrics(h20, actual)["tickets_ge5"] > 0)

    best=max(s,p)
    deltas={k:v-best for k,v in h.items()}
    n_plus=sum(v>0 for v in deltas.values())
    delta_min=min(deltas.values())
    return {
        "world_id": world_id,
        "n_plus": n_plus,
        "delta_min": delta_min,
        "fixed_condition": n_plus >= 4 and delta_min >= 0
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-id",type=int,required=True)
    ap.add_argument("--num-shards",type=int,required=True)
    args=ap.parse_args()

    ids=[i for i in range(1,WORLD_COUNT+1) if (i-1)%args.num_shards==args.shard_id]
    worlds=[evaluate_world(i) for i in ids]

    outdir=Path("results/world_c_lite100")
    outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/f"world_c_lite_shard_{args.shard_id:03d}.json"
    out.write_text(json.dumps({
        "experiment":NAMESPACE,
        "pre_fixed_condition":"N_plus >= 4 AND delta_min >= 0",
        "world_count":WORLD_COUNT,
        "worlds":worlds
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

if __name__=="__main__":
    main()
