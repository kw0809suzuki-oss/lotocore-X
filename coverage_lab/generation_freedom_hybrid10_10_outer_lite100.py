#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K, A_TICKETS,
    main_numbers, unique_structured_bundle, portfolio_bundle, bundle_metrics,
)
from coverage_lab.loto7_hybrid_ratio_sweep_v0 import select, merge_unique
from coverage_lab.generation_freedom_hybrid10_10_vs_random_world_a import (
    favorable_ticket_count, unique_random20_success_probability,
)

WORLD_COUNT = 100
DRAWS_PER_WORLD = 696
BASE_SEED = 20261004
NAMESPACE = "loto7-world-c-lite100-v0"
OUT_DIR = Path("results/hybrid10_10_outer_lite100")
STRUCTURED_N = 10
FREEDOM_N = 10
THRESHOLDS = (3,4,5)


def world_seed(world_id:int)->int:
    raw=f"{BASE_SEED}:{NAMESPACE}:{world_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def make_world(world_id:int)->pd.DataFrame:
    rng=random.Random(world_seed(world_id))
    rows=[]
    for rnd in range(1,DRAWS_PER_WORLD+1):
        nums=sorted(rng.sample(range(1,38),7))
        rows.append({"round":rnd, **{f"n{i+1}":n for i,n in enumerate(nums)}})
    return pd.DataFrame(rows)


def evaluate_world(world_id:int)->dict:
    df=make_world(world_id)
    observed={k:0 for k in THRESHOLDS}
    full_expected={k:0.0 for k in THRESHOLDS}
    full_var={k:0.0 for k in THRESHOLDS}
    core_expected={k:0.0 for k in THRESHOLDS}
    core_var={k:0.0 for k in THRESHOLDS}

    for idx in range(MODEL_WINDOW,len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]
        target=df.iloc[idx]
        rnd=int(target["round"])
        actual=main_numbers(target)

        snap=lotocore.score_snapshot(history)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        core18=sorted(range(1,38), key=lambda n:(ranks[n],n))[:POOL_K]

        seed=rnd*100_003+20261002
        structured20,_=unique_structured_bundle(core18,A_TICKETS,seed+1)
        freedom20,_=portfolio_bundle(core18,scores,seed+2)
        hybrid20,_=merge_unique(
            select(structured20,STRUCTURED_N),
            select(freedom20,FREEDOM_N),
            freedom20,
        )
        hm=bundle_metrics(hybrid20,actual)
        h_core=len(set(core18)&actual)

        for threshold in THRESHOLDS:
            observed[threshold]+=int(hm["max_hits"]>=threshold)

            full_fav=favorable_ticket_count(37,7,threshold)
            p_full=unique_random20_success_probability(37,full_fav)
            full_expected[threshold]+=p_full
            full_var[threshold]+=p_full*(1-p_full)

            core_fav=favorable_ticket_count(18,h_core,threshold)
            p_core=unique_random20_success_probability(18,core_fav)
            core_expected[threshold]+=p_core
            core_var[threshold]+=p_core*(1-p_core)

    return {
        "world_id":world_id,
        "world_seed":world_seed(world_id),
        "observed_success_rounds":{str(k):observed[k] for k in THRESHOLDS},
        "full_random_expected":{str(k):full_expected[k] for k in THRESHOLDS},
        "full_random_variance":{str(k):full_var[k] for k in THRESHOLDS},
        "same_core18_random_expected":{str(k):core_expected[k] for k in THRESHOLDS},
        "same_core18_random_variance":{str(k):core_var[k] for k in THRESHOLDS},
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-id",type=int,required=True)
    ap.add_argument("--num-shards",type=int,required=True)
    args=ap.parse_args()

    ids=[i for i in range(1,WORLD_COUNT+1) if (i-1)%args.num_shards==args.shard_id]
    worlds=[]
    for wid in ids:
        print(f"WORLD {wid}/{WORLD_COUNT}",flush=True)
        worlds.append(evaluate_world(wid))

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    out=OUT_DIR/f"shard_{args.shard_id:03d}.json"
    payload={
        "experiment":"generation_freedom_hybrid10_10_outer_lite100",
        "world_source":{
            "namespace":NAMESPACE,
            "base_seed":BASE_SEED,
            "world_count":WORLD_COUNT,
            "draws_per_world":DRAWS_PER_WORLD,
            "evaluation_rounds_per_world":DRAWS_PER_WORLD-MODEL_WINDOW,
            "note":"Reuses the pre-existing deterministic World C Lite100 family."
        },
        "hybrid":{
            "structured_n":STRUCTURED_N,
            "freedom_n":FREEDOM_N,
            "selection_rule":"Existing spread-index selection + existing collision repair.",
        },
        "boundary":[
            "10+10 is fixed before reading these outer-world results.",
            "No ratio search, threshold search, seed tuning, or generator tuning is performed.",
            "Random baselines use exact per-round combinatorial expectations.",
            "These null/random worlds test whether equal-budget complementarity reproduces outside World A; they do not establish predictive skill in the real lottery."
        ],
        "shard":{"id":args.shard_id,"num_shards":args.num_shards},
        "worlds":worlds,
    }
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("saved ->",out)


if __name__=="__main__":
    main()
