#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

WORLD_COUNT=100
EVAL_PER_WORLD=596
THRESHOLDS=(3,4,5)


def zscore(obs,exp,var):
    return (obs-exp)/math.sqrt(var) if var>0 else None


def main():
    files=sorted(Path("artifacts").glob("**/shard_*.json"))
    worlds=[]
    meta=None
    for f in files:
        p=json.loads(f.read_text(encoding="utf-8"))
        meta=meta or p["world_source"]
        worlds.extend(p["worlds"])
    worlds.sort(key=lambda x:x["world_id"])
    if len(worlds)!=WORLD_COUNT or [w["world_id"] for w in worlds] != list(range(1,WORLD_COUNT+1)):
        raise RuntimeError(f"expected worlds 1..{WORLD_COUNT}, got {len(worlds)}")

    summary={
        "experiment":"generation_freedom_hybrid10_10_outer_lite100",
        "world_source":meta,
        "world_count":WORLD_COUNT,
        "total_evaluation_rounds":WORLD_COUNT*EVAL_PER_WORLD,
        "thresholds":{},
        "boundary":[
            "10+10 was fixed before reading these outer-world results.",
            "Aggregate comparison uses observed Hybrid10+10 success rounds versus exact random expectations.",
            "Z values use the summed Bernoulli variance as a descriptive standardized difference; they are not promoted as standalone proof of real-lottery predictive skill.",
            "World C Lite100 is a null/random-world family, not future real OOS."
        ],
    }

    for th in THRESHOLDS:
        k=str(th)
        obs=sum(w["observed_success_rounds"][k] for w in worlds)
        fexp=sum(w["full_random_expected"][k] for w in worlds)
        fvar=sum(w["full_random_variance"][k] for w in worlds)
        cexp=sum(w["same_core18_random_expected"][k] for w in worlds)
        cvar=sum(w["same_core18_random_variance"][k] for w in worlds)
        summary["thresholds"][f"{th}plus"]={
            "hybrid_success_rounds":obs,
            "hybrid_rate":obs/(WORLD_COUNT*EVAL_PER_WORLD),
            "full_space_random20":{
                "expected_success_rounds":fexp,
                "expected_rate":fexp/(WORLD_COUNT*EVAL_PER_WORLD),
                "standardized_delta_z":zscore(obs,fexp,fvar),
            },
            "same_core18_random20":{
                "expected_success_rounds":cexp,
                "expected_rate":cexp/(WORLD_COUNT*EVAL_PER_WORLD),
                "standardized_delta_z":zscore(obs,cexp,cvar),
            },
            "worlds_hybrid_above_full_expectation":sum(
                w["observed_success_rounds"][k] > w["full_random_expected"][k] for w in worlds
            ),
            "worlds_hybrid_above_same_core18_expectation":sum(
                w["observed_success_rounds"][k] > w["same_core18_random_expected"][k] for w in worlds
            ),
        }

    outdir=Path("results/hybrid10_10_outer_lite100")
    outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/"summary.json"
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
