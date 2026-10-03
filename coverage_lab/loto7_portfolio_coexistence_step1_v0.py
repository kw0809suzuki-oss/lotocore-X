#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K,
    main_numbers, portfolio_bundle, bundle_metrics,
)

EVEN = list(range(0, 20, 2))
ODD = list(range(1, 20, 2))

def merge_unique(a_pick, b_pick, b20):
    chosen=list(a_pick)
    used=set(chosen)
    preferred_set=set(b_pick)
    leftovers=[t for t in b20 if t not in preferred_set]
    repairs=0
    for t in b_pick:
        if t not in used:
            pick=t
        else:
            repairs += 1
            pick=next((x for x in leftovers if x not in used), None)
            if pick is None:
                raise RuntimeError("no unused B20 replacement")
            leftovers.remove(pick)
        chosen.append(pick)
        used.add(pick)
    if len(chosen)!=20 or len(set(chosen))!=20:
        raise RuntimeError("mixed bundle is not 20 unique tickets")
    return chosen, repairs

def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    a_success=set()
    b_success=set()
    base_success=set()
    cand_success=set()
    base_repairs=0
    cand_repairs=0

    for idx in range(MODEL_WINDOW, len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]
        target=df.iloc[idx]
        rnd=int(target["round"])
        actual=main_numbers(target)

        snap=lotocore.score_snapshot(history)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        pool=sorted(range(1,38), key=lambda n:(ranks[n],n))[:POOL_K]

        base_seed=rnd*100_003+20261002
        a20,_=portfolio_bundle(pool,scores,base_seed+2)
        b20,_=portfolio_bundle(pool,scores,base_seed+102)

        base,rep0=merge_unique([a20[i] for i in EVEN],[b20[i] for i in EVEN],b20)
        cand,rep1=merge_unique([a20[i] for i in EVEN],[b20[i] for i in ODD],b20)
        base_repairs += rep0
        cand_repairs += rep1

        if bundle_metrics(a20,actual)["tickets_ge5"]>0:
            a_success.add(rnd)
        if bundle_metrics(b20,actual)["tickets_ge5"]>0:
            b_success.add(rnd)
        if bundle_metrics(base,actual)["tickets_ge5"]>0:
            base_success.add(rnd)
        if bundle_metrics(cand,actual)["tickets_ge5"]>0:
            cand_success.add(rnd)

    payload={
        "experiment":"loto7_portfolio_coexistence_step1_v0",
        "source_success_sets":{
            "a20":sorted(a_success),
            "b20":sorted(b_success),
            "a_only":sorted(a_success-b_success),
            "b_only":sorted(b_success-a_success),
            "both":sorted(a_success&b_success),
            "union":sorted(a_success|b_success),
        },
        "baseline_even_even":{
            "fiveplus_rounds":len(base_success),
            "success_rounds":sorted(base_success),
            "repairs":base_repairs,
        },
        "candidate_even_odd":{
            "fiveplus_rounds":len(cand_success),
            "success_rounds":sorted(cand_success),
            "repairs":cand_repairs,
        },
        "candidate_vs_baseline":{
            "added":sorted(cand_success-base_success),
            "lost":sorted(base_success-cand_success),
            "net":len(cand_success)-len(base_success),
        },
        "candidate_preservation":{
            "a_only_kept":sorted((a_success-b_success)&cand_success),
            "b_only_kept":sorted((b_success-a_success)&cand_success),
            "both_kept":sorted((a_success&b_success)&cand_success),
        },
        "boundary":[
            "Same Portfolio A and Portfolio B source worlds as prior probe.",
            "Total budget fixed at 20 and ratio fixed at 10+10.",
            "A subset fixed to even positions.",
            "Only B subset changes from even to odd positions.",
            "No realized draw information enters selection or collision repair.",
            "One coexistence probe only; no tuning from outcomes."
        ]
    }

    out=Path("results/loto7_portfolio_coexistence_step1_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("A_ONLY",sorted(a_success-b_success))
    print("B_ONLY",sorted(b_success-a_success))
    print("BASE",len(base_success),sorted(base_success))
    print("CAND",len(cand_success),sorted(cand_success))
    print("DELTA",json.dumps(payload["candidate_vs_baseline"],ensure_ascii=False,sort_keys=True))
    print("KEEP",json.dumps(payload["candidate_preservation"],ensure_ascii=False,sort_keys=True))
    print("saved ->",out)

if __name__=="__main__":
    main()
