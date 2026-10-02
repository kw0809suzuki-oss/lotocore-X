#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K, A_TICKETS,
    main_numbers, unique_structured_bundle, portfolio_bundle, bundle_metrics,
)

RATIOS=((5,15),(8,12),(10,10),(12,8),(15,5))

def spread_indices(n, total=20):
    return [int(i*total/n) for i in range(n)]

def select(bundle,n):
    return [bundle[i] for i in spread_indices(n,len(bundle))]

def merge_unique(a_pick,b_pick,b20):
    chosen=list(a_pick); used=set(chosen); repairs=0
    preferred=list(b_pick)
    preferred_set=set(preferred)
    leftovers=[t for t in b20 if t not in preferred_set]
    for t in preferred:
        if t not in used:
            pick=t
        else:
            repairs+=1
            pick=next((x for x in leftovers if x not in used),None)
            if pick is None:
                raise RuntimeError("no unused Portfolio20 replacement")
            leftovers.remove(pick)
        chosen.append(pick); used.add(pick)
    if len(chosen)!=20 or len(set(chosen))!=20:
        raise RuntimeError("hybrid is not 20 unique tickets")
    return chosen,repairs

def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)

    stats={
        f"{a}+{b}":{
            "structured_n":a,"portfolio_n":b,
            "fiveplus_rounds":0,
            "exclusive_recovered":0,
            "structured_only_recovered":0,
            "portfolio_only_recovered":0,
            "collision_repairs":0,
        } for a,b in RATIOS
    }
    structured_only_total=0
    portfolio_only_total=0
    base_s=0
    base_p=0

    for idx in range(MODEL_WINDOW,len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]
        target=df.iloc[idx]
        rnd=int(target["round"])
        actual=main_numbers(target)

        snap=lotocore.score_snapshot(history)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        pool=sorted(range(1,38),key=lambda n:(ranks[n],n))[:POOL_K]

        seed=rnd*100_003+20261002
        a20,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
        b20,_=portfolio_bundle(pool,scores,seed+2)

        ma=bundle_metrics(a20,actual)
        mb=bundle_metrics(b20,actual)
        a5=ma["tickets_ge5"]>0
        b5=mb["tickets_ge5"]>0
        base_s+=int(a5)
        base_p+=int(b5)

        s_only=a5 and not b5
        p_only=b5 and not a5
        structured_only_total+=int(s_only)
        portfolio_only_total+=int(p_only)

        for a_n,b_n in RATIOS:
            key=f"{a_n}+{b_n}"
            h20,rep=merge_unique(select(a20,a_n),select(b20,b_n),b20)
            mh=bundle_metrics(h20,actual)
            h5=mh["tickets_ge5"]>0
            stats[key]["fiveplus_rounds"]+=int(h5)
            stats[key]["collision_repairs"]+=rep
            if s_only and h5:
                stats[key]["structured_only_recovered"]+=1
                stats[key]["exclusive_recovered"]+=1
            if p_only and h5:
                stats[key]["portfolio_only_recovered"]+=1
                stats[key]["exclusive_recovered"]+=1

    payload={
        "experiment":"loto7_hybrid_ratio_sweep_v0",
        "baseline":{
            "structured20_5plus_rounds":base_s,
            "portfolio20_5plus_rounds":base_p,
            "structured_only_total":structured_only_total,
            "portfolio_only_total":portfolio_only_total,
            "exclusive_total":structured_only_total+portfolio_only_total,
        },
        "ratios":stats,
        "selection_rule":"For n tickets from a 20-ticket arm, use fixed spread indices floor(i*20/n), i=0..n-1. Cross-arm duplicates are replaced by the next unused Portfolio20 ticket in original order.",
        "boundary":[
            "Ratios were fixed before observing this sweep: 5+15, 8+12, 10+10, 12+8, 15+5.",
            "Total ticket budget is always 20.",
            "Underlying Structured20 and Portfolio20 generators are unchanged.",
            "No realized draw information enters ticket selection or collision repair.",
            "Primary observations are 5+ successful rounds and recovery of the 13 previously observed exclusive-success rounds.",
            "No ratio is promoted or tuned from this run alone."
        ]
    }

    out=Path("results/loto7_hybrid_ratio_sweep_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== HYBRID RATIO SWEEP v0 ===")
    print("BASELINE",json.dumps(payload["baseline"],sort_keys=True))
    print("RATIOS",json.dumps(stats,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print("saved ->",out)

if __name__=="__main__":
    main()
