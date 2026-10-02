#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lotocore
from coverage_lab.loto7_portfolio20_v0 import MODEL_WINDOW, POOL_K, A_TICKETS, main_numbers, unique_structured_bundle, portfolio_bundle, bundle_metrics

A_IDX=[0,2,4,6,8,10,12,14,16,18]
B_IDX=[0,2,4,6,8,10,12,14,16,18]

def fixed_half(bundle, idxs):
    return [bundle[i] for i in idxs]

def merge_unique(a10,b20):
    # Prefer the fixed B indices; on cross-arm collision, use the next unused
    # B ticket in original B20 order. No outcome data is used.
    chosen=list(a10); used=set(chosen); repairs=0
    preferred=fixed_half(b20,B_IDX)
    leftovers=[t for i,t in enumerate(b20) if i not in B_IDX]
    for t in preferred:
        if t not in used:
            pick=t
        else:
            repairs+=1
            pick=next((x for x in leftovers if x not in used),None)
            if pick is None: raise RuntimeError("no unique B replacement")
            leftovers.remove(pick)
        chosen.append(pick); used.add(pick)
    if len(chosen)!=20 or len(set(chosen))!=20: raise RuntimeError("hybrid not 20 unique")
    return chosen,repairs

def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    rows=[]; collision_repairs=0
    exclusive_recovery={"structured_only_total":0,"structured_only_recovered":0,"portfolio_only_total":0,"portfolio_only_recovered":0}
    for idx in range(MODEL_WINDOW,len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]; target=df.iloc[idx]
        rnd=int(target["round"]); actual=main_numbers(target)
        snap=lotocore.score_snapshot(history)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        pool=sorted(range(1,38),key=lambda n:(ranks[n],n))[:POOL_K]
        seed=rnd*100_003+20261002
        a20,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
        b20,_=portfolio_bundle(pool,scores,seed+2)
        h20,rep=merge_unique(fixed_half(a20,A_IDX),b20); collision_repairs+=rep

        ma=bundle_metrics(a20,actual); mb=bundle_metrics(b20,actual); mh=bundle_metrics(h20,actual)
        a5=ma["tickets_ge5"]>0; b5=mb["tickets_ge5"]>0; h5=mh["tickets_ge5"]>0
        if a5 and not b5:
            exclusive_recovery["structured_only_total"]+=1
            exclusive_recovery["structured_only_recovered"]+=int(h5)
        if b5 and not a5:
            exclusive_recovery["portfolio_only_total"]+=1
            exclusive_recovery["portfolio_only_recovered"]+=int(h5)

        rows.append({
          "round":rnd,
          "structured_max":ma["max_hits"],"portfolio_max":mb["max_hits"],"hybrid_max":mh["max_hits"],
          "structured_5plus":int(a5),"portfolio_5plus":int(b5),"hybrid_5plus":int(h5),
          "hybrid_tickets_ge4":mh["tickets_ge4"],"hybrid_tickets_ge5":mh["tickets_ge5"],
          "collision_repairs":rep
        })

    n=len(rows)
    counts={
      "structured_5plus_rounds":sum(r["structured_5plus"] for r in rows),
      "portfolio_5plus_rounds":sum(r["portfolio_5plus"] for r in rows),
      "hybrid_5plus_rounds":sum(r["hybrid_5plus"] for r in rows),
      "hybrid_p_max_ge5":sum(r["hybrid_5plus"] for r in rows)/n,
      "hybrid_mean_max_hits":sum(r["hybrid_max"] for r in rows)/n,
      "hybrid_total_tickets_ge5":sum(r["hybrid_tickets_ge5"] for r in rows),
      "collision_repairs_total":collision_repairs,
    }
    payload={
      "experiment":"loto7_naive_parallel_hybrid20_v0",
      "selection":{
        "structured_arm":"fixed even positions 0,2,...,18 from frozen Structured20",
        "portfolio_arm":"fixed even positions 0,2,...,18 from frozen Portfolio20; cross-arm duplicate replaced by next unused Portfolio20 ticket in original order",
        "ticket_budget":20
      },
      "counts":counts,
      "exclusive_recovery":exclusive_recovery,
      "rounds":rows,
      "boundary":[
        "Hybrid is a fixed 10+10 mechanical merge; no ratio optimization is performed.",
        "The underlying Structured20 and Portfolio20 generators are unchanged.",
        "Selection uses no realized draw information.",
        "A result here tests whether naive 20-ticket coexistence recovers non-overlapping successes; it does not explain why."
      ]
    }
    out=Path("results/loto7_naive_parallel_hybrid20_v0.json"); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("=== NAIVE PARALLEL HYBRID20 v0 ===")
    print("COUNTS",json.dumps(counts,sort_keys=True))
    print("RECOVERY",json.dumps(exclusive_recovery,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print("saved ->",out)
if __name__=="__main__": main()
