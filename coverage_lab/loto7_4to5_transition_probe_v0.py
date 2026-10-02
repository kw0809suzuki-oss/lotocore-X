#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    A_TICKETS, MODEL_WINDOW, POOL_K,
    main_numbers, portfolio_bundle, unique_structured_bundle,
)

TARGET_ROUNDS=(374,518)


def analyze_round(df, rnd):
    idx=df.index[df["round"]==rnd][0]
    history=df.iloc[idx-MODEL_WINDOW:idx]
    target=df.iloc[idx]
    actual=main_numbers(target)

    snap=lotocore.score_snapshot(history)
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    ranked=sorted(range(1,38),key=lambda n:(ranks[n],n))
    pool=ranked[:POOL_K]

    seed=rnd*100_003+20261002
    structured,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
    portfolio,_=portfolio_bundle(pool,scores,seed+2)

    s4=[]
    for i,t in enumerate(structured,1):
        hit=set(t)&actual
        if len(hit)==4:
            s4.append({
                "ticket_id":i,
                "ticket":list(t),
                "hit_set":sorted(hit),
                "missing_actual":sorted(actual-hit),
                "nonwinning_in_ticket":sorted(set(t)-actual),
            })

    p5=[]
    for i,t in enumerate(portfolio,1):
        hit=set(t)&actual
        if len(hit)>=5:
            p5.append({
                "ticket_id":i,
                "ticket":list(t),
                "hit_count":len(hit),
                "hit_set":sorted(hit),
                "missing_actual":sorted(actual-hit),
                "nonwinning_in_ticket":sorted(set(t)-actual),
            })

    missing_freq={n:sum(n in r["missing_actual"] for r in s4) for n in sorted(actual)}
    common_missing=[n for n,c in missing_freq.items() if c==len(s4) and len(s4)>0]

    transitions=[]
    for sr in s4:
        st=set(sr["ticket"])
        for pr in p5:
            pt=set(pr["ticket"])
            removed=sorted(st-pt)
            added=sorted(pt-st)
            transitions.append({
                "structured_ticket_id":sr["ticket_id"],
                "portfolio_ticket_id":pr["ticket_id"],
                "structured_hits":len(sr["hit_set"]),
                "portfolio_hits":pr["hit_count"],
                "removed":removed,
                "added":added,
                "removed_winning":sorted(set(removed)&actual),
                "removed_nonwinning":sorted(set(removed)-actual),
                "added_winning":sorted(set(added)&actual),
                "added_nonwinning":sorted(set(added)-actual),
                "edit_size":len(removed)+len(added),
                "net_hit_gain":pr["hit_count"]-len(sr["hit_set"]),
            })

    valid=[x for x in transitions if x["net_hit_gain"]>=1]
    valid.sort(key=lambda x:(
        x["edit_size"],
        -len(x["added_winning"]),
        len(x["removed_winning"]),
        x["structured_ticket_id"],
        x["portfolio_ticket_id"],
    ))

    return {
        "round":rnd,
        "actual":sorted(actual),
        "candidate_pool":pool,
        "structured_exact4_count":len(s4),
        "portfolio_5plus_count":len(p5),
        "structured_4plus":s4,
        "portfolio_5plus":p5,
        "missing_actual_frequency_in_structured4":missing_freq,
        "common_missing_actual":common_missing,
        "minimal_transitions":valid[:10],
    }


def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    payload={
        "experiment":"loto7_4to5_transition_probe_v0",
        "rounds":[analyze_round(df,r) for r in TARGET_ROUNDS],
        "boundary":[
            "Only rounds 374 and 518 are observed.",
            "Frozen Structured20 and Portfolio20 generators are replayed unchanged.",
            "No new scoring or optimization is introduced.",
            "The probe records missing winning numbers and minimum ticket-set substitutions only.",
            "A repeated pattern across the two rounds is a candidate, not evidence of generality.",
        ],
    }
    out=Path("results/loto7_4to5_transition_probe_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== LOTO7 4->5 TRANSITION PROBE v0 ===")
    for r in payload["rounds"]:
        print("ROUND",r["round"])
        print("ACTUAL",r["actual"])
        print("COMMON_MISSING",r["common_missing_actual"])
        print("MISSING_FREQ",json.dumps(r["missing_actual_frequency_in_structured4"],ensure_ascii=False,sort_keys=True))
        print("PORTFOLIO_5PLUS",json.dumps(r["portfolio_5plus"],ensure_ascii=False,sort_keys=True))
        print("MIN_TRANSITIONS",json.dumps(r["minimal_transitions"],ensure_ascii=False,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print(f"saved -> {out}")


if __name__=="__main__":
    main()
