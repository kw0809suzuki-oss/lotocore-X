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


def replacement_distance(a, b):
    sa, sb = set(a), set(b)
    return len(sa - sb)


def observe_round(df, idx):
    history=df.iloc[idx-MODEL_WINDOW:idx]
    target=df.iloc[idx]
    rnd=int(target["round"])
    actual=main_numbers(target)

    snap=lotocore.score_snapshot(history)
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    ranked=sorted(range(1,38), key=lambda n:(ranks[n],n))
    pool=ranked[:POOL_K]

    seed=rnd*100_003+20261002
    structured,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
    portfolio,_=portfolio_bundle(pool,scores,seed+2)

    def classify(tickets):
        out=[]
        for i,t in enumerate(tickets,1):
            hit=len(set(t)&actual)
            out.append({"ticket_id":i,"ticket":list(t),"hit_count":hit})
        return out

    a=classify(structured)
    b=classify(portfolio)

    a5=[x for x in a if x["hit_count"]>=5]
    b5=[x for x in b if x["hit_count"]>=5]
    if bool(a5)==bool(b5):
        return None

    if a5:
        winner="structured20"
        win5=a5
        loser="portfolio20"
        lose4=[x for x in b if x["hit_count"]==4]
    else:
        winner="portfolio20"
        win5=b5
        loser="structured20"
        lose4=[x for x in a if x["hit_count"]==4]

    pairs=[]
    for l in lose4:
        for w in win5:
            d=replacement_distance(l["ticket"],w["ticket"])
            pairs.append({
                "loser_ticket_id":l["ticket_id"],
                "winner_ticket_id":w["ticket_id"],
                "distance":d,
                "loser_ticket":l["ticket"],
                "winner_ticket":w["ticket"],
            })

    min_d=min((p["distance"] for p in pairs), default=None)
    nearest=[p for p in pairs if p["distance"]==min_d] if min_d is not None else []

    return {
        "round":rnd,
        "winner":winner,
        "loser":loser,
        "winner_5plus_count":len(win5),
        "loser_exact4_count":len(lose4),
        "min_replacement_distance":min_d,
        "nearest_pairs":nearest,
    }


def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    rows=[]
    for idx in range(MODEL_WINDOW,len(df)):
        rec=observe_round(df,idx)
        if rec is not None:
            rows.append(rec)

    dist_counts={}
    for r in rows:
        key="none" if r["min_replacement_distance"] is None else str(r["min_replacement_distance"])
        dist_counts[key]=dist_counts.get(key,0)+1

    by_winner={}
    for winner in ("structured20","portfolio20"):
        vals=[r["min_replacement_distance"] for r in rows if r["winner"]==winner and r["min_replacement_distance"] is not None]
        by_winner[winner]={
            "rounds":sum(r["winner"]==winner for r in rows),
            "measured_rounds":len(vals),
            "distances":vals,
            "mean_distance":sum(vals)/len(vals) if vals else None,
            "median_distance":sorted(vals)[len(vals)//2] if vals else None,
        }

    payload={
        "experiment":"loto7_exclusive_5plus_min_edit_distance_v0",
        "definition":"replacement distance = number of numbers removed from a losing-side exact-4 ticket to reach a winning-side 5+ ticket; both tickets have seven numbers, so removals equal additions.",
        "exclusive_round_count":len(rows),
        "distance_counts":dist_counts,
        "by_winner":by_winner,
        "rounds":rows,
        "boundary":[
            "Frozen Structured20 and Portfolio20 generators are replayed unchanged.",
            "Only exclusive 5+ rounds are included.",
            "Distance is measured only when the losing side has at least one exact-4 ticket.",
            "No continuity/jump labels or causal interpretation are assigned.",
        ],
    }

    out=Path("results/loto7_exclusive_5plus_min_edit_distance_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== EXCLUSIVE 5+ MIN EDIT DISTANCE v0 ===")
    print("COUNT",len(rows))
    print("DIST",json.dumps(dist_counts,sort_keys=True))
    print("BY_WINNER",json.dumps(by_winner,ensure_ascii=False,sort_keys=True))
    print("ROUNDS",json.dumps([
        {
            "round":r["round"],
            "winner":r["winner"],
            "loser_exact4_count":r["loser_exact4_count"],
            "winner_5plus_count":r["winner_5plus_count"],
            "min_replacement_distance":r["min_replacement_distance"],
        } for r in rows
    ],ensure_ascii=False,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print(f"saved -> {out}")


if __name__=="__main__":
    main()
