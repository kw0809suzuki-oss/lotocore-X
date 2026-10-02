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

TARGET_ROUND = 412


def hitset(ticket, actual):
    return tuple(sorted(set(ticket) & actual))


def overlap(a, b):
    return len(set(a) & set(b))


def analyze(strategy, tickets, actual):
    rows = []
    for i, t in enumerate(tickets, 1):
        hs = hitset(t, actual)
        if len(hs) >= 4:
            rows.append({
                "ticket_id": i,
                "ticket": list(t),
                "hit_count": len(hs),
                "hit_set": list(hs),
            })

    high = [r for r in rows if r["hit_count"] >= 5]
    four = [r for r in rows if r["hit_count"] == 4]

    # Neighborhood of each 5+ ticket in realized winning-number space.
    neighborhoods = []
    for h in high:
        hset = tuple(h["hit_set"])
        neigh = []
        for r in rows:
            if r["ticket_id"] == h["ticket_id"]:
                continue
            shared = overlap(hset, tuple(r["hit_set"]))
            neigh.append({
                "ticket_id": r["ticket_id"],
                "hit_count": r["hit_count"],
                "shared_winning_numbers_with_center": shared,
                "hit_set": r["hit_set"],
            })
        neighborhoods.append({
            "center_ticket_id": h["ticket_id"],
            "center_hit_count": h["hit_count"],
            "center_hit_set": h["hit_set"],
            "neighbors_shared4plus": [x for x in neigh if x["shared_winning_numbers_with_center"] >= 4],
            "neighbors_shared3plus": [x for x in neigh if x["shared_winning_numbers_with_center"] >= 3],
        })

    # Connected components among all 4+/5+ tickets using >=3 shared winning numbers.
    n=len(rows)
    seen=set()
    comps=[]
    for i in range(n):
        if i in seen:
            continue
        stack=[i]; seen.add(i); comp=[]
        while stack:
            x=stack.pop(); comp.append(x)
            sx=tuple(rows[x]["hit_set"])
            for j in range(n):
                if j in seen: continue
                if overlap(sx, tuple(rows[j]["hit_set"])) >= 3:
                    seen.add(j); stack.append(j)
        comps.append(comp)

    component_summaries=[]
    for comp in comps:
        rr=[rows[i] for i in comp]
        component_summaries.append({
            "size": len(rr),
            "n_5plus": sum(r["hit_count"]>=5 for r in rr),
            "n_4": sum(r["hit_count"]==4 for r in rr),
            "ticket_ids": [r["ticket_id"] for r in rr],
            "hit_sets": [r["hit_set"] for r in rr],
        })

    return {
        "strategy": strategy,
        "n_4plus": len(rows),
        "n_exact4": len(four),
        "n_5plus": len(high),
        "rows": rows,
        "neighborhoods": neighborhoods,
        "components_shared3": component_summaries,
    }


def main():
    data=Path("coverage_lab/fixtures/loto7_1_696.csv")
    df=pd.read_csv(data).sort_values("round").reset_index(drop=True)
    idx=df.index[df["round"]==TARGET_ROUND][0]
    history=df.iloc[idx-MODEL_WINDOW:idx]
    target=df.iloc[idx]
    actual=main_numbers(target)

    snap=lotocore.score_snapshot(history)
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    ranked=sorted(range(1,38), key=lambda n:(ranks[n],n))
    pool=ranked[:POOL_K]

    seed=TARGET_ROUND*100_003+20261002
    a,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
    b,_=portfolio_bundle(pool,scores,seed+2)

    payload={
        "round":TARGET_ROUND,
        "actual":sorted(actual),
        "candidate_pool":pool,
        "structured20": analyze("structured20",a,actual),
        "portfolio20": analyze("portfolio20",b,actual),
        "boundary":[
            "Frozen generators replayed exactly as in Portfolio20 v0.",
            "Only realized 4+/5+ local geometry in round 412 is observed.",
            "No Depth rule is inferred from this single round.",
        ],
    }

    out=Path("results/loto7_round412_depth_probe_v0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== ROUND 412 DEPTH PROBE v0 ===")
    for key in ("structured20","portfolio20"):
        x=payload[key]
        print(key, json.dumps({
            "n_4plus":x["n_4plus"],
            "n_exact4":x["n_exact4"],
            "n_5plus":x["n_5plus"],
            "components_shared3":x["components_shared3"],
            "neighborhoods":x["neighborhoods"],
        },ensure_ascii=False,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print(f"saved -> {out}")


if __name__=="__main__":
    main()
