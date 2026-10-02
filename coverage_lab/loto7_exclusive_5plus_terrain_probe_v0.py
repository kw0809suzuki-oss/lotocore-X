#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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


def hitset(ticket, actual):
    return tuple(sorted(set(ticket) & actual))


def shared(a, b):
    return len(set(a) & set(b))


def components(rows, min_shared=3):
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
                if j in seen:
                    continue
                if shared(sx, tuple(rows[j]["hit_set"])) >= min_shared:
                    seen.add(j); stack.append(j)
        comps.append(comp)
    return comps


def observe(tickets, actual):
    all_rows=[]
    for i,t in enumerate(tickets,1):
        hs=hitset(t,actual)
        all_rows.append({
            "ticket_id":i,
            "hit_count":len(hs),
            "hit_set":list(hs),
        })

    rows4=[r for r in all_rows if r["hit_count"]>=4]
    rows5=[r for r in all_rows if r["hit_count"]>=5]
    comps=components(rows4,3) if rows4 else []

    comp_summaries=[]
    for comp in comps:
        rr=[rows4[i] for i in comp]
        comp_summaries.append({
            "size":len(rr),
            "n_5plus":sum(r["hit_count"]>=5 for r in rr),
            "n_exact4":sum(r["hit_count"]==4 for r in rr),
            "ticket_ids":[r["ticket_id"] for r in rr],
        })

    uniq5={tuple(r["hit_set"]) for r in rows5}

    return {
        "max_hit":max(r["hit_count"] for r in all_rows),
        "n_4plus":len(rows4),
        "n_exact4":sum(r["hit_count"]==4 for r in rows4),
        "n_5plus":len(rows5),
        "cluster_count_4plus_shared3":len(comps),
        "max_cluster_depth":max((len(c) for c in comps),default=0),
        "max_cluster_5plus":max((sum(rows4[i]["hit_count"]>=5 for i in c) for c in comps),default=0),
        "unique_5plus_structures":len(uniq5),
        "independent_5plus_rate":(len(uniq5)/len(rows5) if rows5 else None),
        "components":comp_summaries,
        "rows_4plus":rows4,
    }


def main():
    data=Path("coverage_lab/fixtures/loto7_1_696.csv")
    df=pd.read_csv(data).sort_values("round").reset_index(drop=True)

    rounds=[]
    for idx in range(MODEL_WINDOW,len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]
        target=df.iloc[idx]
        rnd=int(target["round"])
        actual=main_numbers(target)

        snap=lotocore.score_snapshot(history)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        ranked=sorted(range(1,38),key=lambda n:(ranks[n],n))
        pool=ranked[:POOL_K]

        seed=rnd*100_003+20261002
        a,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
        b,_=portfolio_bundle(pool,scores,seed+2)

        oa=observe(a,actual)
        ob=observe(b,actual)

        a5=oa["n_5plus"]>0
        b5=ob["n_5plus"]>0
        if a5==b5:
            continue

        winner="structured20_only" if a5 else "portfolio20_only"
        rounds.append({
            "round":rnd,
            "winner":winner,
            "actual":sorted(actual),
            "structured20":oa,
            "portfolio20":ob,
        })

    flat=[]
    for rec in rounds:
        for strategy in ("structured20","portfolio20"):
            o=rec[strategy]
            flat.append({
                "round":rec["round"],
                "winner":rec["winner"],
                "strategy":strategy,
                "max_hit":o["max_hit"],
                "n_4plus":o["n_4plus"],
                "n_exact4":o["n_exact4"],
                "n_5plus":o["n_5plus"],
                "cluster_count_4plus_shared3":o["cluster_count_4plus_shared3"],
                "max_cluster_depth":o["max_cluster_depth"],
                "max_cluster_5plus":o["max_cluster_5plus"],
                "unique_5plus_structures":o["unique_5plus_structures"],
                "independent_5plus_rate":o["independent_5plus_rate"],
            })

    outdf=pd.DataFrame(flat)
    summary={}
    for winner in ("structured20_only","portfolio20_only"):
        g=outdf[outdf.winner==winner]
        summary[winner]={}
        for strategy in ("structured20","portfolio20"):
            s=g[g.strategy==strategy]
            summary[winner][strategy]={
                "rounds":int(len(s)),
                "mean_max_hit":float(s.max_hit.mean()) if len(s) else None,
                "mean_n_4plus":float(s.n_4plus.mean()) if len(s) else None,
                "mean_cluster_count":float(s.cluster_count_4plus_shared3.mean()) if len(s) else None,
                "mean_max_cluster_depth":float(s.max_cluster_depth.mean()) if len(s) else None,
                "mean_unique_5plus_structures":float(s.unique_5plus_structures.mean()) if len(s) else None,
                "mean_independent_5plus_rate":float(s.independent_5plus_rate.dropna().mean()) if s.independent_5plus_rate.notna().any() else None,
            }

    payload={
        "experiment":"loto7_exclusive_5plus_terrain_probe_v0",
        "exclusive_round_count":len(rounds),
        "summary":summary,
        "rounds":rounds,
        "boundary":[
            "Frozen Structured20 and Portfolio20 generators are replayed unchanged.",
            "Exclusive rounds are selected only by realized 5+ presence: one strategy has >=1 5+ ticket and the other has none.",
            "The round-412 terrain ruler is reused unchanged: 4+ tickets are connected when they share >=3 winning numbers.",
            "Depth and independence are observations only; no causal or predictive claim is made.",
        ],
    }

    csv=Path("results/loto7_exclusive_5plus_terrain_probe_v0.csv")
    js=Path("results/loto7_exclusive_5plus_terrain_probe_v0.json")
    csv.parent.mkdir(parents=True,exist_ok=True)
    outdf.to_csv(csv,index=False)
    js.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== EXCLUSIVE 5+ TERRAIN PROBE v0 ===")
    print("exclusive_round_count",len(rounds))
    print("summary",json.dumps(summary,ensure_ascii=False,sort_keys=True))
    print("rows",json.dumps(flat,ensure_ascii=False,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print(f"saved -> {csv}")
    print(f"saved -> {js}")


if __name__=="__main__":
    main()
