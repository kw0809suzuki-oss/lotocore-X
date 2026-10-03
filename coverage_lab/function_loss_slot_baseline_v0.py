#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K, A_TICKETS,
    unique_structured_bundle, portfolio_bundle,
)
from coverage_lab.loto7_naive_parallel_hybrid20_v0 import A_IDX, merge_unique

EVENT_TICKETS = [
    {"round":217,"event":"lost","source":"structured20","index0":9},
    {"round":438,"event":"lost","source":"structured20","index0":7},
    {"round":676,"event":"lost","source":"structured20","index0":7},
    {"round":134,"event":"added","source":"hybrid20","index0":12,"role":"core"},
    {"round":249,"event":"added","source":"hybrid20","index0":18,"role":"falsify"},
    {"round":374,"event":"added","source":"hybrid20","index0":15,"role":"diversify"},
    {"round":413,"event":"added","source":"hybrid20","index0":16,"role":"diversify"},
    {"round":440,"event":"added","source":"hybrid20","index0":15,"role":"diversify"},
    {"round":518,"event":"added","source":"hybrid20","index0":14,"role":"core"},
]


def loss_counts(tickets):
    counters={}
    for k in (3,4):
        c=Counter()
        for t in tickets:
            c.update(itertools.combinations(t,k))
        counters[k]=c
    out=[]
    for i,t in enumerate(tickets):
        out.append({
            "index0":i,
            "lost_unique_triples":sum(counters[3][a]==1 for a in itertools.combinations(t,3)),
            "lost_unique_quadruples":sum(counters[4][a]==1 for a in itertools.combinations(t,4)),
        })
    return out


def pct_le(vals, x):
    return sum(v <= x for v in vals)/len(vals)


def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    rows=[]
    event_lookup={(e["round"],e["source"],e["index0"]):e for e in EVENT_TICKETS}

    for idx in range(MODEL_WINDOW,len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]
        rnd=int(df.iloc[idx]["round"])
        snap=lotocore.score_snapshot(history)
        ranks={int(k):int(v) for k,v in snap["ranks"].items()}
        scores={int(k):float(v) for k,v in snap["scores"].items()}
        pool=sorted(range(1,38),key=lambda n:(ranks[n],n))[:POOL_K]
        seed=rnd*100_003+20261002
        structured20,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
        portfolio20,trace=portfolio_bundle(pool,scores,seed+2)
        structured10=[structured20[i] for i in A_IDX]
        hybrid20,_=merge_unique(structured10,portfolio20)

        role_by_ticket={}
        for role in ("core","diversify","falsify"):
            for t in trace[role]:
                role_by_ticket[tuple(t)]=role

        for source,tickets in (("structured20",structured20),("hybrid20",hybrid20)):
            ls=loss_counts(tickets)
            for rec in ls:
                t=tickets[rec["index0"]]
                rec.update({
                    "round":rnd,
                    "source":source,
                    "role":role_by_ticket.get(tuple(t)) if source=="hybrid20" else None,
                })
                rows.append(rec)

    by_slot=defaultdict(list)
    by_role=defaultdict(list)
    by_source=defaultdict(list)
    for r in rows:
        by_slot[(r["source"],r["index0"])].append(r)
        by_source[r["source"]].append(r)
        if r["role"] is not None:
            by_role[(r["source"],r["role"])].append(r)

    event_results=[]
    for e in EVENT_TICKETS:
        rec=next(r for r in rows if (r["round"],r["source"],r["index0"])==(e["round"],e["source"],e["index0"]))
        slot=by_slot[(e["source"],e["index0"])]
        item={**e,
              "observed":{"lost_unique_triples":rec["lost_unique_triples"],"lost_unique_quadruples":rec["lost_unique_quadruples"]},
              "same_slot_baseline":{
                  "n":len(slot),
                  "triple_mean":statistics.fmean(x["lost_unique_triples"] for x in slot),
                  "quad_mean":statistics.fmean(x["lost_unique_quadruples"] for x in slot),
                  "triple_percentile_le":pct_le([x["lost_unique_triples"] for x in slot],rec["lost_unique_triples"]),
                  "quad_percentile_le":pct_le([x["lost_unique_quadruples"] for x in slot],rec["lost_unique_quadruples"]),
              }}
        role=rec["role"]
        item["observed"]["role"]=role
        if role is not None:
            rr=by_role[(e["source"],role)]
            item["same_role_baseline"]={
                "n":len(rr),
                "triple_mean":statistics.fmean(x["lost_unique_triples"] for x in rr),
                "quad_mean":statistics.fmean(x["lost_unique_quadruples"] for x in rr),
                "triple_percentile_le":pct_le([x["lost_unique_triples"] for x in rr],rec["lost_unique_triples"]),
                "quad_percentile_le":pct_le([x["lost_unique_quadruples"] for x in rr],rec["lost_unique_quadruples"]),
            }
        event_results.append(item)

    payload={
        "experiment":"function_loss_slot_baseline_v0",
        "evaluation_rounds":len(df)-MODEL_WINDOW,
        "hypothesis":"Do event tickets remain unusual in leave-one-out 3/4 coverage loss after subtracting source/slot or role machine signatures?",
        "metrics":[
            "lost_unique_triples",
            "lost_unique_quadruples",
        ],
        "event_results":event_results,
        "boundary":[
            "No winning numbers are read or used in this script.",
            "The nine event ticket identities were frozen from the prior outcome-overlay observer before this baseline run.",
            "Percentiles compare each frozen event ticket to all 596 pre-draw realizations of the same source/slot.",
            "Role percentiles are descriptive secondary baselines for Hybrid tickets.",
            "No Preserve/Exchange/Unknown decision and no composite score is produced.",
            "This remains explanatory because the event identities were discovered from historical outcomes.",
        ],
    }
    out=Path("results/function_loss_slot_baseline_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("=== FUNCTION LOSS SLOT BASELINE v0 ===")
    for x in event_results:
        b=x["same_slot_baseline"]; o=x["observed"]
        print(x["round"],x["event"],x["source"],x["index0"],o["role"],
              "tri",o["lost_unique_triples"],round(b["triple_percentile_le"],3),
              "quad",o["lost_unique_quadruples"],round(b["quad_percentile_le"],3))
    print("saved ->",out)


if __name__=="__main__":
    main()
