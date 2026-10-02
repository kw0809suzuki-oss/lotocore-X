#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K, A_TICKETS,
    unique_structured_bundle, portfolio_bundle,
)
from coverage_lab.loto7_hybrid_ratio_sweep_v0 import RATIOS, select, merge_unique

TARGET_ROUND=697

def digest_tickets(tickets):
    payload=";".join("-".join(f"{n:02d}" for n in t) for t in tickets)
    return hashlib.sha256(payload.encode()).hexdigest()

def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    if int(df.iloc[-1]["round"]) != 696:
        raise RuntimeError("freeze requires fixture ending exactly at round 696")

    history=df.iloc[-MODEL_WINDOW:]
    snap=lotocore.score_snapshot(history)
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    pool=sorted(range(1,38),key=lambda n:(ranks[n],n))[:POOL_K]

    seed=TARGET_ROUND*100_003+20261002
    structured20,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
    portfolio20,_=portfolio_bundle(pool,scores,seed+2)

    hybrids={}
    for a_n,b_n in RATIOS:
        key=f"{a_n}+{b_n}"
        h20,rep=merge_unique(select(structured20,a_n),select(portfolio20,b_n),portfolio20)
        hybrids[key]={
            "structured_n":a_n,
            "portfolio_n":b_n,
            "collision_repairs":rep,
            "tickets":[list(t) for t in h20],
            "sha256":digest_tickets(h20),
        }

    payload={
        "experiment":"loto7_hybrid_reproduction_oos_freeze_v0",
        "target_round":TARGET_ROUND,
        "history_last_round":696,
        "history_window":MODEL_WINDOW,
        "candidate_pool":pool,
        "seed":seed,
        "structured20":{"tickets":[list(t) for t in structured20],"sha256":digest_tickets(structured20)},
        "portfolio20":{"tickets":[list(t) for t in portfolio20],"sha256":digest_tickets(portfolio20)},
        "hybrids":hybrids,
        "evaluation_status":"FROZEN_BEFORE_RESULT",
        "boundary":[
            "Round 697 result is not used or stored by this freeze.",
            "All tickets are generated only from rounds through 696.",
            "Generators, ratio set, spread-index selection and collision repair are unchanged from the discovery probe.",
            "After the draw, evaluation must only compare the frozen tickets with the published round 697 main numbers.",
            "No regeneration after result publication is permitted for this frozen observation."
        ]
    }

    out=Path("results/loto7_hybrid_reproduction_round697_freeze_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== ROUND 697 OOS FREEZE v0 ===")
    print("TARGET",TARGET_ROUND)
    print("HISTORY_LAST",696)
    print("CORE18","-".join(f"{n:02d}" for n in pool))
    print("STRUCTURED_SHA",payload["structured20"]["sha256"])
    print("PORTFOLIO_SHA",payload["portfolio20"]["sha256"])
    print("HYBRID_SHA",json.dumps({k:v["sha256"] for k,v in hybrids.items()},sort_keys=True))
    print("STATUS","FROZEN_BEFORE_RESULT")
    print("saved ->",out)

if __name__=="__main__":
    main()
