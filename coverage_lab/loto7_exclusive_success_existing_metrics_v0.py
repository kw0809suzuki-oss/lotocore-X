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
    main_numbers, portfolio_bundle, unique_structured_bundle, bundle_metrics,
)

METRICS=(
    "winning_pair_coverage",
    "winning_triple_coverage",
    "mean_pairwise_overlap",
    "mean_structure_distance",
    "number_usage_std",
)

def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
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

        ma=bundle_metrics(a,actual)
        mb=bundle_metrics(b,actual)

        a5=ma["tickets_ge5"]>0
        b5=mb["tickets_ge5"]>0
        if a5==b5:
            continue

        winner="structured20" if a5 else "portfolio20"
        loser="portfolio20" if a5 else "structured20"
        wm=ma if a5 else mb
        lm=mb if a5 else ma

        rec={
            "round":rnd,
            "winner":winner,
            "loser":loser,
            "winner_max_hits":wm["max_hits"],
            "loser_max_hits":lm["max_hits"],
        }
        for m in METRICS:
            rec[f"winner_{m}"]=wm[m]
            rec[f"loser_{m}"]=lm[m]
            rec[f"delta_winner_minus_loser_{m}"]=wm[m]-lm[m]
        rounds.append(rec)

    directional={}
    for m in METRICS:
        vals=[r[f"delta_winner_minus_loser_{m}"] for r in rounds]
        directional[m]={
            "winner_gt_loser":sum(v>0 for v in vals),
            "equal":sum(v==0 for v in vals),
            "winner_lt_loser":sum(v<0 for v in vals),
            "mean_delta_winner_minus_loser":sum(vals)/len(vals),
            "structured_only_deltas":[r[f"delta_winner_minus_loser_{m}"] for r in rounds if r["winner"]=="structured20"],
            "portfolio_only_deltas":[r[f"delta_winner_minus_loser_{m}"] for r in rounds if r["winner"]=="portfolio20"],
        }

    payload={
        "experiment":"loto7_exclusive_success_existing_metrics_v0",
        "exclusive_round_count":len(rounds),
        "metrics":list(METRICS),
        "directional":directional,
        "rounds":rounds,
        "boundary":[
            "Uses only metrics already defined in Portfolio20 v0.",
            "Uses the same frozen unique Structured20 and Portfolio20 generation used by the exclusive-round probes.",
            "No new feature, classifier, threshold, or hybrid allocation is introduced.",
            "This asks only whether existing metric differences point consistently toward the strategy that realized 5+ in each exclusive round.",
        ],
    }

    out=Path("results/loto7_exclusive_success_existing_metrics_v0.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("=== EXCLUSIVE SUCCESS EXISTING METRICS v0 ===")
    print("COUNT",len(rounds))
    print("DIRECTIONAL",json.dumps(directional,ensure_ascii=False,sort_keys=True))
    print("ROUNDS",json.dumps(rounds,ensure_ascii=False,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print(f"saved -> {out}")

if __name__=="__main__":
    main()
