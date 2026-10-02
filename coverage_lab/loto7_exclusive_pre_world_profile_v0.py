#!/usr/bin/env python3
from __future__ import annotations
import json, math, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lotocore
from coverage_lab.loto7_portfolio20_v0 import MODEL_WINDOW, POOL_K, A_TICKETS, main_numbers, unique_structured_bundle, portfolio_bundle, bundle_metrics

FEATURES=("core18_score_mean","core18_score_std","core18_score_range","top7_vs_rest_gap")

def pre_world(history):
    snap=lotocore.score_snapshot(history)
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    ranks={int(k):int(v) for k,v in snap["ranks"].items()}
    ranked=sorted(range(1,38),key=lambda n:(ranks[n],n))
    core=ranked[:POOL_K]
    vals=[scores[n] for n in core]
    mean=sum(vals)/len(vals)
    var=sum((x-mean)**2 for x in vals)/len(vals)
    top7=[scores[n] for n in ranked[:7]]
    rest=[scores[n] for n in ranked[7:18]]
    return core,scores,{
      "core18_score_mean":mean,
      "core18_score_std":math.sqrt(var),
      "core18_score_range":max(vals)-min(vals),
      "top7_vs_rest_gap":sum(top7)/7-sum(rest)/11,
    }

def main():
    df=pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    rows=[]
    for idx in range(MODEL_WINDOW,len(df)):
        history=df.iloc[idx-MODEL_WINDOW:idx]
        target=df.iloc[idx]; rnd=int(target["round"]); actual=main_numbers(target)
        pool,scores,feat=pre_world(history)
        seed=rnd*100_003+20261002
        a,_=unique_structured_bundle(pool,A_TICKETS,seed+1)
        b,_=portfolio_bundle(pool,scores,seed+2)
        ma=bundle_metrics(a,actual); mb=bundle_metrics(b,actual)
        a5=ma["tickets_ge5"]>0; b5=mb["tickets_ge5"]>0
        if a5==b5: continue
        rows.append({"round":rnd,"winner":"structured20" if a5 else "portfolio20",**feat})

    summary={}
    for f in FEATURES:
        s=[r[f] for r in rows if r["winner"]=="structured20"]
        p=[r[f] for r in rows if r["winner"]=="portfolio20"]
        summary[f]={
          "structured_only_mean":sum(s)/len(s),
          "portfolio_only_mean":sum(p)/len(p),
          "delta_S_minus_P":sum(s)/len(s)-sum(p)/len(p),
          "structured_only_values":s,
          "portfolio_only_values":p,
        }

    payload={
      "experiment":"loto7_exclusive_pre_world_profile_v0",
      "exclusive_round_count":len(rows),
      "features":list(FEATURES),
      "summary":summary,
      "rounds":rows,
      "boundary":[
        "All features are computed from pre-draw CORE18 score state only.",
        "No realized winning numbers enter the world features.",
        "No classifier, threshold search, feature selection, or post-result tuning is used.",
        "This probe only asks whether the two exclusive-success sets occupy visibly different pre-draw score landscapes."
      ]
    }
    out=Path("results/loto7_exclusive_pre_world_profile_v0.json"); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("=== EXCLUSIVE PRE-WORLD PROFILE v0 ===")
    print("COUNT",len(rows))
    print("SUMMARY",json.dumps(summary,ensure_ascii=False,sort_keys=True))
    print("ROUNDS",json.dumps(rows,ensure_ascii=False,sort_keys=True))
    print("BOUNDARY"," | ".join(payload["boundary"]))
    print("saved ->",out)
if __name__=="__main__": main()
