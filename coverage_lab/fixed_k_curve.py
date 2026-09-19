#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import lotocore
import x_agent

KS=(10,12,15,18,22,26,30,37)
MODELS=("lotocore","x")

def actual(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}

def ranked_numbers(snapshot):
    return sorted(range(1,38), key=lambda n: snapshot["ranks"][str(n)])

def run(data_path: Path, window: int=100):
    df=pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    rows=[]
    for i in range(window,len(df)):
        hist=df.iloc[i-window:i]
        row=df.iloc[i]
        a=actual(row)
        snaps={
            "lotocore":lotocore.score_snapshot(hist),
            "x":x_agent.score_snapshot(hist,competition_gate=True),
        }
        for model,snap in snaps.items():
            rank=ranked_numbers(snap)
            final_pred=(
                lotocore.predict(hist).numbers if model=="lotocore"
                else x_agent.predict(hist,competition_gate=True).numbers
            )
            final_hits=len(set(final_pred)&a)
            for k in KS:
                pool=set(rank[:k])
                h=len(pool&a)
                random_expected=7*k/37
                rows.append({
                    "round":int(row["round"]),
                    "date":row.get("date",""),
                    "model":model,
                    "k":k,
                    "compression_ratio":k/37,
                    "topk_hits":h,
                    "recall":h/7,
                    "precision":h/k,
                    "random_expected_hits":random_expected,
                    "hit_lift":h/random_expected if random_expected else 0,
                    "precision_lift":(h/k)/(7/37) if k else 0,
                    "final7_hits":final_hits,
                })
    return pd.DataFrame(rows)

def summarize(res: pd.DataFrame):
    print("=== LOTO7 FIXED-K CANDIDATE COMPRESSION CURVE ===")
    print(f"rounds={res['round'].nunique()} models={sorted(res.model.unique())}")
    for model in MODELS:
        print(f"MODEL {model}")
        m=res[res.model==model]
        for k in KS:
            g=m[m.k==k]
            print(
                f" K={k:2d} ratio={k/37:.3f} "
                f"mean_hits={g.topk_hits.mean():.4f} "
                f"recall={g.recall.mean():.4f} "
                f"mean_hit_lift={g.hit_lift.mean():.4f} "
                f"P(4+)={(g.topk_hits>=4).mean():.4f} "
                f"P(5+)={(g.topk_hits>=5).mean():.4f} "
                f"P(6+)={(g.topk_hits>=6).mean():.4f} "
                f"P(7)={(g.topk_hits>=7).mean():.4f}"
            )
        print(
            f" final7 mean_hits={m.drop_duplicates('round').final7_hits.mean():.4f}"
        )
    print("BOUNDARY: candidate-layer compression only; no 10-ticket allocator is included.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/loto7.csv"))
    p.add_argument("--window",type=int,default=100)
    p.add_argument("--out",type=Path,default=Path("results/loto7_candidate_compression_fixed_k.csv"))
    a=p.parse_args()
    res=run(a.data,a.window)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(a.out,index=False)
    summarize(res)
    print(f"saved -> {a.out}")

if __name__=="__main__":
    main()
