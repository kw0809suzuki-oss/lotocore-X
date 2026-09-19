#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

RANDOM_FINAL7_MEAN = 49/37

def classify(ratio: float) -> str:
    if ratio < 0.98:
        return "weak"
    if ratio > 1.02:
        return "strong"
    return "neutral"

def run(path: Path, lookback: int=50):
    df=pd.read_csv(path).sort_values(["model","round","k"]).reset_index(drop=True)
    rows=[]
    for model,m in df.groupby("model"):
        per_round=m.drop_duplicates("round")[["round","final7_hits"]].sort_values("round").copy()
        per_round["past_final7_mean"]=per_round["final7_hits"].shift(1).rolling(lookback,min_periods=lookback).mean()
        per_round["strength_ratio"]=per_round["past_final7_mean"]/RANDOM_FINAL7_MEAN
        per_round["strength_bin"]=per_round["strength_ratio"].map(lambda x: classify(x) if pd.notna(x) else None)
        mm=m.merge(per_round[["round","past_final7_mean","strength_ratio","strength_bin"]],on="round",how="left")
        rows.append(mm)
    return pd.concat(rows,ignore_index=True)

def summarize(df: pd.DataFrame):
    use=df.dropna(subset=["strength_bin"])
    print("=== HISTORICAL STRENGTH x CANDIDATE COMPRESSION ===")
    print("strength_t uses only previous 50 rounds final7 performance.")
    for model in sorted(use.model.unique()):
        print(f"MODEL {model}")
        m=use[use.model==model]
        for b in ("weak","neutral","strong"):
            g=m[m.strength_bin==b]
            rounds=g["round"].nunique()
            if rounds==0:
                continue
            print(f" BIN {b} rounds={rounds} strength_ratio_mean={g.strength_ratio.mean():.4f}")
            for k in sorted(g.k.unique()):
                x=g[g.k==k]
                print(
                    f"  K={int(k):2d} mean_hits={x.topk_hits.mean():.4f} "
                    f"mean_hit_lift={x.hit_lift.mean():.4f} "
                    f"P5+={(x.topk_hits>=5).mean():.4f}"
                )
    print("BOUNDARY: strength bins are pre-target, but thresholds 0.98/1.02 and 50-round lookback are exploratory fixed choices, not optimized.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("results/loto7_candidate_compression_fixed_k.csv"))
    p.add_argument("--out",type=Path,default=Path("results/loto7_candidate_compression_strength_bins.csv"))
    p.add_argument("--lookback",type=int,default=50)
    a=p.parse_args()
    df=run(a.input,a.lookback)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.out,index=False)
    summarize(df)
    print(f"saved -> {a.out}")

if __name__=="__main__":
    main()
