#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

KS=(10,12,15,18)

def summarize_metric(df, metric, label):
    out={}
    g=df[df.metric==metric]
    for b in ("low","mid","high"):
        z=g[g.state_bin==b]
        out[b]={int(k): float(z[z.k==k].hit_lift.mean()) for k in KS}
    print(f" {label} metric={metric}")
    for b in ("low","mid","high"):
        vals=" ".join(f"K{k}={out[b][k]:.3f}" for k in KS)
        print(f"  {b:4s} {vals}")
    return out

def score_discovery(g):
    # directional score: high-state lift minus low-state lift at the two most compressed spans.
    return (
        g[(g.state_bin=="high") & (g.k==10)].hit_lift.mean()
        - g[(g.state_bin=="low") & (g.k==10)].hit_lift.mean()
    ) + (
        g[(g.state_bin=="high") & (g.k==12)].hit_lift.mean()
        - g[(g.state_bin=="low") & (g.k==12)].hit_lift.mean()
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("results/loto7_candidate_compression_state_observables.csv"))
    a=p.parse_args()
    df=pd.read_csv(a.input).dropna(subset=["state_bin"])
    rounds=sorted(df["round"].unique())
    cut=rounds[len(rounds)//2]
    discovery=df[df["round"]<cut]
    validation=df[df["round"]>=cut]

    print("=== TEMPORAL DISCOVERY -> VALIDATION FOR STATE COMPRESSION ===")
    print(f"cut_round={cut} discovery_rounds={discovery['round'].nunique()} validation_rounds={validation['round'].nunique()}")

    for model in sorted(df.model.unique()):
        d=discovery[discovery.model==model]
        v=validation[validation.model==model]
        metrics=sorted(d.metric.unique())
        scores=[]
        for metric in metrics:
            g=d[d.metric==metric]
            s=score_discovery(g)
            if pd.notna(s):
                scores.append((s,metric))
        scores.sort(reverse=True)
        best_score,best_metric=scores[0]
        print(f"MODEL {model} selected_on_discovery={best_metric} discovery_direction_score={best_score:.4f}")
        summarize_metric(d,best_metric,"DISCOVERY")
        summarize_metric(v,best_metric,"VALIDATION")
    print("BOUNDARY: one metric per model selected only from the chronological discovery half, then reported unchanged on the later half. No final component promotion.")

if __name__=="__main__":
    main()
