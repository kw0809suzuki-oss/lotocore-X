#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PRESET={"lotocore":"top7_mass","x":"field_center_abs"}
KS=(10,12)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("results/loto7_candidate_compression_state_observables.csv"))
    p.add_argument("--window",type=int,default=60)
    p.add_argument("--step",type=int,default=20)
    a=p.parse_args()

    df=pd.read_csv(a.input).dropna(subset=["state_bin"])
    print("=== COMPRESSION GATE FRESHNESS SCAN ===")
    print(f"window={a.window} step={a.step}")
    for model,metric in PRESET.items():
        m=df[(df.model==model)&(df.metric==metric)].copy()
        rounds=sorted(m["round"].unique())
        print(f"MODEL {model} METRIC {metric}")
        rows=[]
        for start in range(0,max(1,len(rounds)-a.window+1),a.step):
            rr=rounds[start:start+a.window]
            if len(rr)<a.window: continue
            z=m[m["round"].isin(rr)]
            diffs=[]
            for k in KS:
                hi=z[(z.state_bin=="high")&(z.k==k)].hit_lift.mean()
                lo=z[(z.state_bin=="low")&(z.k==k)].hit_lift.mean()
                diffs.append(hi-lo)
            mean_diff=sum(diffs)/len(diffs)
            sign="positive" if mean_diff>0 else "nonpositive"
            rows.append((rr[0],rr[-1],diffs[0],diffs[1],mean_diff,sign))
            print(f" {rr[0]}..{rr[-1]} K10={diffs[0]:+.3f} K12={diffs[1]:+.3f} mean={mean_diff:+.3f} {sign}")
        # Find the latest contiguous positive/nonpositive streak.
        if rows:
            latest_sign=rows[-1][-1]
            streak=0
            for row in reversed(rows):
                if row[-1]==latest_sign: streak+=1
                else: break
            print(f" latest_sign={latest_sign} latest_streak_windows={streak}")
    print("BOUNDARY: descriptive freshness scan over the same historical series. Window/step are fixed for this run, not optimized.")

if __name__=="__main__":
    main()
