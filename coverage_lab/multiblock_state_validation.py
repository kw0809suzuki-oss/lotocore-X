#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PRESET={"lotocore":"top7_mass","x":"field_center_abs"}
KS=(10,12)

def block_summary(df, blocks:int):
    rounds=sorted(df["round"].unique())
    n=len(rounds)
    cuts=[]
    for b in range(blocks):
        lo=round(b*n/blocks)
        hi=round((b+1)*n/blocks)
        cuts.append((b+1,rounds[lo],rounds[hi-1],set(rounds[lo:hi])))
    return cuts

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("results/loto7_candidate_compression_state_observables.csv"))
    p.add_argument("--blocks",type=int,default=4)
    a=p.parse_args()
    df=pd.read_csv(a.input).dropna(subset=["state_bin"])
    print("=== PRE-REGISTERED MULTI-BLOCK STATE VALIDATION ===")
    print(f"blocks={a.blocks} preset={PRESET}")
    for model,metric in PRESET.items():
        m=df[(df.model==model)&(df.metric==metric)]
        print(f"MODEL {model} METRIC {metric}")
        positives=0
        usable=0
        for bid,r0,r1,rset in block_summary(m,a.blocks):
            z=m[m["round"].isin(rset)]
            parts=[]
            block_pos=True
            for k in KS:
                hi=z[(z.state_bin=="high")&(z.k==k)].hit_lift.mean()
                lo=z[(z.state_bin=="low")&(z.k==k)].hit_lift.mean()
                diff=hi-lo
                if pd.notna(diff):
                    usable+=1
                    if diff>0: positives+=1
                block_pos=block_pos and (pd.notna(diff) and diff>0)
                parts.append(f"K{k}:high={hi:.3f} low={lo:.3f} diff={diff:+.3f}")
            print(f" BLOCK {bid} rounds={r0}..{r1} " + " | ".join(parts) + f" both_positive={block_pos}")
        print(f" direction_positive_cells={positives}/{usable}")
    print("BOUNDARY: metrics were fixed before this run. No block-specific metric or threshold tuning.")

if __name__=="__main__":
    main()
