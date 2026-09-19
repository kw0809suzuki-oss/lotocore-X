#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PRESET={"lotocore":"top7_mass","x":"field_center_abs"}
KS=(10,12)

def freshness_windows(m, window=60, step=20):
    rounds=sorted(m["round"].unique())
    out=[]
    for start in range(0,max(1,len(rounds)-window+1),step):
        rr=rounds[start:start+window]
        if len(rr)<window: continue
        z=m[m["round"].isin(rr)]
        diffs={}
        for k in KS:
            hi=z[(z.state_bin=="high")&(z.k==k)].hit_lift.mean()
            lo=z[(z.state_bin=="low")&(z.k==k)].hit_lift.mean()
            diffs[k]=hi-lo
        out.append({
            "start":rr[0],"end":rr[-1],
            "mean_diff":sum(diffs.values())/len(diffs),
            "k10_diff":diffs[10],"k12_diff":diffs[12]
        })
    return pd.DataFrame(out)

def evaluate_rules(w):
    # rule decisions are made at each window end using only windows up to that point
    rows=[]
    for i in range(len(w)):
        hist=w.iloc[:i+1]
        signs=(hist.mean_diff>0).astype(int).tolist()
        candidates={
            "last1": signs[-1]==1,
            "last2_both": len(signs)>=2 and sum(signs[-2:])==2,
            "last3_2of3": len(signs)>=3 and sum(signs[-3:])>=2,
            "last3_all": len(signs)>=3 and sum(signs[-3:])==3,
        }
        for rule,fresh in candidates.items():
            rows.append({"idx":i,"end":int(w.iloc[i].end),"rule":rule,"fresh":bool(fresh),"next_idx":i+1})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("results/loto7_candidate_compression_state_observables.csv"))
    a=p.parse_args()
    df=pd.read_csv(a.input).dropna(subset=["state_bin"])

    print("=== FRESHNESS RULE FORWARD TEST ===")
    print("Rules preregistered: last1, last2_both, last3_2of3, last3_all")
    for model,metric in PRESET.items():
        m=df[(df.model==model)&(df.metric==metric)]
        w=freshness_windows(m)
        r=evaluate_rules(w)
        print(f"MODEL {model} metric={metric} windows={len(w)}")
        for rule in ("last1","last2_both","last3_2of3","last3_all"):
            z=r[(r.rule==rule)&(r.next_idx<len(w))].copy()
            vals=[]
            for _,row in z.iterrows():
                nextw=w.iloc[int(row.next_idx)]
                vals.append((row.fresh,float(nextw.mean_diff)))
            fresh_vals=[v for f,v in vals if f]
            stale_vals=[v for f,v in vals if not f]
            fresh_mean=sum(fresh_vals)/len(fresh_vals) if fresh_vals else float("nan")
            stale_mean=sum(stale_vals)/len(stale_vals) if stale_vals else float("nan")
            fresh_pos=sum(v>0 for v in fresh_vals)
            stale_pos=sum(v>0 for v in stale_vals)
            print(
                f" RULE {rule} fresh_n={len(fresh_vals)} fresh_next_mean={fresh_mean:+.3f} fresh_pos={fresh_pos}/{len(fresh_vals)} "
                f"stale_n={len(stale_vals)} stale_next_mean={stale_mean:+.3f} stale_pos={stale_pos}/{len(stale_vals)}"
            )
    print("BOUNDARY: forward window-level test on same historical series; overlapping windows and small effective sample. No rule promotion from this run alone.")

if __name__=="__main__":
    main()
