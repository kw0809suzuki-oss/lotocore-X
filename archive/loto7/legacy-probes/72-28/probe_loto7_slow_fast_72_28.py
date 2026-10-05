from __future__ import annotations
import random
from collections import Counter
from pathlib import Path
import pandas as pd
import lotocore

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_slow_fast_72_28_probe.csv")
SUMMARY=Path("results/loto7_slow_fast_72_28_summary.csv")
NUMBERS=range(1,38)
WINDOW=100
SYNTH_WORLDS=500
SYNTH_DRAWS=600
SEED=20260926

def draws_from_df(df):
    return [sorted(int(r[f"n{i}"]) for i in range(1,8)) for _,r in df.iterrows()]

def score_ticket(history,long_w,recent_w):
    long=history[-100:]; recent=history[-20:]
    fl=Counter(n for d in long for n in d); fr=Counter(n for d in recent for n in d)
    scores={n:long_w*fl[n]/len(long)+recent_w*fr[n]/len(recent) for n in NUMBERS}
    return tuple(sorted(sorted(NUMBERS,key=lambda n:(-scores[n],n))[:7]))

def core_ticket(history):
    # exact current Core formula, observation copy; production code remains untouched
    long=history[-100:]; recent=history[-20:]
    fl=Counter(n for d in long for n in d); fr=Counter(n for d in recent for n in d)
    gap={n:len(history) for n in NUMBERS}
    for g,d in enumerate(reversed(history)):
        for n in d:
            if gap[n]==len(history): gap[n]=g
    scores={n:.60*fl[n]/len(long)+.25*fr[n]/len(recent)+.15*min(gap[n],12)/12 for n in NUMBERS}
    return tuple(sorted(sorted(NUMBERS,key=lambda n:(-scores[n],n))[:7]))

def hit(t,a): return len(set(t)&set(a))

def eval_world(draws,label):
    rows=[]
    for i in range(WINDOW,len(draws)):
        h=draws[i-WINDOW:i]; a=draws[i]
        tickets={
          "frozen_72_28":score_ticket(h,.72,.28),
          "long_only":score_ticket(h,1,0),
          "recent_only":score_ticket(h,0,1),
          "current_core":core_ticket(h),
        }
        for m,t in tickets.items(): rows.append({"world":label,"step":i,"model":m,"hits":hit(t,a)})
    return rows

def summarize(rows):
    d=pd.DataFrame(rows)
    g=d.groupby(["world","model"]).hits.agg(["count","mean"]).reset_index()
    p=d.assign(hit3=d.hits>=3).groupby(["world","model"]).hit3.mean().reset_index(name="p3")
    return g.merge(p,on=["world","model"])

def main():
    real=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    real_draws=draws_from_df(real)
    all_rows=eval_world(real_draws,"real_fresh")
    rng=random.Random(SEED)
    synth_summaries=[]
    wins_mean=wins_p3=0
    for w in range(SYNTH_WORLDS):
        draws=[sorted(rng.sample(list(NUMBERS),7)) for _ in range(SYNTH_DRAWS)]
        rows=eval_world(draws,f"synth_{w:04d}")
        s=summarize(rows)
        vals=s.set_index("model")
        fm=vals.loc["frozen_72_28","mean"]; fp=vals.loc["frozen_72_28","p3"]
        wins_mean += fm>max(vals.loc["long_only","mean"],vals.loc["recent_only","mean"])
        wins_p3 += fp>max(vals.loc["long_only","p3"],vals.loc["recent_only","p3"])
        synth_summaries.append(s)
    real_s=summarize(all_rows)
    synth=pd.concat(synth_summaries,ignore_index=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT,index=False)
    combined=pd.concat([real_s,synth],ignore_index=True)
    combined.to_csv(SUMMARY,index=False)
    print("=== REAL FRESH ===")
    print(real_s.to_string(index=False))
    print("=== SYNTHETIC NULL ===")
    print(f"worlds={SYNTH_WORLDS} draws_per_world={SYNTH_DRAWS} seed={SEED}")
    print(f"72:28 beats BOTH endpoints by mean hits: {wins_mean}/{SYNTH_WORLDS} = {wins_mean/SYNTH_WORLDS:.4f}")
    print(f"72:28 beats BOTH endpoints by 3+ rate: {wins_p3}/{SYNTH_WORLDS} = {wins_p3/SYNTH_WORLDS:.4f}")
    print("FROZEN: long:recent=72:28, gap off. No tuning after results.")
    print(f"saved {OUT} and {SUMMARY}")

if __name__=="__main__": main()
