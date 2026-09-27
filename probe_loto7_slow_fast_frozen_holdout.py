from __future__ import annotations
from collections import Counter
from pathlib import Path
import random
import numpy as np
import pandas as pd

DATA=Path("data/loto7.csv")
NUMBERS=list(range(1,38))
WINDOW=100
ALPHAS=[i/100 for i in range(101)]
NULL_WORLDS=500
SEED=20260927
TRAIN_FRACTION=0.5

def ticket(history,a):
    long=history[-100:]; recent=history[-20:]
    fl=Counter(n for d in long for n in d)
    fr=Counter(n for d in recent for n in d)
    score={n:a*fl[n]/100+(1-a)*fr[n]/20 for n in NUMBERS}
    return set(sorted(NUMBERS,key=lambda n:(-score[n],n))[:7])

def hits_series(draws,a,start,end):
    hits=[]
    for i in range(start,end):
        t=ticket(draws[i-WINDOW:i],a)
        hits.append(len(t & set(draws[i])))
    return np.array(hits,dtype=float)

def metrics(h):
    return {
        "mean":float(h.mean()),
        "p3":float((h>=3).mean()),
        "n":int(len(h)),
    }

def choose_alpha(draws,train_start,train_end):
    vals=[]
    for a in ALPHAS:
        h=hits_series(draws,a,train_start,train_end)
        m=metrics(h)
        vals.append((m["mean"],-a,a,m))
    return max(vals,key=lambda x:x[:2])

def load_real():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    return [[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]

def evaluate_world(draws):
    eval_n=len(draws)-WINDOW
    train_n=eval_n//2
    train_start=WINDOW
    train_end=WINDOW+train_n
    test_start=train_end
    test_end=len(draws)
    _,_,a,train_m=choose_alpha(draws,train_start,train_end)
    test_m=metrics(hits_series(draws,a,test_start,test_end))
    return a,train_m,test_m,train_start,train_end,test_start,test_end

def main():
    real=load_real()
    a,tr,te,ts,te_idx,hs,he=evaluate_world(real)
    print("FROZEN alpha",a)
    print("TRAIN",ts,te_idx,"n",tr["n"],"mean",tr["mean"],"p3",tr["p3"])
    print("HOLDOUT",hs,he,"n",te["n"],"mean",te["mean"],"p3",te["p3"])

    rng=random.Random(SEED)
    rows=[]
    for w in range(NULL_WORLDS):
        draws=[rng.sample(NUMBERS,7) for _ in range(len(real))]
        na,ntr,nte,*_=evaluate_world(draws)
        rows.append({"world":w,"alpha":na,
                     "train_mean":ntr["mean"],"train_p3":ntr["p3"],
                     "holdout_mean":nte["mean"],"holdout_p3":nte["p3"]})
    n=pd.DataFrame(rows)
    print("NULL worlds",len(n))
    print("P(null holdout mean >= real)",float((n["holdout_mean"]>=te["mean"]).mean()))
    print("P(null holdout p3 >= real)",float((n["holdout_p3"]>=te["p3"]).mean()))
    print("P(null holdout mean>=real AND p3>=real)",
          float(((n["holdout_mean"]>=te["mean"])&(n["holdout_p3"]>=te["p3"])).mean()))
    Path("results").mkdir(exist_ok=True)
    n.to_csv("results/loto7_slow_fast_frozen_holdout.csv",index=False)

if __name__=="__main__":
    main()
