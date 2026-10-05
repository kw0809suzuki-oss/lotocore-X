from __future__ import annotations
from collections import Counter
from pathlib import Path
import os, random, pandas as pd, numpy as np

DATA=Path("data/loto7.csv")
NUMBERS=list(range(1,38))
WINDOW=100
ALPHAS=[i/100 for i in range(101)]
SHARD_INDEX=int(os.environ.get("SHARD_INDEX","0"))
NULL_WORLDS=500
SEED=20260927 + SHARD_INDEX

def ticket(history,a):
    long=history[-100:]; recent=history[-20:]
    fl=Counter(n for d in long for n in d)
    fr=Counter(n for d in recent for n in d)
    score={n:a*fl[n]/100+(1-a)*fr[n]/20 for n in NUMBERS}
    return set(sorted(NUMBERS,key=lambda n:(-score[n],n))[:7])

def eval_alpha(draws,a):
    hits=[]; survivors=[]; reentry=[]; prev=None
    for i in range(WINDOW,len(draws)):
        t=ticket(draws[i-WINDOW:i],a); actual=set(draws[i])
        hits.append(len(t&actual))
        if prev is not None:
            s=t&prev
            survivors.append(len(s&actual))
            reentry.append(len((t-prev)&actual))
        prev=t
    q=np.array([np.mean(x) for x in np.array_split(hits,4)])
    return {"mean":float(np.mean(hits)),"p3":float(np.mean(np.array(hits)>=3)),
            "range":float(q.max()-q.min()),"quarters":q,
            "survivor_hit":float(np.mean(survivors)),
            "reentry_hit":float(np.mean(reentry))}

def search(draws):
    vals=[]
    for a in ALPHAS:
        m=eval_alpha(draws,a)
        vals.append((m["mean"],-m["range"],-a,a,m))
    return max(vals,key=lambda x:x[:3])

def load_real():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    return [[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]

def main():
    real=load_real()
    _,_,_,ra,rm=search(real)
    print("REAL best alpha",ra,"mean",rm["mean"],"p3",rm["p3"],"range",rm["range"])
    print("REAL quarters",rm["quarters"].tolist())
    print("REAL survivor-hit",rm["survivor_hit"],"reentry-hit",rm["reentry_hit"])

    rng=random.Random(SEED); rows=[]
    for w in range(NULL_WORLDS):
        draws=[rng.sample(NUMBERS,7) for _ in range(len(real))]
        _,_,_,a,m=search(draws)
        rows.append({"world":SHARD_INDEX*NULL_WORLDS+w,"shard":SHARD_INDEX,"alpha":a,
                     "mean":m["mean"],"p3":m["p3"],"range":m["range"],
                     "survivor_hit":m["survivor_hit"],"reentry_hit":m["reentry_hit"]})
    out=Path(f"results/loto7_survivor_reentry_shard_{SHARD_INDEX}.csv")
    out.parent.mkdir(exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print("SHARD",SHARD_INDEX,"NULL worlds",NULL_WORLDS,"seed",SEED)

if __name__=="__main__": main()
