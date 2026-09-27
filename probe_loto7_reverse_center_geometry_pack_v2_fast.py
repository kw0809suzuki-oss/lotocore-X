from __future__ import annotations

from pathlib import Path
from itertools import combinations
import statistics
import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
WINDOW = 100
NUMBERS = list(range(1, 38))
KS=(3,6,10)
SWAPS=(1,2,None)
MAGS=(0.5,1.0,1.5,2.0)

def truth(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1,8)))
def center(xs): return sum(xs)/7.0
def span(xs): return max(xs)-min(xs)
def sgn(x,eps=1e-12): return 1 if x>eps else (-1 if x<-eps else 0)
def hits(a,p): return len(set(a)&set(p))
def dist(a,p): return sum(min(abs(x-y) for y in p) for x in a)/7.0

def snap(history):
    core=tuple(sorted(lotocore.predict(history).numbers))
    ss=lotocore.score_snapshot(history)
    scores={int(k):float(v) for k,v in ss["scores"].items()}
    ranks={int(k):int(v) for k,v in ss["ranks"].items()}
    ranked=sorted(NUMBERS,key=lambda n:(ranks[n],n))
    noncore=tuple(n for n in ranked if n not in set(core))
    return core,scores,noncore,center(core),span(core)

def arm_name(k,sw,m):
    return f"k{k}_sw{'all' if sw is None else sw}_m{str(m).replace('.','p')}"

def precompute(core,noncore,scores,k):
    support=tuple(sorted(set(core)|set(noncore[:k])))
    core_set=set(core)
    out=[]
    for comb in combinations(support,7):
        out.append((comb,7-len(core_set&set(comb)),center(comb),sum(scores[n] for n in comb)))
    return out

def pick(cands, target, scale, sw):
    best=None; keybest=None
    for comb,n_sw,c,mass in cands:
        if sw is not None and n_sw>sw: continue
        key=(abs(c-target)/scale,n_sw,-mass,comb)
        if keybest is None or key<keybest:
            keybest=key; best=comb
    return best

def summarize(rows,label):
    print(f"--- {label} n={len(rows)} ---")
    bh=statistics.mean(r["base_h"] for r in rows); bd=statistics.mean(r["base_d"] for r in rows)
    vals=[]
    for k in KS:
      for sw in SWAPS:
       for m in MAGS:
        n=arm_name(k,sw,m)
        mh=statistics.mean(r[n+"_h"] for r in rows); md=statistics.mean(r[n+"_d"] for r in rows)
        p3=statistics.mean(r[n+"_h"]>=3 for r in rows)
        vals.append((mh-bh,bd-md,p3,n,mh,md))
    print("TOP exact")
    for dh,g,p3,n,mh,md in sorted(vals,key=lambda z:z[0],reverse=True)[:8]:
        print(f"{n:18s} hits={mh:.6f} dh={dh:+.6f} dist={md:.6f} gain={g:+.6f} 3+={p3:.6f}")
    print("TOP distance")
    for dh,g,p3,n,mh,md in sorted(vals,key=lambda z:z[1],reverse=True)[:8]:
        print(f"{n:18s} hits={mh:.6f} dh={dh:+.6f} dist={md:.6f} gain={g:+.6f} 3+={p3:.6f}")
    print()

def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    boxes={}
    for i in range(WINDOW,len(df)):
        boxes[int(df.iloc[i]["round"])]=snap(df.iloc[i-WINDOW:i])

    rows=[]
    for i in range(WINDOW+1,len(df)):
        r=int(df.iloc[i]["round"]); pr=int(df.iloc[i-1]["round"])
        core,scores,noncore,c,sp=boxes[r]
        pcore,_,_,pc,_=boxes[pr]
        cdir=sgn(c-pc); act=truth(df.iloc[i])
        rec={"round":r,"base_h":hits(act,core),"base_d":dist(act,core)}
        caches={k:precompute(core,noncore,scores,k) for k in KS}
        scale=max(sp,1.0)
        for k in KS:
          for sw in SWAPS:
           for m in MAGS:
            n=arm_name(k,sw,m)
            p=pick(caches[k],c-cdir*m,scale,sw)
            rec[n+"_h"]=hits(act,p); rec[n+"_d"]=dist(act,p)
        rows.append(rec)

    print("=== LOTO7 REVERSE CENTER GEOMETRY PACK v2 FAST ===")
    print(f"targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    summarize(rows,"ALL")
    summarize(rows[-200:],"RECENT200")
    summarize(rows[-100:],"RECENT100")

if __name__=="__main__":
    main()
