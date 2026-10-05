from __future__ import annotations
import random
from collections import Counter
from pathlib import Path
import pandas as pd

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_72_28_mountain_height.csv")
NUMBERS=range(1,38); WINDOW=100; WORLDS=2000; DRAWS=600; SEED=20260926

def score_ticket(h,lw,rw):
    lo=h[-100:]; re=h[-20:]
    fl=Counter(n for d in lo for n in d); fr=Counter(n for d in re for n in d)
    s={n:lw*fl[n]/len(lo)+rw*fr[n]/len(re) for n in NUMBERS}
    return set(sorted(NUMBERS,key=lambda n:(-s[n],n))[:7])

def metrics(draws):
    hits={"frozen":[],"long":[],"recent":[]}
    for i in range(WINDOW,len(draws)):
        h=draws[i-WINDOW:i]; a=set(draws[i])
        for k,t in (("frozen",score_ticket(h,.72,.28)),("long",score_ticket(h,1,0)),("recent",score_ticket(h,0,1))):
            hits[k].append(len(t&a))
    mean={k:sum(v)/len(v) for k,v in hits.items()}
    p3={k:sum(x>=3 for x in v)/len(v) for k,v in hits.items()}
    return mean,p3

def main():
    df=pd.read_csv(DATA).sort_values("round")
    real=[[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]
    rm,rp=metrics(real)
    real_dm=rm["frozen"]-max(rm["long"],rm["recent"])
    real_dp=rp["frozen"]-max(rp["long"],rp["recent"])
    rng=random.Random(SEED); rows=[]
    for w in range(WORLDS):
        d=[rng.sample(list(NUMBERS),7) for _ in range(DRAWS)]
        m,p=metrics(d)
        rows.append({"world":w,"delta_mean":m["frozen"]-max(m["long"],m["recent"]),
                     "delta_p3":p["frozen"]-max(p["long"],p["recent"])})
    out=pd.DataFrame(rows); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT,index=False)
    ge_m=(out.delta_mean>=real_dm).mean(); ge_p=(out.delta_p3>=real_dp).mean()
    print("REAL frozen",rm["frozen"],rp["frozen"])
    print("REAL endpoints",rm["long"],rm["recent"],rp["long"],rp["recent"])
    print(f"REAL mountain delta_mean={real_dm:.6f} delta_p3={real_dp:.6f}")
    print(f"NULL worlds={WORLDS} P(null mountain >= real): mean={ge_m:.4f} p3={ge_p:.4f}")
    print("NULL delta_mean quantiles",out.delta_mean.quantile([.5,.75,.9,.95,.975,.99]).to_dict())
    print("NULL delta_p3 quantiles",out.delta_p3.quantile([.5,.75,.9,.95,.975,.99]).to_dict())
if __name__=="__main__": main()
