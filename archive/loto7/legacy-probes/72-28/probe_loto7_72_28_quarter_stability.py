from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd

DATA=Path("data/loto7.csv"); OUT=Path("results/loto7_72_28_quarter_stability.csv")
NUMBERS=range(1,38); WINDOW=100

def ticket(h,lw,rw):
    lo=h[-100:]; re=h[-20:]
    fl=Counter(n for d in lo for n in d); fr=Counter(n for d in re for n in d)
    s={n:lw*fl[n]/len(lo)+rw*fr[n]/len(re) for n in NUMBERS}
    return set(sorted(NUMBERS,key=lambda n:(-s[n],n))[:7])

def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=[[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]
    rows=[]
    for i in range(WINDOW,len(draws)):
        a=set(draws[i]); h=draws[i-WINDOW:i]; rnd=int(df.iloc[i]["round"])
        for m,lw,rw in (("frozen_72_28",.72,.28),("long_only",1,0),("recent_only",0,1)):
            x=len(ticket(h,lw,rw)&a)
            rows.append({"round":rnd,"model":m,"hits":x,"hit3":int(x>=3)})
    d=pd.DataFrame(rows)
    rounds=sorted(d["round"].unique())
    # Fixed chronological quarters: 149,149,148,149 evaluable rounds.
    chunks=[rounds[:149],rounds[149:298],rounds[298:446],rounds[446:]]
    qmap={r:f"Q{k+1}" for k,c in enumerate(chunks) for r in c}
    d["quarter"]=d["round"].map(qmap)
    s=d.groupby(["quarter","model"]).agg(n=("hits","size"),mean_hit=("hits","mean"),p3=("hit3","mean")).reset_index()
    out=[]
    for q,g in s.groupby("quarter"):
        z=g.set_index("model"); sub=d[d.quarter==q]
        fm=z.loc["frozen_72_28","mean_hit"]; fp=z.loc["frozen_72_28","p3"]
        dm=fm-max(z.loc["long_only","mean_hit"],z.loc["recent_only","mean_hit"])
        dp=fp-max(z.loc["long_only","p3"],z.loc["recent_only","p3"])
        out.append({"quarter":q,"round_min":int(sub["round"].min()),"round_max":int(sub["round"].max()),
                    "frozen_mean":fm,"frozen_p3":fp,"mountain_delta_mean":dm,"mountain_delta_p3":dp})
    o=pd.DataFrame(out); OUT.parent.mkdir(exist_ok=True); o.to_csv(OUT,index=False)
    print("FROZEN 72:28; chronological quarters; no retuning")
    print(s.to_string(index=False)); print("=== QUARTER STABILITY ==="); print(o.to_string(index=False))
    print("frozen mean range",o.frozen_mean.max()-o.frozen_mean.min())
    print("frozen mean std",o.frozen_mean.std(ddof=0))
    print("all quarters mountain mean positive",bool((o.mountain_delta_mean>0).all()))
    print("all quarters mountain p3 positive",bool((o.mountain_delta_p3>0).all()))
if __name__=="__main__": main()
