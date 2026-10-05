from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_72_28_era_split.csv")
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
    # Predeclared chronological halves of the evaluable 595 rounds: first 297, last 298.
    cut=rounds[297]
    d["era"]=d["round"].apply(lambda r:"early" if r<cut else "late")
    s=d.groupby(["era","model"]).agg(n=("hits","size"),mean_hit=("hits","mean"),p3=("hit3","mean")).reset_index()
    out=[]
    for era,g in s.groupby("era"):
        z=g.set_index("model")
        dm=z.loc["frozen_72_28","mean_hit"]-max(z.loc["long_only","mean_hit"],z.loc["recent_only","mean_hit"])
        dp=z.loc["frozen_72_28","p3"]-max(z.loc["long_only","p3"],z.loc["recent_only","p3"])
        out.append({"era":era,"round_min":int(d[d.era==era]["round"].min()),"round_max":int(d[d.era==era]["round"].max()),
                    "mountain_delta_mean":dm,"mountain_delta_p3":dp})
    o=pd.DataFrame(out); OUT.parent.mkdir(exist_ok=True); o.to_csv(OUT,index=False)
    print("FROZEN 72:28; no retuning")
    print(s.to_string(index=False))
    print("=== MOUNTAIN BY ERA ===")
    print(o.to_string(index=False))
    print("same-sign mean mountain:",bool((o.mountain_delta_mean>0).all()))
    print("same-sign p3 mountain:",bool((o.mountain_delta_p3>0).all()))
if __name__=="__main__": main()
