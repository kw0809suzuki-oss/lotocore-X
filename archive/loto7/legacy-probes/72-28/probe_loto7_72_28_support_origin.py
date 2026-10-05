from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd
DATA=Path("data/loto7.csv"); OUT=Path("results/loto7_72_28_support_origin.csv"); NUMBERS=range(1,38); WINDOW=100
def scores(h,lw,rw):
 lo=h[-100:]; re=h[-20:]; fl=Counter(n for d in lo for n in d); fr=Counter(n for d in re for n in d)
 s={n:lw*fl[n]/100+rw*fr[n]/20 for n in NUMBERS}; return s
def top7(s): return set(sorted(NUMBERS,key=lambda n:(-s[n],n))[:7])
def main():
 df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
 draws=[[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]
 rows=[]
 for i in range(WINDOW,len(draws)):
  h=draws[i-WINDOW:i]; actual=set(draws[i]); rnd=int(df.iloc[i]["round"])
  L=top7(scores(h,1,0)); R=top7(scores(h,0,1)); M=top7(scores(h,.72,.28))
  for n in sorted(M):
   origin="both" if n in L and n in R else ("long_only" if n in L else ("recent_only" if n in R else "blend_only"))
   rows.append({"round":rnd,"number":n,"origin":origin,"hit":int(n in actual)})
 d=pd.DataFrame(rows)
 rounds=sorted(d["round"].unique()); chunks=[rounds[:149],rounds[149:298],rounds[298:446],rounds[446:]]
 qmap={r:f"Q{k+1}" for k,c in enumerate(chunks) for r in c}; d["quarter"]=d["round"].map(qmap)
 s=d.groupby(["quarter","origin"]).agg(selected=("hit","size"),hits=("hit","sum"),hit_rate=("hit","mean")).reset_index()
 all_s=d.groupby("origin").agg(selected=("hit","size"),hits=("hit","sum"),hit_rate=("hit","mean")).reset_index()
 OUT.parent.mkdir(exist_ok=True); s.to_csv(OUT,index=False)
 print("FROZEN 72:28 support-origin decomposition; no retuning")
 print("=== ALL ==="); print(all_s.to_string(index=False))
 print("=== QUARTERS ==="); print(s.to_string(index=False))
 for q in ["Q1","Q2","Q3","Q4"]:
  x=d[d.quarter==q]
  print(q,"hits",int(x.hit.sum()),"both_hit_share", (x[(x.origin=="both")].hit.sum()/x.hit.sum() if x.hit.sum() else 0),
        "both_selection_share",(x.origin=="both").mean())
if __name__=="__main__": main()
