from __future__ import annotations
from collections import Counter
from pathlib import Path
import random, pandas as pd, numpy as np
DATA=Path("data/loto7.csv"); OUT=Path("results/loto7_72_28_temporal_stability_null.csv")
NUMBERS=list(range(1,38)); WINDOW=100; WORLDS=2000; DRAWS=695; SEED=20260926
def ticket(h):
 lo=h[-100:]; re=h[-20:]; fl=Counter(n for d in lo for n in d); fr=Counter(n for d in re for n in d)
 s={n:.72*fl[n]/100+.28*fr[n]/20 for n in NUMBERS}
 return set(sorted(NUMBERS,key=lambda n:(-s[n],n))[:7])
def qmeans(draws):
 hits=[]
 for i in range(WINDOW,len(draws)): hits.append(len(ticket(draws[i-WINDOW:i]) & set(draws[i])))
 chunks=[hits[:149],hits[149:298],hits[298:446],hits[446:]]
 return np.array([np.mean(x) for x in chunks])
def stab(q):
 # Smaller means more stable across chronological quarters.
 return {"range":float(q.max()-q.min()),"std":float(q.std()),"mad":float(np.mean(np.abs(q-q.mean()))),"mean":float(q.mean())}
def main():
 df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
 real_draws=[[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]
 rq=qmeans(real_draws); rs=stab(rq)
 print("FROZEN 72:28 temporal stability; no retuning")
 print("REAL quarter means",rq.tolist()); print("REAL",rs)
 rng=random.Random(SEED); rows=[]
 for w in range(WORLDS):
  draws=[rng.sample(NUMBERS,7) for _ in range(DRAWS)]
  q=qmeans(draws); z=stab(q); rows.append({"world":w,**z})
 n=pd.DataFrame(rows); OUT.parent.mkdir(exist_ok=True); n.to_csv(OUT,index=False)
 print("NULL worlds",WORLDS)
 for metric in ["range","std","mad"]:
  real=rs[metric]
  print(metric,"P(null <= real stability metric)",float((n[metric]<=real).mean()),
        "null_quantiles",n[metric].quantile([.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_dict())
 # Joint: at least as stable AND at least as high in mean.
 print("joint P(null stability range<=real AND mean>=real)",float(((n["range"]<=rs["range"])&(n["mean"]>=rs["mean"])).mean()))
 print("P(null mean>=real)",float((n["mean"]>=rs["mean"]).mean()))
if __name__=="__main__": main()
