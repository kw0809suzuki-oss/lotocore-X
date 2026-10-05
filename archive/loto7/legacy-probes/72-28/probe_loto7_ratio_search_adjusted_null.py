from __future__ import annotations
from collections import Counter
from pathlib import Path
import random, pandas as pd, numpy as np
DATA=Path("data/loto7.csv"); OUT=Path("results/loto7_ratio_search_adjusted_null.csv")
NUMBERS=list(range(1,38)); WINDOW=100; WORLDS=1000; DRAWS=695; SEED=20260926
ALPHAS=[i/100 for i in range(0,101)]
def eval_alpha(draws,a):
 hits=[]
 for i in range(WINDOW,len(draws)):
  h=draws[i-WINDOW:i]; lo=h[-100:]; re=h[-20:]
  fl=Counter(n for d in lo for n in d); fr=Counter(n for d in re for n in d)
  s={n:a*fl[n]/100+(1-a)*fr[n]/20 for n in NUMBERS}
  t=set(sorted(NUMBERS,key=lambda n:(-s[n],n))[:7]); hits.append(len(t & set(draws[i])))
 chunks=[hits[:149],hits[149:298],hits[298:446],hits[446:]]
 q=np.array([np.mean(x) for x in chunks]); return float(np.mean(hits)),float(q.max()-q.min()),q
def search(draws):
 vals=[]
 for a in ALPHAS:
  mean,rng,q=eval_alpha(draws,a)
  # predeclared objective: maximize mean first, then minimize temporal range, then lower alpha
  vals.append((mean,-rng,-a,a,rng,q))
 return max(vals,key=lambda x:x[:3])
def main():
 df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
 real=[[int(r[f"n{i}"]) for i in range(1,8)] for _,r in df.iterrows()]
 rm,rneg,aneg,ra,rr,rq=search(real)
 print("REAL search alpha",ra,"mean",rm,"range",rr,"quarters",rq.tolist())
 rng=random.Random(SEED); rows=[]
 for w in range(WORLDS):
  draws=[rng.sample(NUMBERS,7) for _ in range(DRAWS)]
  m,_,_,a,r,q=search(draws)
  rows.append({"world":w,"best_alpha":a,"best_mean":m,"range_at_best":r,
               "q1":q[0],"q2":q[1],"q3":q[2],"q4":q[3]})
 n=pd.DataFrame(rows); OUT.parent.mkdir(exist_ok=True); n.to_csv(OUT,index=False)
 print("NULL worlds",WORLDS,"alphas",len(ALPHAS))
 print("P(null searched best mean >= real searched best mean)",float((n.best_mean>=rm).mean()))
 print("P(null searched best mean>=real AND range<=real)",float(((n.best_mean>=rm)&(n.range_at_best<=rr)).mean()))
 print("best_mean quantiles",n.best_mean.quantile([.5,.9,.95,.975,.99,.995]).to_dict())
 print("range_at_best quantiles",n.range_at_best.quantile([.01,.05,.1,.25,.5]).to_dict())
 print("alpha median",float(n.best_alpha.median()))
if __name__=="__main__": main()
