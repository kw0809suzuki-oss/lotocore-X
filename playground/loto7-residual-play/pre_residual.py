"""Look backward from each strong residual. Observation only; no causal/prediction claim."""
import sys, math, statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fetch_loto7 import fetch
C_TARGETS=[158,159,160,229,407,426,472,504,521,537,556,669,677]
AX=('mu','var','range','maxgap')
def obs(nums):
 xs=sorted(nums); mu=sum(xs)/7; var=sum((x-mu)**2 for x in xs)/7; gaps=[b-a for a,b in zip(xs,xs[1:])]
 return (mu,var,xs[-1]-xs[0],max(gaps))
def med(x): return statistics.median(x)
def main():
 df=fetch(10000); st={}
 for _,r in df.iterrows():
  n=[int(r[f'n{i}']) for i in range(1,8)]; st[int(r['round'])]=obs(n)
 # For each residual target, inspect raw State deltas at lags -5..0.
 paths=[]
 for t in C_TARGETS:
  p=[]
  for lag in range(-5,1):
   end=t+lag
   if end-1 in st and end in st: p.append((lag,tuple(st[end][d]-st[end-1][d] for d in range(4))))
  paths.append((t,p))
 print('PRE-RESIDUAL WALK / raw State deltas; residual target at lag 0')
 print('axes=mu,var,range,maxgap')
 for t,p in paths:
  print('target',t,' '.join(f'{lag}:'+','.join(f'{v:+.2f}' for v in ds) for lag,ds in p))
 print('LAG AGGREGATES / median delta across residual episodes')
 for lag in range(-5,1):
  vals=[[ds[d] for _,p in paths for l,ds in p if l==lag] for d in range(4)]
  print(f'lag {lag}: '+','.join(f'{AX[d]}={med(vals[d]):+.2f}' for d in range(4)))
 print('DIRECTION CONSISTENCY / fraction sharing median sign')
 for lag in range(-5,1):
  parts=[]
  for d in range(4):
   vals=[ds[d] for _,p in paths for l,ds in p if l==lag]; m=med(vals); sign=1 if m>0 else (-1 if m<0 else 0)
   same=sum((v>0)-(v<0)==sign for v in vals)/len(vals) if sign else sum(v==0 for v in vals)/len(vals)
   parts.append(f'{AX[d]}={same:.2f}')
  print(f'lag {lag}: '+','.join(parts))
 # Compare absolute movement before C with all ordinary transitions: a simple scale check.
 all_delta=[]
 rounds=sorted(st)
 for a,b in zip(rounds,rounds[1:]):
  if b==a+1: all_delta.append(tuple(st[b][d]-st[a][d] for d in range(4)))
 print('ABS MOVEMENT RATIO / median |delta| before residual divided by all-history median |delta|')
 base=[med([abs(x[d]) for x in all_delta]) or 1e-9 for d in range(4)]
 for lag in range(-5,0):
  ratios=[]
  for d in range(4):
   vals=[abs(ds[d]) for _,p in paths for l,ds in p if l==lag]; ratios.append(med(vals)/base[d])
  print(f'lag {lag}: '+','.join(f'{AX[d]}={ratios[d]:.2f}x' for d in range(4)))
if __name__=='__main__': main()
