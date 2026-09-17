"""Look backward from each strong residual and compare its -2 -> -1 shape with ordinary targets. Observation only."""
import sys, statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fetch_loto7 import fetch
C_TARGETS=[158,159,160,229,407,426,472,504,521,537,556,669,677]
AX=('mu','var','range','maxgap')
def obs(nums):
 xs=sorted(nums); mu=sum(xs)/7; var=sum((x-mu)**2 for x in xs)/7; gaps=[b-a for a,b in zip(xs,xs[1:])]
 return (mu,var,xs[-1]-xs[0],max(gaps))
def med(x): return statistics.median(x)
def d(st,end): return tuple(st[end][i]-st[end-1][i] for i in range(4))
def pattern(st,t):
 # Candidate observed in residual aggregate: at -2 mu/range/var expand, then at -1 var/range contract.
 a=d(st,t-2); b=d(st,t-1)
 return a[0]>0 and a[1]>0 and a[2]>0 and b[1]<0 and b[2]<0
def main():
 df=fetch(10000); st={}
 for _,r in df.iterrows():
  n=[int(r[f'n{i}']) for i in range(1,8)]; st[int(r['round'])]=obs(n)
 paths=[]
 for t in C_TARGETS:
  p=[]
  for lag in range(-5,1):
   end=t+lag
   if end-1 in st and end in st:p.append((lag,d(st,end)))
  paths.append((t,p))
 print('PRE-RESIDUAL WALK / raw State deltas; residual target at lag 0')
 for lag in range(-5,1):
  vals=[[ds[q] for _,p in paths for l,ds in p if l==lag] for q in range(4)]
  print(f'lag {lag}: '+','.join(f'{AX[q]}={med(vals[q]):+.2f}' for q in range(4)))
 print('DIRECTION CONSISTENCY')
 for lag in (-2,-1):
  parts=[]
  for q in range(4):
   vals=[ds[q] for _,p in paths for l,ds in p if l==lag]; m=med(vals); s=1 if m>0 else (-1 if m<0 else 0)
   same=sum(((v>0)-(v<0))==s for v in vals)/len(vals)
   parts.append(f'{AX[q]}={same:.2f}')
  print(f'lag {lag}: '+','.join(parts))
 # Simple control: same exact sign-pattern at every target with enough history, excluding C targets.
 valid=[t for t in sorted(st) if t>=4 and t-3 in st and t not in C_TARGETS]
 c_hits=[t for t in C_TARGETS if pattern(st,t)]
 o_hits=[t for t in valid if pattern(st,t)]
 print('CONTROL / exact sign pattern: -2(mu,var,range)>0 then -1(var,range)<0')
 print(f'C: {len(c_hits)}/{len(C_TARGETS)} = {100*len(c_hits)/len(C_TARGETS):.1f}% targets={c_hits}')
 print(f'ordinary: {len(o_hits)}/{len(valid)} = {100*len(o_hits)/len(valid):.1f}%')
 # Looser core ignores mu, retaining only spread/range expand then contract.
 def loose(t):
  a=d(st,t-2); b=d(st,t-1); return a[1]>0 and a[2]>0 and b[1]<0 and b[2]<0
 c2=[t for t in C_TARGETS if loose(t)]; o2=[t for t in valid if loose(t)]
 print('CONTROL / loose spread-range pattern: -2(var,range)>0 then -1(var,range)<0')
 print(f'C: {len(c2)}/{len(C_TARGETS)} = {100*len(c2)/len(C_TARGETS):.1f}% targets={c2}')
 print(f'ordinary: {len(o2)}/{len(valid)} = {100*len(o2)/len(valid):.1f}%')
if __name__=='__main__':main()
