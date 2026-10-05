"""LOTO7 State-transition inventory. Observation only; no prediction claim; no lookahead."""
from __future__ import annotations
import math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fetch_loto7 import fetch
K=20; MIN_HISTORY=80; WARM_ERRORS=40; B_Z=2.0; C_Z=3.0
AX=('mu','var','range','maxgap')
def med(x): return statistics.median(x)
def mad(x):
    m=med(x); return med([abs(v-m) for v in x]) or 1e-9
def rs(x): return max(1e-9,1.4826*mad(x))
def obs(nums):
    xs=sorted(nums); mu=sum(xs)/7; var=sum((x-mu)**2 for x in xs)/7
    gaps=[b-a for a,b in zip(xs,xs[1:])]
    return (mu,var,xs[-1]-xs[0],max(gaps))
def ndist(a,b,sc): return math.sqrt(sum(((x-y)/s)**2 for x,y,s in zip(a,b,sc)))
def delta(a,b): return tuple(y-x for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def main():
    df=fetch(10000); rows=[]
    for _,r in df.iterrows():
        nums=[int(r[f'n{i}']) for i in range(1,8)]; rows.append((int(r['round']),obs(nums),nums))
    hist=[]; out=[]
    for i in range(MIN_HISTORY,len(rows)-1):
        hs=[x[1] for x in rows[:i+1]]; sc=tuple(rs([s[d] for s in hs]) for d in range(4)); cur=rows[i][1]
        near=[j for _,j in sorted((ndist(cur,rows[j][1],sc),j) for j in range(i))[:K]]
        ds=[delta(rows[j][1],rows[j+1][1]) for j in near]
        pred=add(cur,tuple(med([d[a] for d in ds]) for a in range(4))); actual=rows[i+1][1]
        err=ndist(pred,actual,sc); axis=tuple(abs(actual[d]-pred[d])/sc[d] for d in range(4))
        label='NA'; bz=cz=None
        if len(hist)>=WARM_ERRORS:
            m=med(hist); s=rs(hist); bz=m+B_Z*s; cz=m+C_Z*s; label='C' if err>cz else ('B' if err>bz else 'A')
        out.append((rows[i][0],rows[i+1][0],err,label,axis,pred,actual,bz,cz)); hist.append(err)
    eligible=[x for x in out if x[3]!='NA']; cs=[x for x in eligible if x[3]=='C']
    print('LOTO7 THEORY INVENTORY / no lookahead')
    print(f'draws={len(rows)} rounds={rows[0][0]}..{rows[-1][0]} eligible={len(eligible)} K={K}')
    for lab in 'ABC':
        z=[x for x in eligible if x[3]==lab]; print(f'{lab}={len(z)} rate={100*len(z)/len(eligible):.2f}%')
    print('C RESIDUALS')
    for x in cs: print(f'{x[0]}->{x[1]} err={x[2]:.3f} Cthr={x[8]:.3f} axis_z(mu,var,range,maxgap)='+','.join(f'{v:.2f}' for v in x[4]))
    if cs:
        gaps=[b[1]-a[1] for a,b in zip(cs,cs[1:])]; print(f'C gaps median={med(gaps) if gaps else "NA"} mean={(sum(gaps)/len(gaps) if gaps else "NA")} min={min(gaps) if gaps else "NA"} max={max(gaps) if gaps else "NA"}')
    print('MOVING RESIDUAL WINDOWS / 30 eligible transitions, step=10')
    # A moving view of both residual density and which observable carries the largest miss.
    for end in range(30,len(eligible)+1,10):
        w=eligible[end-30:end]; c=[x for x in w if x[3]=='C']; b=[x for x in w if x[3]=='B']
        means=[sum(x[4][d] for x in w)/len(w) for d in range(4)]; dom=max(range(4),key=lambda d:means[d])
        cdom=[0]*4
        for x in c: cdom[max(range(4),key=lambda d:x[4][d])]+=1
        caxis=AX[max(range(4),key=lambda d:cdom[d])] if c else '-'
        print(f'{w[0][1]}..{w[-1][1]} C={len(c):2d} B={len(b):2d} C_rate={100*len(c)/len(w):5.1f}% mean_dom={AX[dom]} C_dom={caxis}')
    print('RESIDUAL DOMINANT-AXIS SEQUENCE')
    for x in cs:
        d=max(range(4),key=lambda q:x[4][q]); print(f'{x[1]}:{AX[d]}({x[4][d]:.2f}z)',end=' ')
    print()
if __name__=='__main__': main()
