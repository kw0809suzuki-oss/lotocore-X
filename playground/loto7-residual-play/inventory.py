"""Inventory what the current LOTO7 State/Transition observation can and cannot absorb.
No prediction claim. No lookahead: every target is judged only by prior transitions.

Observables: centroid, variance, range, max_gap.
For each current draw, find K similar prior States in robust-normalized 4D space.
Their realized next-state deltas define the local transition tendency.
We measure target surprise per axis and jointly.

A/B/C are descriptive shelves, not truth labels:
 A = within robust local tendency
 B = unusual/boundary but still near prior tendency
 C = strong residual across the current observation axes
"""
from __future__ import annotations
import math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fetch_loto7 import fetch

K=20
MIN_HISTORY=80
WARM_ERRORS=40
B_Z=2.0
C_Z=3.0


def med(x): return statistics.median(x)
def mad(x):
    m=med(x); return med([abs(v-m) for v in x]) or 1e-9
def rs(x): return max(1e-9,1.4826*mad(x))

def obs(nums):
    xs=sorted(nums); mu=sum(xs)/7
    var=sum((x-mu)**2 for x in xs)/7
    gaps=[b-a for a,b in zip(xs,xs[1:])]
    return (mu,var,xs[-1]-xs[0],max(gaps))

def ndist(a,b,sc): return math.sqrt(sum(((x-y)/s)**2 for x,y,s in zip(a,b,sc)))
def delta(a,b): return tuple(y-x for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))

def main():
    df=fetch(10000)
    rows=[]
    for _,r in df.iterrows():
        nums=[int(r[f'n{i}']) for i in range(1,8)]
        rows.append((int(r['round']),obs(nums),nums))
    historical_errors=[]; out=[]
    for i in range(MIN_HISTORY,len(rows)-1):
        hs=[x[1] for x in rows[:i+1]]
        sc=tuple(rs([s[d] for s in hs]) for d in range(4))
        cur=rows[i][1]
        near=[j for _,j in sorted((ndist(cur,rows[j][1],sc),j) for j in range(i))[:K]]
        ds=[delta(rows[j][1],rows[j+1][1]) for j in near]
        pred=add(cur,tuple(med([d[a] for d in ds]) for a in range(4)))
        actual=rows[i+1][1]
        err=ndist(pred,actual,sc)
        # Per-axis normalized miss makes the type of unexplained remainder visible.
        axis=tuple(abs(actual[d]-pred[d])/sc[d] for d in range(4))
        label='NA'; bz=cz=None
        if len(historical_errors)>=WARM_ERRORS:
            m=med(historical_errors); s=rs(historical_errors)
            bz=m+B_Z*s; cz=m+C_Z*s
            label='C' if err>cz else ('B' if err>bz else 'A')
        out.append((rows[i][0],rows[i+1][0],err,label,axis,pred,actual,bz,cz))
        historical_errors.append(err)
    eligible=[x for x in out if x[3]!='NA']
    print('LOTO7 THEORY INVENTORY / no lookahead')
    print(f'draws={len(rows)} rounds={rows[0][0]}..{rows[-1][0]} eligible={len(eligible)} K={K}')
    print('axes=centroid,variance,range,max_gap; shelves A<=2 robust sigma, B<=3, C>3 relative to prior joint-error history')
    for lab in 'ABC':
        z=[x for x in eligible if x[3]==lab]
        print(f'{lab}={len(z)} rate={100*len(z)/len(eligible):.2f}%')
    cs=[x for x in eligible if x[3]=='C']
    print('C RESIDUALS')
    for x in cs:
        print(f'{x[0]}->{x[1]} err={x[2]:.3f} Cthr={x[8]:.3f} axis_z(mu,var,range,maxgap)='+','.join(f'{v:.2f}' for v in x[4]))
    if cs:
        gaps=[b[1]-a[1] for a,b in zip(cs,cs[1:])]
        print(f'C gaps median={med(gaps) if gaps else "NA"} mean={(sum(gaps)/len(gaps) if gaps else "NA")} min={min(gaps) if gaps else "NA"} max={max(gaps) if gaps else "NA"}')
    # Show largest non-C boundary cases too, to avoid pretending the C cut is ontological.
    print('TOP BOUNDARY/RESIDUAL ERRORS')
    for x in sorted(eligible,key=lambda z:z[2],reverse=True)[:20]:
        print(f'{x[0]}->{x[1]} shelf={x[3]} err={x[2]:.3f} axis='+','.join(f'{v:.2f}' for v in x[4]))

if __name__=='__main__': main()
