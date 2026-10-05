"""Freeze the round-695 Closure before the result exists.
Observation/play only; not a claim of predictive power.
Uses the same state/distance/analogue machinery as residual_walk.py.
"""
from __future__ import annotations

import math
import statistics
from fetch_loto7 import fetch

K = 15
ROBUST_Z = 3.0
MIN_HISTORY = 60


def state(nums):
    mu = sum(nums) / 7.0
    var = sum((x - mu) ** 2 for x in nums) / 7.0
    return mu, var


def med(xs): return statistics.median(xs)
def mad(xs):
    m = med(xs)
    return med([abs(x-m) for x in xs]) or 1e-9

def robust_scale(xs): return max(1e-9, 1.4826 * mad(xs))
def dist(a,b,sm,sv): return math.hypot((a[0]-b[0])/sm,(a[1]-b[1])/sv)


def main():
    df=fetch(10000)
    rows=[]
    for _,r in df.iterrows():
        nums=[int(r[f'n{i}']) for i in range(1,8)]
        rows.append((int(r['round']),state(nums),nums))
    assert rows[-1][0] == 694, f"freeze expected round 694 latest, got {rows[-1][0]}"

    # Reproduce all historical one-step errors with no lookahead.
    past_errors=[]
    for i in range(MIN_HISTORY,len(rows)-1):
        hs=[s for _,s,_ in rows[:i+1]]
        sm=robust_scale([s[0] for s in hs]); sv=robust_scale([s[1] for s in hs])
        cur=rows[i][1]
        near=[j for _,j in sorted((dist(cur,rows[j][1],sm,sv),j) for j in range(i))[:K]]
        dmu=med([rows[j+1][1][0]-rows[j][1][0] for j in near])
        dv=med([rows[j+1][1][1]-rows[j][1][1] for j in near])
        pred=(cur[0]+dmu,cur[1]+dv)
        past_errors.append(dist(pred,rows[i+1][1],sm,sv))

    hs=[s for _,s,_ in rows]
    sm=robust_scale([s[0] for s in hs]); sv=robust_scale([s[1] for s in hs])
    cur=rows[-1][1]
    near_pairs=sorted((dist(cur,rows[j][1],sm,sv),j) for j in range(len(rows)-1))[:K]
    near=[j for _,j in near_pairs]
    dmus=[rows[j+1][1][0]-rows[j][1][0] for j in near]
    dvs=[rows[j+1][1][1]-rows[j][1][1] for j in near]
    pred=(cur[0]+med(dmus),cur[1]+med(dvs))
    center=med(past_errors); sigma=robust_scale(past_errors); threshold=center+ROBUST_Z*sigma

    # Human-readable axis-aligned projection of the same normalized-distance Closure.
    mu_radius=threshold*sm; var_radius=threshold*sv
    print('LOTO7 ROUND 695 / FROZEN CLOSURE')
    print(f'latest=694 nums={rows[-1][2]} state mu={cur[0]:.4f} var={cur[1]:.4f}')
    print(f'K={K} analogue rounds={[rows[j][0] for j in near]}')
    print(f'predicted_state_center mu={pred[0]:.4f} var={pred[1]:.4f}')
    print(f'normalized residual boundary error={threshold:.4f} (median={center:.4f}, robust_sigma={sigma:.4f})')
    print(f'scales mu={sm:.4f} var={sv:.4f}')
    print(f'axis_projection mu=[{pred[0]-mu_radius:.4f},{pred[0]+mu_radius:.4f}] var=[{max(0,pred[1]-var_radius):.4f},{pred[1]+var_radius:.4f}]')
    print('AFTER 695: error = sqrt(((mu695-mu_center)/mu_scale)^2 + ((var695-var_center)/var_scale)^2)')
    print('closure_ratio = error / residual_boundary; <=1 inside, >1 residual-side')

if __name__=='__main__': main()
