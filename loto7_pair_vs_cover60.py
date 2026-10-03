from __future__ import annotations
import json, math, os, random, statistics
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import numpy as np

NUMS=list(range(1,38))
WORLD_IDS=list(range(23001,23051))
DRAWS=696
WARMUP=100
SEEDS=[20261012,20261013,20261014,20261015,20261016,20261017]
RANDOM_REPS=200
OUT=Path("results/loto7_pair_vs_cover60.json")

PAIR_INDEX={p:i for i,p in enumerate(combinations(NUMS,2))}
NP=len(PAIR_INDEX)

def hit5(tickets,actual):
    a=set(actual)
    return int(any(len(a.intersection(t))>=5 for t in tickets))

def point_state(nums):
    return sum(nums)/7.0, nums[-1]-nums[0]

def transition(a,b):
    dc=b[0]-a[0]; dr=b[1]-a[1]
    return ('HIGH' if dc>0 else 'LOW' if dc<0 else 'FLAT')+'+'+('OPEN' if dr>0 else 'CLOSE' if dr<0 else 'SAME')

def branch_from_history(draws,states,trans,past_len,n_tickets):
    tr=trans[:past_len-1]
    if len(tr)<2:return None
    orbit=(tr[-2],tr[-1]); cur=states[past_len-1]; ex=[]
    for j in range(1,len(tr)-1):
        if tr[j-1]==orbit[0] and tr[j]==orbit[1]:
            ex.append((tr[j+1],states[j+1],draws[j+2],j+2))
    if len(ex)<n_tickets:return None
    counts=Counter(e[0] for e in ex); total=len(ex); rows=[]
    for lab,count in counts.items():
        raw=count*n_tickets/total
        rows.append([lab,count,raw,math.floor(raw)])
    left=n_tickets-sum(r[3] for r in rows)
    rows.sort(key=lambda r:(-(r[2]-r[3]),-r[1],r[0]))
    for z in range(left): rows[z][3]+=1
    def dist(st): return abs(st[0]-cur[0])/36+abs(st[1]-cur[1])/36
    out=[]
    for lab,count,raw,slots in rows:
        if slots<=0: continue
        pool=[e for e in ex if e[0]==lab]
        pool.sort(key=lambda e:(dist(e[1]),-e[3]))
        out.extend(tuple(e[2]) for e in pool[:slots])
    return out if len(out)==n_tickets else None

def core_rank(longc,recc,last,i):
    vals=[]
    for n in NUMS:
        gap=i if last[n] is None else i-1-last[n]
        score=.60*longc[n]/100+.25*recc[n]/20+.15*min(gap,12)/12
        vals.append((-score,n))
    vals.sort()
    return [n for _,n in vals[:18]]

def bridge_independent(core18,outside,seed,k_core,n):
    out=[]; seen=set(); k_out=7-k_core
    for slot in range(n):
        salt=0
        while True:
            rng=random.Random(seed+(slot+1)*1_000_003+salt*10_000_019)
            t=tuple(sorted(rng.sample(core18,k_core)+rng.sample(outside,k_out)))
            if t not in seen:
                seen.add(t); out.append(t); break
            salt+=1
    return out

def ticket_pairs(t):
    return [PAIR_INDEX[p] for p in combinations(t,2)]

def select_k(pool,target,k,global_sel_counts,global_pair_counts):
    rem=list(enumerate(pool)); out=[]
    pairlists=[ticket_pairs(t) for t in pool]
    while len(out)<k:
        best=None
        for idx,t in rem:
            usage_ex=0.0; usage_delta=0.0
            for n in t:
                before=global_sel_counts[n]; after=before+1
                usage_ex+=max(0.0,after-target[n])
                usage_delta+=abs(after-target[n])-abs(before-target[n])
            pair_pen=sum(global_pair_counts[p] for p in pairlists[idx])
            score=(usage_ex,pair_pen,usage_delta,idx)
            if best is None or score<best[0]: best=(score,idx,t)
        _,idx,t=best
        out.append(t); rem.remove((idx,t))
        for n in t: global_sel_counts[n]+=1
        for p in pairlists[idx]: global_pair_counts[p]+=1
    return out

def pair60(branch40,dmix40):
    full=branch40+dmix40
    counts=Counter(n for t in full for n in t)
    target={n:counts[n]*0.75 for n in NUMS}
    sc=Counter(); pc=[0]*NP
    sig=sum(sum(t) for t in full)
    if sig%2==0:
        bs=select_k(branch40,target,30,sc,pc); ds=select_k(dmix40,target,30,sc,pc)
    else:
        ds=select_k(dmix40,target,30,sc,pc); bs=select_k(branch40,target,30,sc,pc)
    return bs+ds

def cover60(branch40,dmix40):
    # Frozen covering baseline: preserve module balance 30/30.
    # Within each module greedily maximize newly covered 5-subsets.
    def pick30(pool):
        subsets=[set(combinations(t,5)) for t in pool]
        covered=set(); chosen=[]; rem=set(range(len(pool)))
        while len(chosen)<30:
            idx=max(rem,key=lambda i:(len(subsets[i]-covered),-i))
            chosen.append(pool[idx])
            covered.update(subsets[idx]); rem.remove(idx)
        return chosen
    return pick30(branch40)+pick30(dmix40)

def run_world(wid):
    rng=random.Random(wid)
    draws=[tuple(sorted(rng.sample(NUMS,7))) for _ in range(DRAWS)]
    states=[point_state(d) for d in draws]
    trans=[transition(states[k],states[k+1]) for k in range(DRAWS-1)]
    longq,recq=deque(),deque(); longc,recc=Counter(),Counter()
    last={n:None for n in NUMS}
    for j in range(WARMUP):
        d=draws[j]; longq.append(d); recq.append(d); longc.update(d); recc.update(d)
        if len(recq)>20: recc.subtract(recq.popleft())
        for n in d:last[n]=j
    totals={s:dict(raw80=0,pair60=0,cover60=0,points=0) for s in SEEDS}
    for i in range(WARMUP,DRAWS):
        actual=draws[i]
        b40=branch_from_history(draws,states,trans,i,40)
        if b40 is not None:
            core18=core_rank(longc,recc,last,i)
            outside=[n for n in NUMS if n not in core18]
            for s in SEEDS:
                base=(i+1)*100_003+s
                d40=(bridge_independent(core18,outside,base+4051,4,20)
                     +bridge_independent(core18,outside,base+5051,5,20))
                raw=b40+d40
                p60=pair60(b40,d40)
                c60=cover60(b40,d40)
                totals[s]["points"]+=1
                totals[s]["raw80"]+=hit5(raw,actual)
                totals[s]["pair60"]+=hit5(p60,actual)
                totals[s]["cover60"]+=hit5(c60,actual)
        longq.append(actual); longc.update(actual)
        if len(longq)>100: longc.subtract(longq.popleft())
        recq.append(actual); recc.update(actual)
        if len(recq)>20: recc.subtract(recq.popleft())
        for n in actual:last[n]=i
    return wid,totals

def random_p(n):
    total=math.comb(37,7)
    good=sum(math.comb(7,j)*math.comb(30,7-j) for j in range(5,8))
    no=1.0
    for z in range(n):
        no*=(total-good-z)/(total-z)
    return 1-no

def main():
    with ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 4)) as ex:
        rows=list(ex.map(run_world,WORLD_IDS,chunksize=1))
    rows.sort()
    points_by_world=[totals[SEEDS[0]]["points"] for _,totals in rows]
    p60=random_p(60)
    rvals=[]
    for rs in range(2300101,2300101+RANDOM_REPS):
        rr=np.random.default_rng(rs)
        rvals.append(sum(int(rr.binomial(n,p60)) for n in points_by_world))
    rmed=statistics.median(rvals)

    families=[]
    for s in SEEDS:
        vals={k:sum(t[s][k] for _,t in rows) for k in ("raw80","pair60","cover60")}
        families.append({
            "seed":s,
            "evaluation_points":sum(points_by_world),
            "raw80_union":vals["raw80"],
            "pair60_union":vals["pair60"],
            "cover60_union":vals["cover60"],
            "pair_minus_cover":vals["pair60"]-vals["cover60"],
            "pair_minus_random60":vals["pair60"]-rmed,
            "cover_minus_random60":vals["cover60"]-rmed
        })

    out={
        "experiment":"loto7_pair_vs_cover60_v1",
        "world_ids":[WORLD_IDS[0],WORLD_IDS[-1]],
        "worlds":len(WORLD_IDS),
        "seeds":SEEDS,
        "pool":"BRANCH40 + D-mix40",
        "pair60":"30+30 using frozen usage/pair score",
        "cover60":"30+30 greedy maximum new 5-subset coverage within each module",
        "random60":{"repeats":RANDOM_REPS,"median":rmed,"min":min(rvals),"max":max(rvals)},
        "families":families,
        "aggregate":{
            "raw80_mean":statistics.mean(f["raw80_union"] for f in families),
            "pair60_mean":statistics.mean(f["pair60_union"] for f in families),
            "cover60_mean":statistics.mean(f["cover60_union"] for f in families),
            "pair_minus_cover_mean":statistics.mean(f["pair_minus_cover"] for f in families),
            "pair_minus_random60_mean":statistics.mean(f["pair_minus_random60"] for f in families),
            "cover_minus_random60_mean":statistics.mean(f["cover_minus_random60"] for f in families)
        },
        "boundary":"Fresh Worlds. No outcome-informed selection, tuning, or adoption."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
