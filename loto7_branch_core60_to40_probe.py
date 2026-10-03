from __future__ import annotations

import json, math, os, random, statistics
from collections import Counter, deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

NUMS=list(range(1,38))
WORLD_IDS=list(range(20001,20051))
DRAWS=696; WARMUP=100
SEEDS=[20261012,20261013,20261014,20261015,20261016,20261017]
RANDOM_REPS=200
OUT=Path("results/loto7_branch_core60_to40_probe.json")

PAIR_INDEX={}
k=0
for a in NUMS:
    for b in range(a+1,38):
        PAIR_INDEX[(a,b)]=k; k+=1
NP=k

def hit5(tickets,actual):
    aset=set(actual)
    return int(any(len(aset.intersection(t))>=5 for t in tickets))

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
    for z in range(left):rows[z][3]+=1
    def dist(st):return abs(st[0]-cur[0])/36+abs(st[1]-cur[1])/36
    out=[]
    for lab,count,raw,slots in rows:
        if slots<=0:continue
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

def core_degrees(pool18,n_tickets):
    total_slots=n_tickets*7
    deg=Counter({n:1 for n in pool18})
    remaining=total_slots-len(pool18)
    rank={n:i for i,n in enumerate(pool18)}
    weight={n:len(pool18)-i for i,n in enumerate(pool18)}
    for _ in range(remaining):
        candidates=[n for n in pool18 if deg[n]<n_tickets]
        pick=max(candidates,key=lambda n:(weight[n]/(deg[n]+1.0),-rank[n]))
        deg[pick]+=1
    return deg

def structured_core(pool18,seed,n_tickets):
    # Existing CORE10 allocator generalized only by ticket count.
    remaining=Counter(core_degrees(pool18,n_tickets))
    pair_count=defaultdict(int)
    rank={n:i for i,n in enumerate(pool18)}
    rng=random.Random(seed); out=[]
    for i in range(n_tickets):
        tickets_left=n_tickets-i
        chosen=[n for n,c in remaining.items() if c==tickets_left]
        if len(chosen)>7:raise RuntimeError("infeasible CORE degree schedule")
        while len(chosen)<7:
            candidates=[n for n,c in remaining.items() if c>0 and n not in chosen]
            rng.shuffle(candidates)
            pick=min(candidates,key=lambda n:(sum(pair_count[tuple(sorted((n,x)))] for x in chosen),-remaining[n],rank[n]))
            chosen.append(pick)
        t=tuple(sorted(chosen))
        for n in t:remaining[n]-=1
        for a in range(7):
            for b in range(a+1,7):
                pair_count[(t[a],t[b])]+=1
        out.append(t)
    return out

def ticket_pairs(t):
    return [PAIR_INDEX[(t[i],t[j])] for i in range(7) for j in range(i+1,7)]

def select20(pool30,target,global_sel_counts,global_pair_counts):
    rem=list(enumerate(pool30)); out=[]
    pairlists=[ticket_pairs(t) for t in pool30]
    while len(out)<20:
        best=None
        for idx,t in rem:
            usage_ex=0.0; usage_delta=0.0
            for n in t:
                before=global_sel_counts[n]; after=before+1
                usage_ex+=max(0.0,after-target[n])
                usage_delta+=abs(after-target[n])-abs(before-target[n])
            pair_pen=sum(global_pair_counts[p] for p in pairlists[idx])
            score=(usage_ex,pair_pen,usage_delta,idx)
            if best is None or score<best[0]:best=(score,idx,t)
        _,idx,t=best
        out.append(t); rem.remove((idx,t))
        for n in t:global_sel_counts[n]+=1
        for p in pairlists[idx]:global_pair_counts[p]+=1
    return out

def assembler40(branch30,core30):
    full=branch30+core30
    counts=Counter(n for t in full for n in t)
    target={n:counts[n]*2/3 for n in NUMS}
    sc=Counter(); pc=[0]*NP
    sig=sum(sum(t) for t in full)
    if sig%2==0:
        bs=select20(branch30,target,sc,pc); cs=select20(core30,target,sc,pc)
    else:
        cs=select20(core30,target,sc,pc); bs=select20(branch30,target,sc,pc)
    return bs+cs

def run_world(wid):
    rng=random.Random(wid)
    draws=[tuple(sorted(rng.sample(NUMS,7))) for _ in range(DRAWS)]
    states=[point_state(d) for d in draws]
    trans=[transition(states[k],states[k+1]) for k in range(DRAWS-1)]
    longq,recq=deque(),deque(); longc,recc=Counter(),Counter()
    last={n:None for n in NUMS}
    for j in range(WARMUP):
        d=draws[j]; longq.append(d); recq.append(d); longc.update(d); recc.update(d)
        if len(recq)>20:recc.subtract(recq.popleft())
        for n in d:last[n]=j
    counts={sf:[0,0] for sf in SEEDS}
    for i in range(WARMUP,DRAWS):
        target_draw=draws[i]
        b30=branch_from_history(draws,states,trans,i,30)
        if b30 is not None:
            core18=core_rank(longc,recc,last,i)
            for sf in SEEDS:
                base=(i+1)*100_003+sf
                c30=structured_core(core18,base+101,30)
                a40=assembler40(b30,c30)
                counts[sf][0]+=1
                counts[sf][1]+=hit5(a40,target_draw)
        longq.append(target_draw); longc.update(target_draw)
        if len(longq)>100:longc.subtract(longq.popleft())
        recq.append(target_draw); recc.update(target_draw)
        if len(recq)>20:recc.subtract(recq.popleft())
        for n in target_draw:last[n]=i
    return wid,counts

def random40_success_probability():
    total=math.comb(37,7)
    hit=sum(math.comb(7,j)*math.comb(30,7-j) for j in range(5,8))
    no=1.0
    for z in range(40):
        no*=(total-hit-z)/(total-z)
    return 1.0-no

def main():
    with ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 4)) as ex:
        rows=list(ex.map(run_world,WORLD_IDS,chunksize=1))
    rows.sort()
    world_n=[counts[SEEDS[0]][0] for _,counts in rows]
    p40=random40_success_probability()
    random_world_reps=[]
    for rs in range(2000101,2000101+RANDOM_REPS):
        r=np.random.default_rng(rs)
        random_world_reps.append([int(r.binomial(n,p40)) for n in world_n])
    random_totals=[sum(x) for x in random_world_reps]
    random_median=statistics.median(random_totals)
    families=[]
    for sf in SEEDS:
        by_world=[counts[sf][1] for _,counts in rows]
        total=sum(by_world)
        families.append({
            "seed":sf,
            "assembler40_union":total,
            "random40_median":random_median,
            "diff_vs_random_median":total-random_median,
            "random_reps_below_candidate":sum(total>x for x in random_totals),
            "random_reps_equal_candidate":sum(total==x for x in random_totals),
            "random_reps_above_candidate":sum(total<x for x in random_totals),
            "by_world":[{"world":wid,"comparable_points":counts[sf][0],"union":counts[sf][1]} for wid,counts in rows],
        })
    vals=[f["assembler40_union"] for f in families]
    diffs=[f["diff_vs_random_median"] for f in families]
    out={
        "experiment":"loto7_branch_core60_to40_probe_v1",
        "world_ids":[WORLD_IDS[0],WORLD_IDS[-1]],
        "worlds":len(WORLD_IDS),
        "seeds":SEEDS,
        "pool60":"BRANCH30 + LOTO CORE30",
        "loto_core30":"CORE18 score + existing rank-weight/pair-suppression allocator generalized only from 10 to 30 tickets; seed=base+101",
        "assembler40":"20 BRANCH + 20 CORE selected from Pool60; number-use preservation primary, pair reuse secondary; no outcome data",
        "random40":{
            "method":"exact per-point 40-ticket success probability + binomial world simulation",
            "repeats":RANDOM_REPS,
            "success_probability_per_point":p40,
            "total_distribution_median":random_median,
            "total_distribution_min":min(random_totals),
            "total_distribution_max":max(random_totals),
        },
        "families":families,
        "aggregate":{
            "assembler_union_mean":statistics.mean(vals),
            "assembler_union_median":statistics.median(vals),
            "diff_vs_random_median_by_seed":diffs,
            "diff_mean":statistics.mean(diffs),
            "diff_median":statistics.median(diffs),
            "above_random_median_seed_count":sum(x>0 for x in diffs),
            "equal_random_median_seed_count":sum(x==0 for x in diffs),
            "below_random_median_seed_count":sum(x<0 for x in diffs),
        },
        "boundary":"Observed Calculation only. No adoption or future-lottery claim.",
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
