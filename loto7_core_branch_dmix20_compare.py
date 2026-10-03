from __future__ import annotations
import json, math, os, random, statistics
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

NUMS=list(range(1,38))
WORLD_IDS=list(range(21001,21051))
DRAWS=696
WARMUP=100
SEEDS=[20261012,20261013,20261014,20261015,20261016,20261017]
RANDOM_REPS=200
OUT=Path("results/loto7_core_branch_dmix20_compare.json")

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

def structured_core(pool18,seed,n_tickets=20):
    remaining=Counter(core_degrees(pool18,n_tickets))
    pair_count=defaultdict(int)
    rank={n:i for i,n in enumerate(pool18)}
    rng=random.Random(seed); out=[]
    for i in range(n_tickets):
        tickets_left=n_tickets-i
        chosen=[n for n,c in remaining.items() if c==tickets_left]
        if len(chosen)>7: raise RuntimeError("infeasible CORE degree schedule")
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
    totals={s:dict(core=0,branch=0,dmix=0,points=0) for s in SEEDS}
    for i in range(WARMUP,DRAWS):
        actual=draws[i]
        b20=branch_from_history(draws,states,trans,i,20)
        if b20 is not None:
            core18=core_rank(longc,recc,last,i)
            outside=[n for n in NUMS if n not in core18]
            for s in SEEDS:
                base=(i+1)*100_003+s
                c20=structured_core(core18,base+101,20)
                d20=(bridge_independent(core18,outside,base+4051,4,10)
                     +bridge_independent(core18,outside,base+5051,5,10))
                totals[s]["points"]+=1
                totals[s]["core"]+=hit5(c20,actual)
                totals[s]["branch"]+=hit5(b20,actual)
                totals[s]["dmix"]+=hit5(d20,actual)
        longq.append(actual); longc.update(actual)
        if len(longq)>100: longc.subtract(longq.popleft())
        recq.append(actual); recc.update(actual)
        if len(recq)>20: recc.subtract(recq.popleft())
        for n in actual:last[n]=i
    return wid,totals

def random20_success_probability():
    total=math.comb(37,7)
    good=sum(math.comb(7,j)*math.comb(30,7-j) for j in range(5,8))
    no=1.0
    for z in range(20):
        no*=(total-good-z)/(total-z)
    return 1-no

def main():
    with ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 4)) as ex:
        rows=list(ex.map(run_world,WORLD_IDS,chunksize=1))
    rows.sort()

    point_counts=[totals[SEEDS[0]]["points"] for _,totals in rows]
    p20=random20_success_probability()
    random_totals=[]
    for rs in range(2100101,2100101+RANDOM_REPS):
        rr=np.random.default_rng(rs)
        random_totals.append(sum(int(rr.binomial(n,p20)) for n in point_counts))
    rmed=statistics.median(random_totals)

    families=[]
    for s in SEEDS:
        vals={name:sum(totals[s][name] for _,totals in rows) for name in ("core","branch","dmix")}
        families.append({
            "seed":s,
            "evaluation_points":sum(point_counts),
            "core20_union":vals["core"],
            "branch20_union":vals["branch"],
            "dmix20_union":vals["dmix"],
            "random20_median":rmed,
            "core_minus_random":vals["core"]-rmed,
            "branch_minus_random":vals["branch"]-rmed,
            "dmix_minus_random":vals["dmix"]-rmed,
        })

    out={
        "experiment":"loto7_core_branch_dmix20_compare_v1",
        "world_ids":[WORLD_IDS[0],WORLD_IDS[-1]],
        "worlds":len(WORLD_IDS),
        "seeds":SEEDS,
        "ticket_count":20,
        "models":{
            "core20":"CORE18 ranking + rank-weight/pair-suppression 20-ticket allocator",
            "branch20":"existing transition-orbit BRANCH, 20 tickets",
            "dmix20":"10 tickets 4+3 and 10 tickets 5+2, independent-slot seeds",
            "random20":"uniform random 20-ticket null distribution",
        },
        "random20":{
            "repeats":RANDOM_REPS,
            "success_probability_per_point":p20,
            "median":rmed,
            "min":min(random_totals),
            "max":max(random_totals),
        },
        "families":families,
        "aggregate":{
            "core_mean":statistics.mean(f["core20_union"] for f in families),
            "branch_mean":statistics.mean(f["branch20_union"] for f in families),
            "dmix_mean":statistics.mean(f["dmix20_union"] for f in families),
            "random_median":rmed,
            "core_diff_mean":statistics.mean(f["core_minus_random"] for f in families),
            "branch_diff_mean":statistics.mean(f["branch_minus_random"] for f in families),
            "dmix_diff_mean":statistics.mean(f["dmix_minus_random"] for f in families),
        },
        "boundary":"Observed Calculation only; no causal or future-lottery claim."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
