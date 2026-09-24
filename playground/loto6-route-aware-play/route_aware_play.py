from __future__ import annotations

import csv, io, json, math
from collections import Counter
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_route_aware_play_v0.json")

SEED = 20260925
BURN = 150
POOL = 18
CENTER = 37 / 6
CENTER_BAND = 0.25
K_SHAPE = 15
MIN_ROUTE_MATCH = 8

PHASES = ["Flow","Break","Uncertainty","Re-formation"]

def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X route-aware-play"})
    r.raise_for_status()
    try: text = r.content.decode("cp932")
    except UnicodeDecodeError: text = r.content.decode("utf-8")
    rows=[]
    rd=csv.reader(io.StringIO(text)); next(rd,None)
    for row in rd:
        try:
            rnd=int(row[0]); nums=[int(row[2+i]) for i in range(6)]
        except Exception:
            continue
        if len(nums)==6 and len(set(nums))==6 and all(1<=n<=43 for n in nums):
            rows.append((rnd, sorted(nums)))
    rows.sort(key=lambda x:x[0])
    if len(rows)<1000: raise RuntimeError(len(rows))
    return rows

def draw_shape(draw):
    a=np.asarray(draw,dtype=float)
    return float(a.mean()), float(a.var())

def exact_random18():
    den=math.comb(43,6)
    pmf={k: math.comb(POOL,k)*math.comb(43-POOL,6-k)/den for k in range(7)}
    return {
        "mean_hits":6*POOL/43,
        "p_ge_3":sum(v for k,v in pmf.items() if k>=3),
        "p_ge_4":sum(v for k,v in pmf.items() if k>=4),
        "p_ge_5":sum(v for k,v in pmf.items() if k>=5),
    }

def classify_phase(A, prev_A, prev_label):
    x=A-CENTER
    if abs(x)<=CENTER_BAND: return "Uncertainty"
    dA=0.0 if prev_A is None else A-prev_A
    centerward=(x*dA)<0
    if centerward and prev_label in {"Break","Uncertainty"}: return "Re-formation"
    if centerward: return "Flow"
    return "Break"

def motion_family(dmu,dvar):
    am,av=abs(dmu),abs(dvar); t=.35
    if am<=t and av<=t: return "Hold"
    if av>am*1.25: return "Expand" if dvar>0 else "Contract"
    if am>av*1.25: return "Shift-R" if dmu>0 else "Shift-L"
    if dvar>0: return "Shift+Expand"
    return "Shift+Contract"

def normalize(v):
    a=np.asarray(v,dtype=float)
    lo,hi=float(a.min()),float(a.max())
    if hi<=lo: return np.zeros_like(a)
    return (a-lo)/(hi-lo)

def top18(score, salt):
    rng=np.random.default_rng(SEED + 1000003*(salt+1))
    jitter=rng.random(43)*1e-9
    order=np.argsort(-(np.asarray(score,dtype=float)+jitter), kind="stable")
    return [int(x+1) for x in order[:POOL]]

def summarize(hits):
    a=np.asarray(hits,dtype=float)
    return {
        "n":int(len(a)),
        "mean_hits":float(a.mean()),
        "p_ge_3":float(np.mean(a>=3)),
        "p_ge_4":float(np.mean(a>=4)),
        "p_ge_5":float(np.mean(a>=5)),
        "max_hits":int(a.max()),
        "hist":{str(k):int(np.sum(a==k)) for k in range(7)}
    }

def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))

def build_states(draws):
    draws=np.asarray(draws,dtype=int)
    shapes=[draw_shape(d) for d in draws]

    # Use exact-uniform shape moments estimated from all 43 choose 6 structure via a large deterministic sample.
    rng=np.random.default_rng(SEED)
    u=rng.random((120000,43))
    idx=np.argpartition(u,6,axis=1)[:,:6]+1
    sx=np.sort(idx,axis=1)
    smu=sx.mean(axis=1); sv=sx.var(axis=1)
    cal={"mu_mean":float(smu.mean()),"mu_sd":float(smu.std(ddof=1)),
         "var_mean":float(sv.mean()),"var_sd":float(sv.std(ddof=1))}

    ages=np.zeros(43,dtype=float)
    prev_A=None; prev_phase="Uncertainty"
    states=[]

    for i,draw in enumerate(draws):
        A=float(ages.mean())
        p=classify_phase(A,prev_A,prev_phase)
        if i==0:
            mu,vv=cal["mu_mean"],cal["var_mean"]
        else:
            mu,vv=shapes[i-1]
        z=np.array([(mu-cal["mu_mean"])/cal["mu_sd"], (vv-cal["var_mean"])/cal["var_sd"]],dtype=float)
        if states:
            dz=z-states[-1]["z"]
            pmotion=motion_family(float(dz[0]),float(dz[1]))
            pphase=states[-1]["phase"]
        else:
            pmotion="None"; pphase="None"
        states.append({"phase":p,"prev_phase":pphase,"prev_motion":pmotion,"z":z,"A":A})

        nxt=ages+1
        for n in draw: nxt[int(n)-1]=0
        ages=nxt; prev_A=A; prev_phase=p

    # Next-draw state after the latest observed draw.
    i=len(draws)
    A=float(ages.mean())
    p=classify_phase(A,prev_A,prev_phase)
    mu,vv=shapes[-1]
    z=np.array([(mu-cal["mu_mean"])/cal["mu_sd"], (vv-cal["var_mean"])/cal["var_sd"]],dtype=float)
    dz=z-states[-1]["z"]
    next_state={"phase":p,"prev_phase":states[-1]["phase"],
                "prev_motion":motion_family(float(dz[0]),float(dz[1])),"z":z,"A":A}
    return states,next_state,cal

def shape_score(states, draws, i):
    # strict-forward: prior targets j < i only
    if i < 2: return np.zeros(43,dtype=float), 0
    z=np.asarray([s["z"] for s in states[:i]],dtype=float)
    cur=states[i]["z"]
    d=np.sqrt(np.sum((z-cur)**2,axis=1))
    k=min(K_SHAPE,len(d))
    near=np.argpartition(d,k-1)[:k]
    counts=np.zeros(43,dtype=float)
    for j in near:
        for n in draws[j]:
            counts[int(n)-1]+=1
    return normalize(counts), int(k)

def shape_score_next(states, next_state, draws):
    z=np.asarray([s["z"] for s in states],dtype=float)
    cur=next_state["z"]
    d=np.sqrt(np.sum((z-cur)**2,axis=1))
    k=min(K_SHAPE,len(d))
    near=np.argpartition(d,k-1)[:k]
    counts=np.zeros(43,dtype=float)
    for j in near:
        for n in draws[j]:
            counts[int(n)-1]+=1
    return normalize(counts), [int(j) for j in near]

def route_score(states, draws, i):
    s=states[i]
    tiers=[
        ("phase+prev_phase+prev_motion", lambda x: x["phase"]==s["phase"] and x["prev_phase"]==s["prev_phase"] and x["prev_motion"]==s["prev_motion"]),
        ("phase+prev_motion", lambda x: x["phase"]==s["phase"] and x["prev_motion"]==s["prev_motion"]),
        ("phase+prev_phase", lambda x: x["phase"]==s["phase"] and x["prev_phase"]==s["prev_phase"]),
        ("phase", lambda x: x["phase"]==s["phase"]),
    ]
    for name,pred in tiers:
        js=[j for j in range(1,i) if pred(states[j])]
        if len(js)>=MIN_ROUTE_MATCH or name=="phase":
            counts=np.zeros(43,dtype=float)
            for j in js:
                for n in draws[j]: counts[int(n)-1]+=1
            return normalize(counts),name,len(js)
    return np.zeros(43,dtype=float),"none",0

def route_score_next(states, next_state, draws):
    s=next_state
    tiers=[
        ("phase+prev_phase+prev_motion", lambda x: x["phase"]==s["phase"] and x["prev_phase"]==s["prev_phase"] and x["prev_motion"]==s["prev_motion"]),
        ("phase+prev_motion", lambda x: x["phase"]==s["phase"] and x["prev_motion"]==s["prev_motion"]),
        ("phase+prev_phase", lambda x: x["phase"]==s["phase"] and x["prev_phase"]==s["prev_phase"]),
        ("phase", lambda x: x["phase"]==s["phase"]),
    ]
    for name,pred in tiers:
        js=[j for j in range(1,len(states)) if pred(states[j])]
        if len(js)>=MIN_ROUTE_MATCH or name=="phase":
            counts=np.zeros(43,dtype=float)
            for j in js:
                for n in draws[j]: counts[int(n)-1]+=1
            return normalize(counts),name,len(js),js
    return np.zeros(43,dtype=float),"none",0,[]

def random_pool(i):
    rng=np.random.default_rng(SEED + 7919*(i+1))
    return sorted(int(x) for x in rng.choice(np.arange(1,44),size=POOL,replace=False))

def main():
    rows=fetch_history()
    rounds=[r for r,_ in rows]
    draws=np.asarray([x for _,x in rows],dtype=int)
    states,next_state,cal=build_states(draws)

    hits={"shape_only":[],"route_only":[],"hybrid":[],"random":[]}
    fallback=Counter()

    for i in range(BURN,len(draws)):
        ss,_=shape_score(states,draws,i)
        rs,tier,nmatch=route_score(states,draws,i)
        fallback[tier]+=1
        pools={
            "shape_only":top18(ss,i),
            "route_only":top18(rs,i+10000),
            "hybrid":top18(0.55*ss+0.45*rs,i+20000),
            "random":random_pool(i),
        }
        for k,p in pools.items():
            hits[k].append(overlap(p,draws[i]))

    ss,near=shape_score_next(states,next_state,draws)
    rs,tier,nmatch,route_js=route_score_next(states,next_state,draws)
    next_pools={
        "shape_only":top18(ss,len(draws)+1),
        "route_only":top18(rs,len(draws)+10001),
        "hybrid":top18(0.55*ss+0.45*rs,len(draws)+20001),
    }

    out={
        "probe":"LOTO6 Route-Aware Play v0",
        "source":URL,
        "rounds":[rounds[0],rounds[-1]],
        "draws":len(rows),
        "burn":BURN,
        "pool_size":POOL,
        "question":"Can the current Phase/Shape/Route map be used as a light exploratory tilt for an 18-number candidate pool?",
        "rule":{
            "shape":"15 nearest prior pre-draw states in centroid x variance space; next-draw number frequency",
            "route":"strict-forward frequency after matching current phase + previous phase + previous Shape Motion; fallback only when <8 matches",
            "hybrid":"0.55 shape + 0.45 route score, fixed before run",
            "future_leakage":"none by construction for historical evaluation; only j < target i contributes"
        },
        "random18_exact":exact_random18(),
        "historical_strict_forward":{
            "shape_only":summarize(hits["shape_only"]),
            "route_only":summarize(hits["route_only"]),
            "hybrid":summarize(hits["hybrid"]),
            "deterministic_random_sample":summarize(hits["random"]),
            "route_context_tiers_used":dict(fallback),
        },
        "latest_map_position":{
            "latest_observed_round":rounds[-1],
            "next_target_round":rounds[-1]+1,
            "phase":next_state["phase"],
            "previous_phase":next_state["prev_phase"],
            "previous_shape_motion":next_state["prev_motion"],
            "mean_age_A":float(next_state["A"]),
            "route_context_tier":tier,
            "route_context_matches":nmatch,
        },
        "next_18_play":{
            "shape_only":next_pools["shape_only"],
            "route_only":next_pools["route_only"],
            "map_shape_route_hybrid":next_pools["hybrid"],
        },
        "boundary":[
            "Play probe only. Route-aware rule was designed after exploratory work on the same historical corpus.",
            "Historical uplift is not an untouched estimate of future lottery advantage.",
            "No claim is made that lottery draws cease to be random or that these 18 numbers are more likely in a validated prospective sense.",
            "The useful question is whether this map gives a coherent candidate-space tilt that can later be frozen and tested prospectively."
        ]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
