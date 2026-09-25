from __future__ import annotations

import csv, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_route_corridor_v0.json")

SEED = 20260925
BURN = 150
POOL = 18
WIDTHS = [90, 180, 360]
PRIMARY_WIDTH = 180
MIN_BRANCH = 8
NULL_WORLDS = 8

CENTER = 37 / 6
CENTER_BAND = 0.25

MU_MEAN = 22.0
MU_SD = 4.755114206198086
VAR_MEAN = 131.38888888573075
VAR_SD = 54.75268470147889


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X route-corridor-v0"})
    r.raise_for_status()
    try:
        text = r.content.decode("cp932")
    except UnicodeDecodeError:
        text = r.content.decode("utf-8")
    rows = []
    rd = csv.reader(io.StringIO(text))
    next(rd, None)
    for row in rd:
        try:
            rnd = int(row[0])
            nums = [int(row[2+i]) for i in range(6)]
        except Exception:
            continue
        if len(nums)==6 and len(set(nums))==6 and all(1<=n<=43 for n in nums):
            rows.append((rnd, sorted(nums)))
    rows.sort(key=lambda x:x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few rows: {len(rows)}")
    return rows


def random_world(rng, n):
    u = rng.random((n,43))
    idx = np.argpartition(u, 6, axis=1)[:,:6] + 1
    return np.sort(idx, axis=1)


def draw_shape(draw):
    a = np.asarray(draw, dtype=float)
    return float(a.mean()), float(a.var())


def classify_phase(A, prev_A, prev_label):
    x = A - CENTER
    if abs(x) <= CENTER_BAND:
        return "Uncertainty"
    dA = 0.0 if prev_A is None else A - prev_A
    centerward = (x * dA) < 0
    if centerward and prev_label in {"Break","Uncertainty"}:
        return "Re-formation"
    if centerward:
        return "Flow"
    return "Break"


def motion_family(dmu, dvar):
    am,av = abs(dmu),abs(dvar)
    t=.35
    if am<=t and av<=t: return "Hold"
    if av>am*1.25: return "Expand" if dvar>0 else "Contract"
    if am>av*1.25: return "Shift-R" if dmu>0 else "Shift-L"
    return "Shift+Expand" if dvar>0 else "Shift+Contract"


def exact_random18():
    den = math.comb(43,6)
    pmf = {k:math.comb(POOL,k)*math.comb(43-POOL,6-k)/den for k in range(7)}
    return {
        "mean_hits": 6*POOL/43,
        "p_ge_3": sum(v for k,v in pmf.items() if k>=3),
        "p_ge_4": sum(v for k,v in pmf.items() if k>=4),
        "p_ge_5": sum(v for k,v in pmf.items() if k>=5),
    }


def build_states(draws):
    draws = np.asarray(draws, dtype=int)
    shapes = [draw_shape(d) for d in draws]
    ages = np.zeros(43, dtype=float)
    prev_A = None
    prev_phase = "Uncertainty"
    states = []

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        phase = classify_phase(A, prev_A, prev_phase)

        if i == 0:
            zmu, zv = 0.0, 0.0
        else:
            mu, vv = shapes[i-1]
            zmu = (mu-MU_MEAN)/MU_SD
            zv = (vv-VAR_MEAN)/VAR_SD

        dA = 0.0 if prev_A is None else A-prev_A
        if states:
            dzmu = zmu-states[-1]["zmu"]
            dzv = zv-states[-1]["zv"]
            pphase = states[-1]["phase"]
        else:
            dzmu = dzv = 0.0
            pphase = "None"

        states.append({
            "A":A,
            "dA":dA,
            "phase":phase,
            "prev_phase":pphase,
            "zmu":zmu,
            "zv":zv,
            "dzmu":dzmu,
            "dzv":dzv,
            "motion":motion_family(dzmu,dzv) if i else "None",
        })

        ages += 1
        ages[np.asarray(draw,dtype=int)-1] = 0
        prev_A = A
        prev_phase = phase

    # next state after latest observed draw
    A = float(ages.mean())
    phase = classify_phase(A, prev_A, prev_phase)
    mu,vv = shapes[-1]
    zmu = (mu-MU_MEAN)/MU_SD
    zv = (vv-VAR_MEAN)/VAR_SD
    dA = A-prev_A
    dzmu = zmu-states[-1]["zmu"]
    dzv = zv-states[-1]["zv"]
    next_state = {
        "A":A,
        "dA":dA,
        "phase":phase,
        "prev_phase":states[-1]["phase"],
        "zmu":zmu,
        "zv":zv,
        "dzmu":dzmu,
        "dzv":dzv,
        "motion":motion_family(dzmu,dzv),
    }
    return states,next_state


def oriented_age_velocity(states, t):
    if t <= 0:
        return 0.0
    prev_x = states[t-1]["A"] - CENTER
    # positive = moving toward equilibrium, negative = moving away
    return -np.sign(prev_x) * states[t]["dA"]


def corridor_signature(states, i):
    # A "road" signature: recent movement only, not an exact absolute point.
    # Three successive transitions, mirrored around the age equilibrium so
    # left/right excursions can share the same corridor geometry.
    vals = []
    for lag in (2,1,0):
        t = i-lag
        if t <= 0:
            vals.extend([0.0,0.0,0.0])
        else:
            vals.extend([
                float(oriented_age_velocity(states,t)),
                float(states[t]["dzmu"]),
                float(states[t]["dzv"]),
            ])
    return np.asarray(vals,dtype=float)


def calibrate_signature_scale(rng):
    world = random_world(rng, 30000)
    states,_ = build_states(world)
    X = np.asarray([corridor_signature(states,i) for i in range(500,len(states))],dtype=float)
    mean = X.mean(axis=0)
    sd = X.std(axis=0,ddof=1)
    sd[sd<=1e-12] = 1.0
    return mean,sd


def signature_matrix(states, mean, sd):
    return np.asarray([(corridor_signature(states,i)-mean)/sd for i in range(len(states))],dtype=float)


def top18_from_examples(draws, js, salt):
    counts = np.zeros(43,dtype=float)
    for j in js:
        for n in draws[j]:
            counts[int(n)-1] += 1
    rng = np.random.default_rng(SEED + 1000003*(salt+1))
    jitter = rng.random(43)*1e-9
    order = np.argsort(-(counts+jitter),kind="stable")
    return [int(x+1) for x in order[:POOL]]


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def summarize(hits):
    if not hits:
        return {"n":0}
    a=np.asarray(hits,dtype=float)
    return {
        "n":int(len(a)),
        "mean_hits":float(a.mean()),
        "p_ge_3":float(np.mean(a>=3)),
        "p_ge_4":float(np.mean(a>=4)),
        "p_ge_5":float(np.mean(a>=5)),
        "hist":{str(k):int(np.sum(a==k)) for k in range(7)},
    }


def shape_reference_pool(states,draws,i,salt):
    prior=np.arange(1,i,dtype=int)
    cur=np.array([states[i]["zmu"],states[i]["zv"]],dtype=float)
    arr=np.asarray([[states[j]["zmu"],states[j]["zv"]] for j in prior],dtype=float)
    d=np.sqrt(np.sum((arr-cur)**2,axis=1))
    k=min(15,len(prior))
    js=[int(prior[x]) for x in np.argsort(d,kind="stable")[:k]]
    return top18_from_examples(draws,js,salt)


def corridor_neighbors(states,X,i,width):
    # Same island (Phase), then nearest recent trajectory geometry.
    js=np.asarray([j for j in range(3,i) if states[j]["phase"]==states[i]["phase"]],dtype=int)
    if len(js)==0:
        return []
    d=np.sqrt(np.sum((X[js]-X[i])**2,axis=1))
    k=min(width,len(js))
    order=np.argsort(d,kind="stable")[:k]
    return [int(js[x]) for x in order]


def evaluate(draws, mean, sd):
    draws=np.asarray(draws,dtype=int)
    states0,next_state=build_states(draws)
    states=states0+[next_state]
    X=signature_matrix(states,mean,sd)

    shape_hits=[]
    by_width={
        w:{
            "corridor_hits":[],
            "oracle_hits":[],
            "oracle_valid":0,
            "oracle_invalid":0,
            "oracle_branch_n":[],
            "oracle_by_road":defaultdict(list),
        } for w in WIDTHS
    }

    for i in range(BURN,len(draws)):
        shape_pool=shape_reference_pool(states,draws,i,i)
        shape_hits.append(overlap(shape_pool,draws[i]))
        actual_road=states[i+1]["phase"]

        for wi,w in enumerate(WIDTHS):
            neigh=corridor_neighbors(states,X,i,w)
            corridor_pool=top18_from_examples(draws,neigh,i+10000+wi*100000)
            by_width[w]["corridor_hits"].append(overlap(corridor_pool,draws[i]))

            road_neigh=[j for j in neigh if states[j+1]["phase"]==actual_road]
            by_width[w]["oracle_branch_n"].append(len(road_neigh))
            if len(road_neigh) < MIN_BRANCH:
                by_width[w]["oracle_invalid"] += 1
                continue
            oracle_pool=top18_from_examples(draws,road_neigh,i+20000+wi*100000)
            h=overlap(oracle_pool,draws[i])
            by_width[w]["oracle_hits"].append(h)
            by_width[w]["oracle_valid"] += 1
            by_width[w]["oracle_by_road"][actual_road].append(h)

    out={"shape18_reference":summarize(shape_hits),"widths":{}}
    for w in WIDTHS:
        b=by_width[w]
        arr=np.asarray(b["oracle_branch_n"],dtype=float)
        out["widths"][str(w)]={
            "corridor18_unconditioned":summarize(b["corridor_hits"]),
            "correct_road_corridor18_oracle":summarize(b["oracle_hits"]),
            "oracle_valid":b["oracle_valid"],
            "oracle_invalid":b["oracle_invalid"],
            "oracle_branch_examples":{
                "mean":float(arr.mean()),
                "median":float(np.median(arr)),
                "q10":float(np.quantile(arr,.1)),
                "q90":float(np.quantile(arr,.9)),
            },
            "oracle_by_actual_road":{k:summarize(v) for k,v in sorted(b["oracle_by_road"].items())},
        }
    return out


def current_scenarios(draws,rounds,mean,sd):
    draws=np.asarray(draws,dtype=int)
    states0,next_state=build_states(draws)
    states=states0+[next_state]
    X=signature_matrix(states,mean,sd)
    i=len(draws)

    # Traffic shares stay from the same coarse intersection because that is the
    # user-facing map question. Candidate material comes from a broader corridor.
    coarse=[
        j for j in range(3,i)
        if states[j]["phase"]==next_state["phase"]
        and states[j]["prev_phase"]==next_state["prev_phase"]
        and states[j]["motion"]==next_state["motion"]
    ]
    traffic=Counter(states[j+1]["phase"] for j in coarse)
    total=sum(traffic.values())

    widths={}
    for wi,w in enumerate(WIDTHS):
        neigh=corridor_neighbors(states,X,i,w)
        scenarios={}
        for road,count in sorted(traffic.items(),key=lambda kv:(-kv[1],kv[0])):
            road_neigh=[j for j in neigh if states[j+1]["phase"]==road]
            pool=top18_from_examples(draws,road_neigh,i+30000+wi*100000+len(scenarios))
            scenarios[road]={
                "traffic_count":int(count),
                "traffic_share":float(count/total) if total else 0.0,
                "corridor_examples_for_this_road":len(road_neigh),
                "candidate18_score_order":pool,
                "candidate18_sorted":sorted(pool),
                "source_rounds":[int(rounds[j]) for j in road_neigh],
            }

        names=list(scenarios)
        pairwise=[]
        for a in range(len(names)):
            for b in range(a+1,len(names)):
                A,B=names[a],names[b]
                sa=set(scenarios[A]["candidate18_sorted"])
                sb=set(scenarios[B]["candidate18_sorted"])
                pairwise.append({
                    "a":A,"b":B,
                    "overlap":len(sa&sb),
                    "only_a":sorted(sa-sb),
                    "only_b":sorted(sb-sa),
                })
        widths[str(w)]={
            "corridor_size":len(neigh),
            "road_scenarios":scenarios,
            "pairwise_candidate_overlap":pairwise,
        }

    return {
        "latest_observed_round":int(rounds[-1]),
        "target_round":int(rounds[-1]+1),
        "current_map":{
            "phase":next_state["phase"],
            "previous_phase":next_state["prev_phase"],
            "motion":next_state["motion"],
            "mean_age_A":float(next_state["A"]),
        },
        "same_coarse_intersection_examples":total,
        "traffic_distribution":[
            {"phase":p,"count":int(c),"share":float(c/total)}
            for p,c in sorted(traffic.items(),key=lambda kv:(-kv[1],kv[0]))
        ],
        "corridor_widths":widths,
        "primary_width":PRIMARY_WIDTH,
    }


def null_compare(actual,nulls,path):
    parts=path.split(".")
    def get(x):
        for p in parts:
            x=x[p]
        return float(x)
    a=get(actual)
    vals=np.asarray([get(n) for n in nulls],dtype=float)
    return {
        "actual":a,
        "B_mean":float(vals.mean()),
        "B_q025":float(np.quantile(vals,.025)),
        "B_q50":float(np.quantile(vals,.5)),
        "B_q975":float(np.quantile(vals,.975)),
        "actual_percentile":float(np.mean(vals<=a)),
    }


def main():
    rows=fetch_history()
    rounds=[r for r,_ in rows]
    draws=np.asarray([x for _,x in rows],dtype=int)

    rng=np.random.default_rng(SEED)
    mean,sd=calibrate_signature_scale(rng)

    actual=evaluate(draws,mean,sd)
    current=current_scenarios(draws,rounds,mean,sd)

    nulls=[]
    for _ in range(NULL_WORLDS):
        world=random_world(rng,len(draws))
        nulls.append(evaluate(world,mean,sd))

    watched={}
    for w in WIDTHS:
        for metric in ["mean_hits","p_ge_3"]:
            path=f"widths.{w}.correct_road_corridor18_oracle.{metric}"
            watched[path]=null_compare(actual,nulls,path)

    out={
        "probe":"LOTO6 Route Corridor v0",
        "source":URL,
        "rounds":[int(rounds[0]),int(rounds[-1])],
        "draws":len(rows),
        "burn":BURN,
        "pool_size":POOL,
        "question":"If the current position is treated as a broad movement corridor rather than a fixed coordinate, what happens to Candidate18 when the correct next road is chosen?",
        "corridor_definition":{
            "island":"same current Phase only",
            "road_signature":"last 3 transitions of [age motion oriented toward/away from equilibrium, delta shape centroid, delta shape variance]",
            "translation":"absolute shape position is intentionally omitted from corridor distance",
            "mirror":"age motion is mirrored around equilibrium so equivalent left/right excursions can share a corridor",
            "widths":WIDTHS,
            "branching":"first form a movement corridor, then partition that corridor by realized next Phase; no second nearest-state filter",
            "minimum_branch_examples":MIN_BRANCH,
        },
        "random18_exact":exact_random18(),
        "historical_strict_forward":actual,
        "B_world_comparison":watched,
        "current_2141_scenarios":current,
        "signature_calibration_from_B_world":{
            "mean":[float(x) for x in mean],
            "sd":[float(x) for x in sd],
        },
        "null":{
            "worlds":NULL_WORLDS,
            "seed":SEED,
            "generator":"independent exact-uniform 6-of-43 worlds",
        },
        "boundary":[
            "No Floot app changes were made.",
            "Correct-road results are oracle diagnostics because the actual next Phase is unavailable before the draw.",
            "Road identity is partly determined by the transition-causing draw, so B-world controls are required to distinguish mechanical conditioning from A-world structure.",
            "Three corridor widths are reported together; no width is selected after seeing the results.",
            "Historical A remains exploratory and is not a pristine future holdout."
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
