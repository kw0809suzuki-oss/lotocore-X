from __future__ import annotations

import csv, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_consistency_threshold_v0.json")

SEED = 20260925
BURN = 150
CORRIDOR_WIDTH = 90  # frozen from prior Route Corridor v0; not retuned here
MIN_BRANCH = 8
FIT_NEIGHBORS = 8
POOL_SIZES = [18, 15, 12]
RETENTION = [1.00, 0.70, 0.50, 0.30, 0.20, 0.10]
NULL_WORLDS = 12

CENTER = 37 / 6
CENTER_BAND = 0.25

MU_MEAN = 22.0
MU_SD = 4.755114206198086
VAR_MEAN = 131.38888888573075
VAR_SD = 54.75268470147889

PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X consistency-threshold-v0"})
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
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 43 for n in nums):
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
    return -np.sign(prev_x) * states[t]["dA"]


def corridor_signature(states, i):
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


def corridor_with_distances(states, X, i):
    js = np.asarray([j for j in range(3,i) if states[j]["phase"] == states[i]["phase"]],dtype=int)
    if len(js) == 0:
        return [], {}
    d = np.sqrt(np.sum((X[js]-X[i])**2,axis=1))
    k = min(CORRIDOR_WIDTH,len(js))
    order = np.argsort(d,kind="stable")[:k]
    selected = [int(js[x]) for x in order]
    dist = {int(js[x]):float(d[x]) for x in order}
    return selected,dist


def branch_fit(neigh, dist, states, road):
    ds = sorted(dist[j] for j in neigh if states[j+1]["phase"] == road)
    if len(ds) < MIN_BRANCH:
        return None
    m = min(FIT_NEIGHBORS, len(ds))
    # Higher is better: negative mean distance of the closest branch examples.
    return -float(np.mean(ds[:m]))


def counts_for_road(draws, neigh, states, road):
    js = [j for j in neigh if states[j+1]["phase"] == road]
    counts = np.zeros(43,dtype=float)
    for j in js:
        for n in draws[j]:
            counts[int(n)-1] += 1
    return js,counts


def ranked_numbers(counts, salt):
    rng = np.random.default_rng(SEED + 1000003*(salt+1))
    jitter = rng.random(43)*1e-9
    order = np.argsort(-(counts+jitter),kind="stable")
    return [int(x+1) for x in order]


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def random_pool_metrics(pool_size):
    den = math.comb(43,6)
    pmf = {
        k: math.comb(pool_size,k)*math.comb(43-pool_size,6-k)/den
        for k in range(7) if 0 <= 6-k <= 43-pool_size and k <= pool_size
    }
    return {
        "mean_hits":6*pool_size/43,
        "p_ge_3":sum(v for k,v in pmf.items() if k>=3),
        "p_ge_4":sum(v for k,v in pmf.items() if k>=4),
        "p_ge_5":sum(v for k,v in pmf.items() if k>=5),
    }


def build_records(draws, mean, sd):
    draws=np.asarray(draws,dtype=int)
    states0,next_state=build_states(draws)
    states=states0+[next_state]
    X=signature_matrix(states,mean,sd)

    records=[]
    for i in range(BURN,len(draws)):
        neigh,dist=corridor_with_distances(states,X,i)
        if not neigh:
            continue

        fits={}
        for road in PHASES:
            f=branch_fit(neigh,dist,states,road)
            if f is not None:
                fits[road]=f

        actual_road=states[i+1]["phase"]
        if actual_road not in fits or len(fits)<2:
            continue

        wrong=[v for road,v in fits.items() if road!=actual_road]
        fit_margin=fits[actual_road]-float(np.mean(wrong))

        js,counts=counts_for_road(draws,neigh,states,actual_road)
        ranking=ranked_numbers(counts,i)
        hits={}
        for p in POOL_SIZES:
            hits[str(p)]=overlap(ranking[:p],draws[i])

        records.append({
            "index":i,
            "road":actual_road,
            "fit":fits[actual_road],
            "fit_margin":fit_margin,
            "available_roads":len(fits),
            "branch_examples":len(js),
            "hits":hits,
        })

    return records,states,X


def curve_from_records(records):
    if not records:
        return []
    margins=np.asarray([r["fit_margin"] for r in records],dtype=float)
    out=[]
    for keep in RETENTION:
        n=max(1,int(math.ceil(len(records)*keep)))
        order=np.argsort(-margins,kind="stable")[:n]
        selected=[records[int(x)] for x in order]
        threshold=float(margins[order[-1]])

        pools={}
        for p in POOL_SIZES:
            h=np.asarray([r["hits"][str(p)] for r in selected],dtype=float)
            pools[str(p)]={
                "n":int(len(h)),
                "mean_hits":float(h.mean()),
                "p_ge_3":float(np.mean(h>=3)),
                "p_ge_4":float(np.mean(h>=4)),
                "p_ge_5":float(np.mean(h>=5)),
                "random_exact":random_pool_metrics(p),
            }

        road_counts=Counter(r["road"] for r in selected)
        out.append({
            "retain_fraction":keep,
            "n":len(selected),
            "fit_margin_threshold":threshold,
            "fit_margin_mean":float(np.mean([r["fit_margin"] for r in selected])),
            "branch_examples_mean":float(np.mean([r["branch_examples"] for r in selected])),
            "road_mix":dict(road_counts),
            "pools":pools,
        })
    return out


def current_scenarios(draws, rounds, mean, sd, historical_records):
    draws=np.asarray(draws,dtype=int)
    states0,next_state=build_states(draws)
    states=states0+[next_state]
    X=signature_matrix(states,mean,sd)
    i=len(draws)

    neigh,dist=corridor_with_distances(states,X,i)
    fits={}
    for road in PHASES:
        f=branch_fit(neigh,dist,states,road)
        if f is not None:
            fits[road]=f

    coarse=[
        j for j in range(3,i)
        if states[j]["phase"]==next_state["phase"]
        and states[j]["prev_phase"]==next_state["prev_phase"]
        and states[j]["motion"]==next_state["motion"]
    ]
    traffic=Counter(states[j+1]["phase"] for j in coarse)
    total=sum(traffic.values())

    hist_margins=np.asarray([r["fit_margin"] for r in historical_records],dtype=float)
    roads={}
    for road,count in sorted(traffic.items(),key=lambda kv:(-kv[1],kv[0])):
        if road not in fits:
            continue
        wrong=[v for rr,v in fits.items() if rr!=road]
        margin=fits[road]-float(np.mean(wrong))
        pct=float(np.mean(hist_margins <= margin)) if len(hist_margins) else None
        js,counts=counts_for_road(draws,neigh,states,road)
        ranking=ranked_numbers(counts,i+100000+len(roads))
        roads[road]={
            "traffic_count":int(count),
            "traffic_share":float(count/total) if total else 0.0,
            "corridor_examples":len(js),
            "fit_score":float(fits[road]),
            "fit_margin_vs_other_roads":float(margin),
            "historical_correct_route_fit_margin_percentile":pct,
            "candidate18":sorted(ranking[:18]),
            "candidate15":sorted(ranking[:15]),
            "candidate12":sorted(ranking[:12]),
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
        "corridor_width":CORRIDOR_WIDTH,
        "same_coarse_intersection_examples":int(total),
        "roads":roads,
    }


def null_curve_summary(actual_curve, null_curves):
    out=[]
    for idx,row in enumerate(actual_curve):
        item={
            "retain_fraction":row["retain_fraction"],
            "n":row["n"],
            "fit_margin_threshold":row["fit_margin_threshold"],
            "pools":{},
        }
        for p in POOL_SIZES:
            pm={}
            for metric in ["mean_hits","p_ge_3","p_ge_4"]:
                a=float(row["pools"][str(p)][metric])
                vals=np.asarray([nc[idx]["pools"][str(p)][metric] for nc in null_curves],dtype=float)
                pm[metric]={
                    "actual":a,
                    "B_mean":float(vals.mean()),
                    "B_q025":float(np.quantile(vals,.025)),
                    "B_q50":float(np.quantile(vals,.5)),
                    "B_q975":float(np.quantile(vals,.975)),
                    "actual_percentile":float(np.mean(vals<=a)),
                }
            item["pools"][str(p)]=pm
        out.append(item)
    return out


def main():
    rows=fetch_history()
    rounds=[r for r,_ in rows]
    draws=np.asarray([x for _,x in rows],dtype=int)

    rng=np.random.default_rng(SEED)
    mean,sd=calibrate_signature_scale(rng)

    records,_,_=build_records(draws,mean,sd)
    actual_curve=curve_from_records(records)
    current=current_scenarios(draws,rounds,mean,sd,records)

    null_curves=[]
    for _ in range(NULL_WORLDS):
        world=random_world(rng,len(draws))
        recs,_,_=build_records(world,mean,sd)
        null_curves.append(curve_from_records(recs))

    out={
        "probe":"LOTO6 Correct Route × Consistency Threshold Curve v0",
        "source":URL,
        "rounds":[int(rounds[0]),int(rounds[-1])],
        "draws":len(rows),
        "burn":BURN,
        "question":"Assuming the chosen road is the road that actually occurs, does stronger pre-draw alignment between the current movement corridor and that road permit progressively deeper number-space narrowing?",
        "fixed_design":{
            "corridor_width":CORRIDOR_WIDTH,
            "fit_neighbors":FIT_NEIGHBORS,
            "pool_sizes":POOL_SIZES,
            "retention_curve":RETENTION,
            "minimum_branch_examples":MIN_BRANCH,
        },
        "consistency_score":{
            "available_before_draw_if_a_road_is_chosen":True,
            "definition":"For each candidate road, take the closest prior examples of that road inside the current movement corridor. Fit is negative mean movement distance. Consistency margin = chosen-road fit minus mean fit of other available roads.",
            "oracle_part":"Historical diagnostics assume the human happened to choose the actually realized next road. The target draw's six numbers are not used to compute the consistency threshold.",
        },
        "historical_records":len(records),
        "threshold_curve":actual_curve,
        "B_world_comparison":null_curve_summary(actual_curve,null_curves),
        "current_2141_scenarios":current,
        "null":{
            "worlds":NULL_WORLDS,
            "seed":SEED,
            "generator":"independent exact-uniform 6-of-43 worlds",
        },
        "boundary":[
            "No Floot app changes were made.",
            "Correct-road identity is an oracle condition in historical evaluation; this does not measure the probability that a user will choose the right road.",
            "The consistency score itself uses only pre-draw movement geometry and prior labeled routes, not the target draw numbers.",
            "Thresholds are reported as a full retention curve rather than selecting one best cutoff after seeing results.",
            "B-world uses the same retention fractions and selection process, which controls for gains caused merely by keeping fewer cases.",
            "Historical A remains exploratory and is not a pristine future holdout."
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
