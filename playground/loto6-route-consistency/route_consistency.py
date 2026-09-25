from __future__ import annotations

import csv, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_route_consistency_v0.json")

SEED = 20260925
BURN = 150
WIDTHS = [90, 180, 360]
MIN_BRANCH = 8
NULL_WORLDS = 8
ALPHA = 1.0

CENTER = 37 / 6
CENTER_BAND = 0.25

MU_MEAN = 22.0
MU_SD = 4.755114206198086
VAR_MEAN = 131.38888888573075
VAR_SD = 54.75268470147889

PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X route-consistency-v0"})
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


def corridor_neighbors(states, X, i, width):
    js = np.asarray([j for j in range(3,i) if states[j]["phase"] == states[i]["phase"]],dtype=int)
    if len(js) == 0:
        return []
    d = np.sqrt(np.sum((X[js]-X[i])**2,axis=1))
    k = min(width,len(js))
    order = np.argsort(d,kind="stable")[:k]
    return [int(js[x]) for x in order]


def branch_model(draws, js):
    counts = np.zeros(43,dtype=float)
    for j in js:
        for n in draws[j]:
            counts[int(n)-1] += 1
    total = float(counts.sum())
    q = (counts + ALPHA) / (total + 43*ALPHA)
    return counts, q


def token_log_lift(q, draw):
    a = np.asarray(draw,dtype=int)-1
    return float(np.mean(np.log(43.0*q[a])))


def mean_rank_percentile(q, draw):
    vals = []
    for n in draw:
        qn = q[int(n)-1]
        lower = float(np.sum(q < qn))
        equal = float(np.sum(q == qn))
        vals.append((lower + 0.5*equal) / 43.0)
    return float(np.mean(vals))


def top18(counts, salt):
    rng = np.random.default_rng(SEED + 1000003*(salt+1))
    jitter = rng.random(43)*1e-9
    order = np.argsort(-(counts+jitter),kind="stable")
    return [int(x+1) for x in order[:18]]


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def summarize(vals):
    a=np.asarray(vals,dtype=float)
    if len(a)==0:
        return {"n":0}
    return {
        "n":int(len(a)),
        "mean":float(a.mean()),
        "median":float(np.median(a)),
        "q10":float(np.quantile(a,.1)),
        "q90":float(np.quantile(a,.9)),
    }


def evaluate(draws, mean, sd):
    draws=np.asarray(draws,dtype=int)
    states0,next_state=build_states(draws)
    states=states0+[next_state]
    X=signature_matrix(states,mean,sd)

    out={"widths":{}}

    for wi,w in enumerate(WIDTHS):
        correct_scores=[]
        wrong_mean_scores=[]
        margins=[]
        correct_rank=[]
        wrong_rank=[]
        top18_hits=[]
        model_win=[]
        valid_models_per_target=[]
        by_road=defaultdict(lambda: {"margin":[],"win":[],"top18":[],"correct_log":[]})

        for i in range(BURN,len(draws)):
            neigh=corridor_neighbors(states,X,i,w)
            if not neigh:
                continue

            models={}
            for road in PHASES:
                js=[j for j in neigh if states[j+1]["phase"]==road]
                if len(js) < MIN_BRANCH:
                    continue
                counts,q=branch_model(draws,js)
                models[road]=(counts,q,len(js))

            actual_road=states[i+1]["phase"]
            if actual_road not in models or len(models)<2:
                continue

            draw=draws[i]
            scores={road:token_log_lift(q,draw) for road,(_,q,_) in models.items()}
            ranks={road:mean_rank_percentile(q,draw) for road,(_,q,_) in models.items()}

            cs=scores[actual_road]
            wrong=[v for road,v in scores.items() if road!=actual_road]
            cr=ranks[actual_road]
            wr=[v for road,v in ranks.items() if road!=actual_road]

            margin=cs-float(np.mean(wrong))
            win=float(cs > max(wrong))

            counts=models[actual_road][0]
            pool=top18(counts, i+wi*100000)
            hit=overlap(pool,draw)

            correct_scores.append(cs)
            wrong_mean_scores.append(float(np.mean(wrong)))
            margins.append(margin)
            correct_rank.append(cr)
            wrong_rank.append(float(np.mean(wr)))
            top18_hits.append(hit)
            model_win.append(win)
            valid_models_per_target.append(len(models))

            by_road[actual_road]["margin"].append(margin)
            by_road[actual_road]["win"].append(win)
            by_road[actual_road]["top18"].append(hit)
            by_road[actual_road]["correct_log"].append(cs)

        hits=np.asarray(top18_hits,dtype=float)
        out["widths"][str(w)]={
            "n":len(margins),
            "correct_route_log_lift":summarize(correct_scores),
            "wrong_routes_mean_log_lift":summarize(wrong_mean_scores),
            "correct_minus_wrong_log_margin":summarize(margins),
            "correct_route_rank_percentile":summarize(correct_rank),
            "wrong_routes_rank_percentile":summarize(wrong_rank),
            "correct_route_model_win_rate":float(np.mean(model_win)) if model_win else None,
            "correct_route_top18":{
                "n":int(len(hits)),
                "mean_hits":float(hits.mean()) if len(hits) else None,
                "p_ge_3":float(np.mean(hits>=3)) if len(hits) else None,
                "p_ge_4":float(np.mean(hits>=4)) if len(hits) else None,
                "p_ge_5":float(np.mean(hits>=5)) if len(hits) else None,
            },
            "valid_route_models_per_target":summarize(valid_models_per_target),
            "by_actual_road":{
                road:{
                    "n":len(v["margin"]),
                    "mean_margin":float(np.mean(v["margin"])) if v["margin"] else None,
                    "model_win_rate":float(np.mean(v["win"])) if v["win"] else None,
                    "top18_mean_hits":float(np.mean(v["top18"])) if v["top18"] else None,
                    "top18_p_ge_3":float(np.mean(np.asarray(v["top18"])>=3)) if v["top18"] else None,
                    "correct_log_lift_mean":float(np.mean(v["correct_log"])) if v["correct_log"] else None,
                }
                for road,v in sorted(by_road.items())
            }
        }

    return out


def current_templates(draws, rounds, mean, sd):
    draws=np.asarray(draws,dtype=int)
    states0,next_state=build_states(draws)
    states=states0+[next_state]
    X=signature_matrix(states,mean,sd)
    i=len(draws)

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
        roads={}
        for road,count in sorted(traffic.items(),key=lambda kv:(-kv[1],kv[0])):
            js=[j for j in neigh if states[j+1]["phase"]==road]
            if len(js)<MIN_BRANCH:
                continue
            counts,q=branch_model(draws,js)
            roads[road]={
                "traffic_count":int(count),
                "traffic_share":float(count/total) if total else 0.0,
                "corridor_examples":len(js),
                "candidate18":sorted(top18(counts,i+50000+wi*100000+len(roads))),
                "top_number_weights":[
                    {"number":int(n+1),"token_probability":float(q[n])}
                    for n in np.argsort(-q,kind="stable")[:10]
                ]
            }
        widths[str(w)]={"corridor_size":len(neigh),"roads":roads}

    return {
        "latest_observed_round":int(rounds[-1]),
        "target_round":int(rounds[-1]+1),
        "current_map":{
            "phase":next_state["phase"],
            "previous_phase":next_state["prev_phase"],
            "motion":next_state["motion"],
            "mean_age_A":float(next_state["A"]),
        },
        "same_coarse_intersection_examples":int(total),
        "traffic_distribution":[
            {"phase":p,"count":int(c),"share":float(c/total)}
            for p,c in sorted(traffic.items(),key=lambda kv:(-kv[1],kv[0]))
        ],
        "widths":widths,
    }


def nested_get(x, path):
    for p in path.split("."):
        x=x[p]
    return float(x)


def null_compare(actual, nulls, path):
    a=nested_get(actual,path)
    vals=np.asarray([nested_get(n,path) for n in nulls],dtype=float)
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
    current=current_templates(draws,rounds,mean,sd)

    nulls=[]
    for _ in range(NULL_WORLDS):
        world=random_world(rng,len(draws))
        nulls.append(evaluate(world,mean,sd))

    watched={}
    for w in WIDTHS:
        for suffix in [
            "correct_minus_wrong_log_margin.mean",
            "correct_route_model_win_rate",
            "correct_route_top18.mean_hits",
            "correct_route_top18.p_ge_3",
        ]:
            path=f"widths.{w}.{suffix}"
            watched[path]=null_compare(actual,nulls,path)

    out={
        "probe":"LOTO6 Route Consistency v0",
        "source":URL,
        "rounds":[int(rounds[0]),int(rounds[-1])],
        "draws":len(rows),
        "burn":BURN,
        "question":"When the predicted road is in fact the realized road, how much more consistent is the actual six-number draw with that road's corridor dynamics than with the counterfactual roads?",
        "consistency_definition":{
            "corridor":"same as Route Corridor v0: same Phase + nearest last-3-transition movement signatures; absolute shape position omitted",
            "road_model":"within a corridor, prior transitions are split by realized next Phase",
            "number_profile":"Laplace-smoothed frequency over the 43 numbers in each road subset",
            "primary_score":"mean token log-lift of the actual 6 numbers versus uniform 1/43 under each road profile",
            "contrast":"correct-road score minus mean score of the other available road profiles",
            "discrimination":"whether the correct-road profile scores the actual draw higher than every available wrong-road profile",
            "secondary":"mean rank percentile and top18 overlap",
            "alpha":ALPHA,
            "widths":WIDTHS,
            "minimum_branch_examples":MIN_BRANCH,
        },
        "historical_strict_forward":actual,
        "B_world_comparison":watched,
        "current_2141_templates":current,
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
            "This measures conditional consistency after the realized road is known; it is not an operational forecast accuracy estimate.",
            "The transition-causing draw partly determines the next Phase, so correct-road consistency can rise mechanically. B-world comparison is the control for that effect.",
            "All three previously declared corridor widths are reported; no width is retuned after seeing results.",
            "Historical A is exploratory and not a pristine future holdout."
        ]
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
