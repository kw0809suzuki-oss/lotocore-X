from __future__ import annotations

import csv, io, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_route_normalization_v0.json")

SEED = 20260924
BURN = 100
CENTER = 37 / 6
CENTER_BAND = 0.25
NULL_WORLDS = 200
CAL_DRAWS = 200000

PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]
MOTIONS = ["Hold", "Expand", "Contract", "Shift-L", "Shift-R", "Shift+Expand", "Shift+Contract"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X route-normalization"})
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
        raise RuntimeError(len(rows))
    return rows


def random_world(rng, n):
    u = rng.random((n,43))
    idx = np.argpartition(u, 6, axis=1)[:,:6] + 1
    return np.sort(idx, axis=1)


def draw_shape(draw):
    a = np.asarray(draw, dtype=float)
    return float(a.mean()), float(a.var())


def calibrate(rng):
    x = random_world(rng, CAL_DRAWS)
    mus = x.mean(axis=1)
    vv = x.var(axis=1)
    return {
        "mu_mean": float(mus.mean()),
        "mu_sd": float(mus.std(ddof=1)),
        "var_mean": float(vv.mean()),
        "var_sd": float(vv.std(ddof=1)),
    }


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


def motion_family(dmu_z, dvar_z):
    am, av = abs(dmu_z), abs(dvar_z)
    t = 0.35
    if am <= t and av <= t:
        return "Hold"
    if av > am * 1.25:
        return "Expand" if dvar_z > 0 else "Contract"
    if am > av * 1.25:
        return "Shift-R" if dmu_z > 0 else "Shift-L"
    if dvar_z > 0:
        return "Shift+Expand"
    return "Shift+Contract"


def build_sequence(draws, cal):
    draws = np.asarray(draws, dtype=int)
    shapes = [draw_shape(d) for d in draws]

    ages = np.zeros(43, dtype=float)
    prev_A = None
    prev_phase = "Uncertainty"

    phases, z = [], []

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        p = classify_phase(A, prev_A, prev_phase)

        if i == 0:
            mu, vv = cal["mu_mean"], cal["var_mean"]
        else:
            mu, vv = shapes[i-1]

        zv = np.array([
            (mu-cal["mu_mean"]) / max(cal["mu_sd"], 1e-9),
            (vv-cal["var_mean"]) / max(cal["var_sd"], 1e-9),
        ], dtype=float)

        if i >= BURN:
            phases.append(p)
            z.append(zv)

        nxt = ages + 1
        for n in draw:
            nxt[int(n)-1] = 0
        ages = nxt
        prev_A = A
        prev_phase = p

    motions = []
    for a,b in zip(z, z[1:]):
        d = b-a
        motions.append(motion_family(float(d[0]), float(d[1])))

    return phases, motions


def evaluate(draws, cal):
    phases, motions = build_sequence(draws, cal)

    edge_den = Counter()
    edge_num = Counter()

    # motion_t accompanies Phase_t -> Phase_t+1
    for i,m in enumerate(motions):
        edge = (phases[i], phases[i+1])
        edge_den[edge] += 1
        edge_num[(edge[0], edge[1], m)] += 1

    triple_den = Counter()
    triple_num = Counter()

    # motion_t, motion_t+1 under Phase_t, Phase_t+1, Phase_t+2
    for i in range(len(motions)-1):
        tri = (phases[i], phases[i+1], phases[i+2])
        pair = (motions[i], motions[i+1])
        triple_den[tri] += 1
        triple_num[(tri[0],tri[1],tri[2],pair[0],pair[1])] += 1

    edge_rates = {}
    for a in PHASES:
        for b in PHASES:
            den = edge_den[(a,b)]
            for m in MOTIONS:
                edge_rates[(a,b,m)] = edge_num[(a,b,m)]/den if den else None

    triple_rates = {}
    for a in PHASES:
        for b in PHASES:
            for c in PHASES:
                den = triple_den[(a,b,c)]
                for m1 in MOTIONS:
                    for m2 in MOTIONS:
                        triple_rates[(a,b,c,m1,m2)] = triple_num[(a,b,c,m1,m2)]/den if den else None

    return {
        "edge_den": {"|".join(k): int(v) for k,v in edge_den.items()},
        "edge_num": {"|".join(k): int(v) for k,v in edge_num.items()},
        "edge_rates": {"|".join(k): v for k,v in edge_rates.items()},
        "triple_den": {"|".join(k): int(v) for k,v in triple_den.items()},
        "triple_num": {"|".join(k): int(v) for k,v in triple_num.items()},
        "triple_rates": {"|".join(k): v for k,v in triple_rates.items()},
    }


def compare_edge(actual, worlds):
    rows = []
    for key, aval in actual["edge_rates"].items():
        if aval is None:
            continue
        a,b,m = key.split("|")
        aden = actual["edge_den"].get(f"{a}|{b}", 0)
        vals = []
        dens = []
        for w in worlds:
            v = w["edge_rates"].get(key)
            d = w["edge_den"].get(f"{a}|{b}", 0)
            if v is not None:
                vals.append(v); dens.append(d)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        rows.append({
            "intersection": f"{a}->{b}",
            "motion": m,
            "A_opportunities": int(aden),
            "A_rate": float(aval),
            "B_worlds_with_opportunity": int(len(arr)),
            "B_mean_rate": mean,
            "B_q025": float(np.quantile(arr,.025)),
            "B_q975": float(np.quantile(arr,.975)),
            "B_min": float(arr.min()),
            "B_max": float(arr.max()),
            "delta": float(aval-mean),
            "z": float((aval-mean)/sd) if sd>1e-12 else 0.0,
            "percentile": float(np.mean(arr <= aval)),
            "B_mean_opportunities": float(np.mean(dens)),
        })
    # Main map: enough A traffic to represent an actual intersection.
    main = [r for r in rows if r["A_opportunities"] >= 20 and r["B_mean_opportunities"] >= 15]
    main.sort(key=lambda x: abs(x["z"]), reverse=True)
    rows.sort(key=lambda x: abs(x["z"]), reverse=True)
    return {"main": main[:32], "all_top": rows[:40]}


def compare_triple(actual, worlds):
    rows=[]
    for key, aval in actual["triple_rates"].items():
        if aval is None:
            continue
        a,b,c,m1,m2 = key.split("|")
        tri = f"{a}|{b}|{c}"
        aden = actual["triple_den"].get(tri,0)
        vals=[]; dens=[]
        for w in worlds:
            v=w["triple_rates"].get(key)
            d=w["triple_den"].get(tri,0)
            if v is not None:
                vals.append(v); dens.append(d)
        if not vals:
            continue
        arr=np.asarray(vals,dtype=float)
        mean=float(arr.mean()); sd=float(arr.std(ddof=1))
        rows.append({
            "intersection":f"{a}->{b}->{c}",
            "motion_pair":f"{m1}->{m2}",
            "A_opportunities":int(aden),
            "A_rate":float(aval),
            "B_worlds_with_opportunity":int(len(arr)),
            "B_mean_rate":mean,
            "B_q025":float(np.quantile(arr,.025)),
            "B_q975":float(np.quantile(arr,.975)),
            "delta":float(aval-mean),
            "z":float((aval-mean)/sd) if sd>1e-12 else 0.0,
            "percentile":float(np.mean(arr<=aval)),
            "B_mean_opportunities":float(np.mean(dens)),
        })
    main=[r for r in rows if r["A_opportunities"]>=20 and r["B_mean_opportunities"]>=15]
    main.sort(key=lambda x: abs(x["z"]), reverse=True)
    rows.sort(key=lambda x: abs(x["z"]), reverse=True)
    return {"main":main[:32],"all_top":rows[:40]}


def main():
    rows=fetch_history()
    actual_draws=np.asarray([x for _,x in rows],dtype=int)

    rng=np.random.default_rng(SEED)
    cal=calibrate(rng)
    actual=evaluate(actual_draws,cal)
    worlds=[evaluate(random_world(rng,len(rows)),cal) for _ in range(NULL_WORLDS)]

    out={
        "probe":"LOTO6 Route Grammar Normalization v0",
        "source":URL,
        "rounds":[rows[0][0],rows[-1][0]],
        "draws":len(rows),
        "question":"After removing traffic volume, does A still choose different Shape Motion at the same Phase intersection?",
        "definition":{
            "edge":"P(Motion | Phase_t -> Phase_t+1)",
            "triple":"P(Motion_t -> Motion_t+1 | Phase_t -> Phase_t+1 -> Phase_t+2)",
            "motion_families":MOTIONS,
            "comparison":"A rate versus full per-world B distribution; B average is not used alone."
        },
        "B_calibration":cal,
        "actual_opportunity_counts":{
            "edge":actual["edge_den"],
            "triple":actual["triple_den"]
        },
        "comparison":{
            "edge_conditional":compare_edge(actual,worlds),
            "triple_conditional":compare_triple(actual,worlds)
        },
        "null":{
            "worlds":NULL_WORLDS,
            "seed":SEED,
            "generator":"independent exact-uniform 6-of-43 worlds with identical phase and shape-motion reader"
        },
        "boundary":[
            "Normalization only: no new predictor or route rule is introduced.",
            "Main view requires at least 20 A opportunities and mean 15 B opportunities.",
            "Historical A is exploratory; conditional deviations remain candidates until later consistency checks."
        ]
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
