from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_route_map_v0.json")

SEED = 20260924
BURN = 100
CENTER = 37 / 6
CENTER_BAND = 0.25
NULL_WORLDS = 120
CAL_DRAWS = 200000

PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X route-map probe"})
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
    vars_ = x.var(axis=1)
    return {
        "mu_mean": float(mus.mean()),
        "mu_sd": float(mus.std(ddof=1)),
        "var_mean": float(vars_.mean()),
        "var_sd": float(vars_.std(ddof=1)),
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


def motion_symbol(dmu_z, dvar_z):
    # Deliberately coarse large-map symbols.
    # Shift: centroid movement, Spread: variance movement.
    t = 0.35
    if dmu_z > t:
        h = "R"
    elif dmu_z < -t:
        h = "L"
    else:
        h = "M"

    if dvar_z > t:
        v = "E"   # expand
    elif dvar_z < -t:
        v = "C"   # contract
    else:
        v = "S"   # stable spread
    return h + v


def family_symbol(dmu_z, dvar_z):
    # Fewer, more human-readable route families.
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


def evaluate(draws, cal, rounds=None):
    draws = np.asarray(draws, dtype=int)
    shp = [draw_shape(d) for d in draws]

    ages = np.zeros(43, dtype=float)
    prev_A = None
    prev_phase = "Uncertainty"

    phase_seq = []
    z_seq = []
    round_seq = []

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        phase = classify_phase(A, prev_A, prev_phase)
        if i == 0:
            mu, vv = cal["mu_mean"], cal["var_mean"]
        else:
            mu, vv = shp[i-1]
        z = np.array([
            (mu-cal["mu_mean"]) / max(cal["mu_sd"],1e-9),
            (vv-cal["var_mean"]) / max(cal["var_sd"],1e-9),
        ])

        if i >= BURN:
            phase_seq.append(phase)
            z_seq.append(z)
            round_seq.append(int(rounds[i]) if rounds is not None else i+1)

        nxt = ages + 1
        for n in draw:
            nxt[int(n)-1] = 0
        ages = nxt
        prev_A = A
        prev_phase = phase

    edge_family = Counter()
    edge_motion = Counter()
    triple_family = Counter()
    route_windows = []

    for i in range(1, len(phase_seq)):
        a, b = phase_seq[i-1], phase_seq[i]
        dz = z_seq[i] - z_seq[i-1]
        fam = family_symbol(float(dz[0]), float(dz[1]))
        mot = motion_symbol(float(dz[0]), float(dz[1]))
        edge_family[(a,b,fam)] += 1
        edge_motion[(a,b,mot)] += 1

    for i in range(2, len(phase_seq)):
        a,b,c = phase_seq[i-2],phase_seq[i-1],phase_seq[i]
        dz1 = z_seq[i-1]-z_seq[i-2]
        dz2 = z_seq[i]-z_seq[i-1]
        f1 = family_symbol(float(dz1[0]),float(dz1[1]))
        f2 = family_symbol(float(dz2[0]),float(dz2[1]))
        key = (a,b,c,f1,f2)
        triple_family[key] += 1

    total_edges = max(1, len(phase_seq)-1)
    total_triples = max(1, len(phase_seq)-2)

    return {
        "n": len(phase_seq),
        "edge_family_freq": {"|".join(k): v/total_edges for k,v in edge_family.items()},
        "edge_motion_freq": {"|".join(k): v/total_edges for k,v in edge_motion.items()},
        "triple_family_freq": {"|".join(k): v/total_triples for k,v in triple_family.items()},
        "edge_family_count": {"|".join(k): v for k,v in edge_family.items()},
        "triple_family_count": {"|".join(k): v for k,v in triple_family.items()},
    }


def compare_sparse(actual, worlds, field, count_field, min_expected=3.0, topn=20):
    keys = set(actual[field].keys())
    for w in worlds:
        keys.update(w[field].keys())
    ntrans = max(1, actual["n"]-1 if "edge" in field else actual["n"]-2)
    rows = []
    for k in keys:
        vals = np.asarray([w[field].get(k,0.0) for w in worlds], dtype=float)
        a = actual[field].get(k,0.0)
        mean = float(vals.mean())
        expected = mean*ntrans
        if expected < min_expected and a == 0:
            continue
        sd = float(vals.std(ddof=1))
        z = (a-mean)/sd if sd>1e-12 else 0.0
        rows.append({
            "route": k,
            "actual_count": int(actual[count_field].get(k,0)),
            "actual_freq": a,
            "null_mean_freq": mean,
            "expected_count": expected,
            "delta": a-mean,
            "z": float(z),
            "percentile": float(np.mean(vals <= a)),
        })
    rows.sort(key=lambda x: abs(x["z"]), reverse=True)
    return rows[:topn]


def aggregate_family(edge_family_count):
    out = Counter()
    for k,c in edge_family_count.items():
        _,_,fam = k.split("|")
        out[fam] += c
    total = sum(out.values()) or 1
    return {k:v/total for k,v in out.items()}


def main():
    rows = fetch_history()
    rounds = [r for r,_ in rows]
    actual_draws = np.asarray([x for _,x in rows], dtype=int)

    rng = np.random.default_rng(SEED)
    cal = calibrate(rng)
    actual = evaluate(actual_draws, cal, rounds)

    worlds = [evaluate(random_world(rng, len(rows)), cal) for _ in range(NULL_WORLDS)]

    # Compare large route families, not individual raw edges only.
    actual_agg = aggregate_family(actual["edge_family_count"])
    fam_names = sorted({k for w in worlds for k in aggregate_family(w["edge_family_count"])} | set(actual_agg))
    agg_cmp = []
    for fam in fam_names:
        vals = np.asarray([aggregate_family(w["edge_family_count"]).get(fam,0.0) for w in worlds], dtype=float)
        a = actual_agg.get(fam,0.0)
        sd = float(vals.std(ddof=1))
        agg_cmp.append({
            "family": fam,
            "actual": a,
            "null_mean": float(vals.mean()),
            "delta": a-float(vals.mean()),
            "z": float((a-float(vals.mean()))/sd) if sd>1e-12 else 0.0,
            "percentile": float(np.mean(vals <= a)),
        })
    agg_cmp.sort(key=lambda x: abs(x["z"]), reverse=True)

    out = {
        "probe": "LOTO6 Large Route Map v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "definition": {
            "phase_layer": "Flow / Break / Uncertainty / Re-formation",
            "shape_motion_layer": "centroid and variance movement in B-standardized units",
            "route_families": {
                "Hold": "little centroid or spread movement",
                "Expand": "spread expansion dominates",
                "Contract": "spread contraction dominates",
                "Shift-L/R": "centroid shift dominates",
                "Shift+Expand": "centroid shift and expansion both substantial",
                "Shift+Contract": "centroid shift and contraction both substantial",
            },
            "route_bundle": "phase transition + coarse shape-motion family",
            "triple_route": "three consecutive phase states + two coarse shape-motion families",
        },
        "B_calibration": cal,
        "actual": {
            "n": actual["n"],
            "aggregate_route_family": actual_agg,
        },
        "comparison": {
            "aggregate_route_families": agg_cmp,
            "top_phase_shape_routes": compare_sparse(
                actual, worlds, "edge_family_freq", "edge_family_count", min_expected=3.0, topn=24
            ),
            "top_three-step_routes": compare_sparse(
                actual, worlds, "triple_family_freq", "triple_family_count", min_expected=2.0, topn=24
            ),
        },
        "null": {
            "worlds": NULL_WORLDS,
            "seed": SEED,
            "generator": "independent exact-uniform 6-of-43 worlds with identical phase and shape-motion reader",
        },
        "boundary": [
            "Exploratory map-building only; multiple route families are inspected and no route is adopted as predictive.",
            "The purpose is to see route bundles and large path grammar before later consistency checks.",
            "Historical A includes previously explored periods.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
