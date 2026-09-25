from __future__ import annotations

import csv, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_map_readability_v0.json")

SEED = 20260925
BURN = 150
NULL_WORLDS = 160
CAL_DRAWS = 300000
MIN_CONTEXT = 8
CENTER = 37 / 6
CENTER_BAND = 0.25

PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]
MOTIONS = ["Hold", "Expand", "Contract", "Shift-L", "Shift-R", "Shift+Expand", "Shift+Contract", "None"]

def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent":"lotocore-X map-readability"})
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

def calibrate_shape_cuts(rng):
    x = random_world(rng, CAL_DRAWS)
    mus = x.mean(axis=1)
    variances = x.var(axis=1)
    return {
        "mu_q1": float(np.quantile(mus, 1/3)),
        "mu_q2": float(np.quantile(mus, 2/3)),
        "var_q1": float(np.quantile(variances, 1/3)),
        "var_q2": float(np.quantile(variances, 2/3)),
        "mu_mean": float(mus.mean()),
        "mu_sd": float(mus.std(ddof=1)),
        "var_mean": float(variances.mean()),
        "var_sd": float(variances.std(ddof=1)),
    }

def classify_phase(A, prev_A, prev_label):
    x = A - CENTER
    if abs(x) <= CENTER_BAND:
        return "Uncertainty"
    dA = 0.0 if prev_A is None else A - prev_A
    centerward = (x * dA) < 0
    if centerward and prev_label in {"Break", "Uncertainty"}:
        return "Re-formation"
    if centerward:
        return "Flow"
    return "Break"

def motion_family(dmu, dvar):
    am, av = abs(dmu), abs(dvar)
    t = .35
    if am <= t and av <= t:
        return "Hold"
    if av > am * 1.25:
        return "Expand" if dvar > 0 else "Contract"
    if am > av * 1.25:
        return "Shift-R" if dmu > 0 else "Shift-L"
    return "Shift+Expand" if dvar > 0 else "Shift+Contract"

def shape_region(mu, variance, cal):
    m = 0 if mu <= cal["mu_q1"] else 1 if mu <= cal["mu_q2"] else 2
    v = 0 if variance <= cal["var_q1"] else 1 if variance <= cal["var_q2"] else 2
    m_name = ["Low", "Middle", "High"][m]
    v_name = ["Compact", "Normal", "Wide"][v]
    return f"{m_name}-{v_name}"

def build_states(draws, cal):
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
            mu, variance = cal["mu_mean"], cal["var_mean"]
        else:
            mu, variance = shapes[i-1]

        z = np.array([
            (mu - cal["mu_mean"]) / max(cal["mu_sd"], 1e-12),
            (variance - cal["var_mean"]) / max(cal["var_sd"], 1e-12),
        ], dtype=float)

        if states:
            dz = z - states[-1]["z"]
            prev_motion = motion_family(float(dz[0]), float(dz[1]))
            previous_phase = states[-1]["phase"]
        else:
            prev_motion = "None"
            previous_phase = "None"

        states.append({
            "phase": phase,
            "previous_phase": previous_phase,
            "previous_motion": prev_motion,
            "shape_region": shape_region(mu, variance, cal),
            "z": z,
            "A": A,
        })

        ages += 1
        ages[np.asarray(draw, dtype=int)-1] = 0
        prev_A = A
        prev_phase = phase

    # State for the next, not-yet-observed draw.
    A = float(ages.mean())
    phase = classify_phase(A, prev_A, prev_phase)
    mu, variance = shapes[-1]
    z = np.array([
        (mu - cal["mu_mean"]) / max(cal["mu_sd"], 1e-12),
        (variance - cal["var_mean"]) / max(cal["var_sd"], 1e-12),
    ], dtype=float)
    dz = z - states[-1]["z"]
    next_state = {
        "phase": phase,
        "previous_phase": states[-1]["phase"],
        "previous_motion": motion_family(float(dz[0]), float(dz[1])),
        "shape_region": shape_region(mu, variance, cal),
        "z": z,
        "A": A,
    }
    return states, next_state

def context_keys(s):
    return [
        ("full", (s["phase"], s["previous_phase"], s["previous_motion"])),
        ("phase+motion", (s["phase"], s["previous_motion"])),
        ("phase+previous", (s["phase"], s["previous_phase"])),
        ("phase", (s["phase"],)),
    ]

def ranked(counter):
    return [k for k,_ in sorted(counter.items(), key=lambda kv:(-kv[1], str(kv[0])))]

def choose_counter(store, state):
    keys = context_keys(state)
    for tier, key in keys[:-1]:
        c = store[tier].get(key, Counter())
        if sum(c.values()) >= MIN_CONTEXT:
            return tier, c
    tier, key = keys[-1]
    c = store[tier].get(key, Counter())
    if sum(c.values()) >= 1:
        return tier, c
    return "unconditional", Counter()

def add_transition(store, baseline, source_state, target_label):
    for tier, key in context_keys(source_state):
        store[tier][key][target_label] += 1
    baseline[target_label] += 1

def predict(counter, baseline, k):
    source = counter if sum(counter.values()) else baseline
    order = ranked(source)
    return order[:k]

def distribution(counter, baseline):
    source = counter if sum(counter.values()) else baseline
    total = sum(source.values())
    if total == 0:
        return []
    return [{"label": k, "count": int(v), "pct": float(v/total)}
            for k,v in sorted(source.items(), key=lambda kv:(-kv[1], str(kv[0])))]

def evaluate(draws, cal):
    states, next_state = build_states(draws, cal)

    phase_store = {tier: defaultdict(Counter) for tier in ["full","phase+motion","phase+previous","phase"]}
    shape_store = {tier: defaultdict(Counter) for tier in ["full","phase+motion","phase+previous","phase"]}
    phase_base = Counter()
    shape_base = Counter()

    scored = []
    tier_use_phase = Counter()
    tier_use_shape = Counter()

    phase_hits = {"map_top1":0, "map_top2":0, "phase_top1":0, "phase_top2":0, "base_top1":0, "base_top2":0}
    shape_hits = {"map_top1":0, "map_top3":0, "phase_top1":0, "phase_top3":0, "base_top1":0, "base_top3":0}

    for t in range(1, len(states)-1):
        # Transition (t-1 -> t) is already observable when standing at state t.
        add_transition(phase_store, phase_base, states[t-1], states[t]["phase"])
        add_transition(shape_store, shape_base, states[t-1], states[t]["shape_region"])

        if t < BURN:
            continue

        current = states[t]
        target_phase = states[t+1]["phase"]
        target_shape = states[t+1]["shape_region"]

        phase_tier, pc = choose_counter(phase_store, current)
        shape_tier, sc = choose_counter(shape_store, current)
        tier_use_phase[phase_tier] += 1
        tier_use_shape[shape_tier] += 1

        # Current-Phase-only comparator.
        phase_only_pc = phase_store["phase"].get((current["phase"],), Counter())
        phase_only_sc = shape_store["phase"].get((current["phase"],), Counter())

        mp1 = predict(pc, phase_base, 1)
        mp2 = predict(pc, phase_base, 2)
        pp1 = predict(phase_only_pc, phase_base, 1)
        pp2 = predict(phase_only_pc, phase_base, 2)
        bp1 = predict(Counter(), phase_base, 1)
        bp2 = predict(Counter(), phase_base, 2)

        ms1 = predict(sc, shape_base, 1)
        ms3 = predict(sc, shape_base, 3)
        ps1 = predict(phase_only_sc, shape_base, 1)
        ps3 = predict(phase_only_sc, shape_base, 3)
        bs1 = predict(Counter(), shape_base, 1)
        bs3 = predict(Counter(), shape_base, 3)

        phase_hits["map_top1"] += int(target_phase in mp1)
        phase_hits["map_top2"] += int(target_phase in mp2)
        phase_hits["phase_top1"] += int(target_phase in pp1)
        phase_hits["phase_top2"] += int(target_phase in pp2)
        phase_hits["base_top1"] += int(target_phase in bp1)
        phase_hits["base_top2"] += int(target_phase in bp2)

        shape_hits["map_top1"] += int(target_shape in ms1)
        shape_hits["map_top3"] += int(target_shape in ms3)
        shape_hits["phase_top1"] += int(target_shape in ps1)
        shape_hits["phase_top3"] += int(target_shape in ps3)
        shape_hits["base_top1"] += int(target_shape in bs1)
        shape_hits["base_top3"] += int(target_shape in bs3)

        scored.append({
            "index": t,
            "phase": current["phase"],
            "previous_phase": current["previous_phase"],
            "previous_motion": current["previous_motion"],
            "target_phase": target_phase,
            "target_shape": target_shape,
            "phase_tier": phase_tier,
            "shape_tier": shape_tier,
        })

    n = len(scored)
    def rate(x): return x/n if n else None

    metrics = {
        "n": n,
        "phase": {k:rate(v) for k,v in phase_hits.items()},
        "shape": {k:rate(v) for k,v in shape_hits.items()},
        "tier_use_phase": dict(tier_use_phase),
        "tier_use_shape": dict(tier_use_shape),
    }
    metrics["phase"]["map_top1_lift_vs_base"] = metrics["phase"]["map_top1"] - metrics["phase"]["base_top1"]
    metrics["phase"]["map_top2_lift_vs_base"] = metrics["phase"]["map_top2"] - metrics["phase"]["base_top2"]
    metrics["phase"]["map_top1_lift_vs_phase_only"] = metrics["phase"]["map_top1"] - metrics["phase"]["phase_top1"]
    metrics["phase"]["map_top2_lift_vs_phase_only"] = metrics["phase"]["map_top2"] - metrics["phase"]["phase_top2"]
    metrics["shape"]["map_top1_lift_vs_base"] = metrics["shape"]["map_top1"] - metrics["shape"]["base_top1"]
    metrics["shape"]["map_top3_lift_vs_base"] = metrics["shape"]["map_top3"] - metrics["shape"]["base_top3"]
    metrics["shape"]["map_top1_lift_vs_phase_only"] = metrics["shape"]["map_top1"] - metrics["shape"]["phase_top1"]
    metrics["shape"]["map_top3_lift_vs_phase_only"] = metrics["shape"]["map_top3"] - metrics["shape"]["phase_top3"]

    # Build forecast for next round using every observed transition through the latest state.
    add_transition(phase_store, phase_base, states[-2], states[-1]["phase"])
    add_transition(shape_store, shape_base, states[-2], states[-1]["shape_region"])

    ptier, pc = choose_counter(phase_store, next_state)
    stier, sc = choose_counter(shape_store, next_state)
    next_forecast = {
        "state": {
            "phase": next_state["phase"],
            "previous_phase": next_state["previous_phase"],
            "previous_motion": next_state["previous_motion"],
            "shape_region": next_state["shape_region"],
            "mean_age_A": float(next_state["A"]),
        },
        "phase_context_tier": ptier,
        "shape_context_tier": stier,
        "next_phase_distribution": distribution(pc, phase_base),
        "next_shape_distribution": distribution(sc, shape_base),
    }
    return metrics, next_forecast

def null_summary(actual, null_metrics, path):
    parts = path.split(".")
    def get(d):
        for p in parts:
            d = d[p]
        return float(d)
    a = get(actual)
    vals = np.asarray([get(x) for x in null_metrics], dtype=float)
    return {
        "actual": a,
        "B_mean": float(vals.mean()),
        "B_q025": float(np.quantile(vals,.025)),
        "B_q50": float(np.quantile(vals,.5)),
        "B_q975": float(np.quantile(vals,.975)),
        "actual_percentile": float(np.mean(vals <= a)),
        "upper_tail_p_with_plus1": float((1 + np.sum(vals >= a)) / (len(vals)+1)),
    }

def main():
    rows = fetch_history()
    draws = np.asarray([nums for _,nums in rows], dtype=int)
    rng = np.random.default_rng(SEED)
    cal = calibrate_shape_cuts(rng)

    actual, forecast = evaluate(draws, cal)

    null_metrics = []
    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(draws))
        m, _ = evaluate(world, cal)
        null_metrics.append(m)

    watched = {}
    for path in [
        "phase.map_top1_lift_vs_base",
        "phase.map_top2_lift_vs_base",
        "phase.map_top1_lift_vs_phase_only",
        "phase.map_top2_lift_vs_phase_only",
        "shape.map_top1_lift_vs_base",
        "shape.map_top3_lift_vs_base",
        "shape.map_top1_lift_vs_phase_only",
        "shape.map_top3_lift_vs_phase_only",
    ]:
        watched[path] = null_summary(actual, null_metrics, path)

    out = {
        "probe":"LOTO6 Map Readability v0",
        "source":URL,
        "rounds":[rows[0][0], rows[-1][0]],
        "draws":len(rows),
        "burn":BURN,
        "question":"How much does the current map state narrow the next Phase and next coarse Shape region under strict-forward reading?",
        "map_context":"current Phase + previous Phase + previous Shape Motion; fallback tiers are frozen and require >=8 prior matches",
        "shape_target":"9 coarse regions = centroid tercile x variance tercile using exact-uniform calibration simulation, not A quantiles",
        "actual_strict_forward":actual,
        "B_world_comparison":watched,
        "next_round_read": {
            "latest_observed_round":rows[-1][0],
            "target_round":rows[-1][0]+1,
            **forecast,
        },
        "shape_calibration":cal,
        "null":{
            "worlds":NULL_WORLDS,
            "seed":SEED,
            "generator":"independent exact-uniform 6-of-43 worlds",
        },
        "boundary":[
            "Historical A is exploratory and not a pristine holdout.",
            "Context definitions were developed during prior exploration; strict-forward prevents future leakage but does not erase research selection.",
            "This probe measures next-state readability, not lottery profitability and not exact next-number probability.",
            "B-world comparison asks whether the same reading machinery also looks predictive in pure random worlds."
        ]
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
