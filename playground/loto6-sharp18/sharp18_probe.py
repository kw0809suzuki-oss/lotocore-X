from __future__ import annotations

import csv, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_sharp18_v0.json")

SEED = 20260925
BURN = 150
POOL = 18
K_SHAPE = 15
MIN_CONTEXT = 8
NULL_WORLDS = 12

CENTER = 37 / 6
CENTER_BAND = 0.25
PHASES = ["Flow", "Break", "Uncertainty", "Re-formation"]


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X sharp18"})
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
    rows.sort(key=lambda x: x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few rows: {len(rows)}")
    return rows


def random_world(rng, n):
    u = rng.random((n, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
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


def exact_random18():
    den = math.comb(43, 6)
    pmf = {k: math.comb(POOL, k) * math.comb(43-POOL, 6-k) / den for k in range(7)}
    return {
        "mean_hits": 6 * POOL / 43,
        "p_ge_3": sum(v for k, v in pmf.items() if k >= 3),
        "p_ge_4": sum(v for k, v in pmf.items() if k >= 4),
        "p_ge_5": sum(v for k, v in pmf.items() if k >= 5),
    }


def calibrate_shape(rng):
    u = rng.random((180000, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
    sx = np.sort(idx, axis=1)
    smu = sx.mean(axis=1)
    sv = sx.var(axis=1)
    return {
        "mu_mean": float(smu.mean()),
        "mu_sd": float(smu.std(ddof=1)),
        "var_mean": float(sv.mean()),
        "var_sd": float(sv.std(ddof=1)),
    }


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
            mu, vv = cal["mu_mean"], cal["var_mean"]
        else:
            mu, vv = shapes[i-1]
        z = np.array([
            (mu - cal["mu_mean"]) / max(cal["mu_sd"], 1e-12),
            (vv - cal["var_mean"]) / max(cal["var_sd"], 1e-12),
        ], dtype=float)
        if states:
            dz = z - states[-1]["z"]
            pmotion = motion_family(float(dz[0]), float(dz[1]))
            pphase = states[-1]["phase"]
        else:
            pmotion, pphase = "None", "None"
        states.append({
            "phase": phase,
            "prev_phase": pphase,
            "prev_motion": pmotion,
            "z": z,
            "A": A,
        })
        ages += 1
        ages[np.asarray(draw, dtype=int)-1] = 0
        prev_A = A
        prev_phase = phase

    A = float(ages.mean())
    phase = classify_phase(A, prev_A, prev_phase)
    mu, vv = shapes[-1]
    z = np.array([
        (mu - cal["mu_mean"]) / max(cal["mu_sd"], 1e-12),
        (vv - cal["var_mean"]) / max(cal["var_sd"], 1e-12),
    ], dtype=float)
    dz = z - states[-1]["z"]
    next_state = {
        "phase": phase,
        "prev_phase": states[-1]["phase"],
        "prev_motion": motion_family(float(dz[0]), float(dz[1])),
        "z": z,
        "A": A,
    }
    return states, next_state


def context_tiers(state):
    return [
        ("full", lambda x: x["phase"] == state["phase"] and x["prev_phase"] == state["prev_phase"] and x["prev_motion"] == state["prev_motion"]),
        ("phase+motion", lambda x: x["phase"] == state["phase"] and x["prev_motion"] == state["prev_motion"]),
        ("phase+previous", lambda x: x["phase"] == state["phase"] and x["prev_phase"] == state["prev_phase"]),
        ("phase", lambda x: x["phase"] == state["phase"]),
    ]


def choose_context_examples(states, current_index, state):
    # j < current_index. states[j+1] is already observable at current_index.
    for name, pred in context_tiers(state):
        js = [j for j in range(1, current_index) if pred(states[j])]
        if len(js) >= MIN_CONTEXT or name == "phase":
            return name, js
    return "none", []


def branch_examples(states, current_index, state, chosen_next_phase):
    for name, pred in context_tiers(state):
        js = [
            j for j in range(1, current_index)
            if pred(states[j]) and states[j+1]["phase"] == chosen_next_phase
        ]
        if len(js) >= MIN_CONTEXT or name == "phase":
            return name, js
    return "none", []


def nearest_by_shape(states, current_z, js, k=K_SHAPE):
    if not js:
        return []
    arr = np.asarray([states[j]["z"] for j in js], dtype=float)
    d = np.sqrt(np.sum((arr - current_z) ** 2, axis=1))
    kk = min(k, len(js))
    order = np.argsort(d, kind="stable")[:kk]
    return [js[int(x)] for x in order]


def score_from_examples(draws, js):
    counts = np.zeros(43, dtype=float)
    for j in js:
        for n in draws[j]:
            counts[int(n)-1] += 1
    return counts


def top18(score, salt):
    rng = np.random.default_rng(SEED + 1000003 * (salt + 1))
    jitter = rng.random(43) * 1e-9
    order = np.argsort(-(np.asarray(score, dtype=float) + jitter), kind="stable")
    return [int(x+1) for x in order[:POOL]]


def standard_shape_pool(states, draws, i, salt):
    js = list(range(1, i))
    near = nearest_by_shape(states, states[i]["z"], js)
    return top18(score_from_examples(draws, near), salt), near


def sharp18_pool(states, draws, i, chosen_branch, salt):
    tier, js = branch_examples(states, i, states[i], chosen_branch)
    near = nearest_by_shape(states, states[i]["z"], js)
    return top18(score_from_examples(draws, near), salt), tier, len(js), near


def route_distribution(states, i):
    tier, js = choose_context_examples(states, i, states[i])
    c = Counter(states[j+1]["phase"] for j in js)
    total = sum(c.values())
    dist = {p: (c[p] / total if total else 0.0) for p in PHASES}
    ranked = sorted(PHASES, key=lambda p: (-dist[p], p))
    return tier, len(js), dist, ranked


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def summarize(hits):
    if not hits:
        return {"n": 0}
    a = np.asarray(hits, dtype=float)
    return {
        "n": int(len(a)),
        "mean_hits": float(a.mean()),
        "p_ge_3": float(np.mean(a >= 3)),
        "p_ge_4": float(np.mean(a >= 4)),
        "p_ge_5": float(np.mean(a >= 5)),
        "hist": {str(k): int(np.sum(a == k)) for k in range(7)},
    }


def evaluate(draws, cal):
    draws = np.asarray(draws, dtype=int)
    states0, next_state = build_states(draws, cal)
    states = states0 + [next_state]

    shape_hits = []
    auto_hits = []
    oracle_hits = []
    auto_correct_hits = []
    auto_wrong_hits = []
    auto_route_correct = []
    per_branch_oracle = defaultdict(list)
    branch_pool_overlap = []

    tier_use_auto = Counter()
    tier_use_oracle = Counter()

    for i in range(BURN, len(draws)):
        actual_branch = states[i+1]["phase"]

        shape_pool, _ = standard_shape_pool(states, draws, i, i)
        route_tier, route_n, dist, ranked = route_distribution(states, i)
        chosen_branch = ranked[0]
        auto_route_correct.append(int(chosen_branch == actual_branch))

        auto_pool, auto_tier, _, _ = sharp18_pool(states, draws, i, chosen_branch, i + 10000)
        oracle_pool, oracle_tier, _, _ = sharp18_pool(states, draws, i, actual_branch, i + 20000)

        hs = overlap(shape_pool, draws[i])
        ha = overlap(auto_pool, draws[i])
        ho = overlap(oracle_pool, draws[i])
        shape_hits.append(hs)
        auto_hits.append(ha)
        oracle_hits.append(ho)
        per_branch_oracle[actual_branch].append(ho)
        tier_use_auto[auto_tier] += 1
        tier_use_oracle[oracle_tier] += 1

        if chosen_branch == actual_branch:
            auto_correct_hits.append(ha)
        else:
            auto_wrong_hits.append(ha)

    return {
        "shape18_unconditioned": summarize(shape_hits),
        "sharp18_auto_top_route": summarize(auto_hits),
        "sharp18_if_correct_route_oracle": summarize(oracle_hits),
        "auto_route_top1_accuracy": float(np.mean(auto_route_correct)),
        "auto_when_route_correct": summarize(auto_correct_hits),
        "auto_when_route_wrong": summarize(auto_wrong_hits),
        "oracle_by_actual_next_phase": {k: summarize(v) for k, v in sorted(per_branch_oracle.items())},
        "context_tiers_auto": dict(tier_use_auto),
        "context_tiers_oracle": dict(tier_use_oracle),
    }


def current_scenarios(draws, rounds, cal):
    draws = np.asarray(draws, dtype=int)
    states0, next_state = build_states(draws, cal)
    states = states0 + [next_state]
    i = len(draws)

    # route distribution uses all observed historical transitions j < i
    tier, nctx, dist, ranked = route_distribution(states, i)

    scenarios = {}
    for offset, branch in enumerate(ranked):
        if dist[branch] <= 0:
            continue
        pool, btier, nmatch, near = sharp18_pool(states, draws, i, branch, i + 30000 + offset)
        scenarios[branch] = {
            "route_share": dist[branch],
            "route_count": int(round(dist[branch] * nctx)),
            "context_tier": btier,
            "branch_context_matches": nmatch,
            "shape_neighbors_used": len(near),
            "candidate18_score_order": pool,
            "candidate18_sorted": sorted(pool),
            "nearest_rounds": [rounds[j] for j in near],
        }

    names = list(scenarios)
    pairwise = []
    for a in range(len(names)):
        for b in range(a+1, len(names)):
            A, B = names[a], names[b]
            sa, sb = set(scenarios[A]["candidate18_sorted"]), set(scenarios[B]["candidate18_sorted"])
            pairwise.append({
                "a": A, "b": B,
                "overlap": len(sa & sb),
                "union": len(sa | sb),
                "jaccard": len(sa & sb) / len(sa | sb),
                "only_a": sorted(sa - sb),
                "only_b": sorted(sb - sa),
            })

    return {
        "latest_observed_round": rounds[-1],
        "target_round": rounds[-1] + 1,
        "current_state": {
            "phase": next_state["phase"],
            "previous_phase": next_state["prev_phase"],
            "previous_motion": next_state["prev_motion"],
            "mean_age_A": float(next_state["A"]),
        },
        "route_context_tier": tier,
        "route_context_matches": nctx,
        "route_distribution": [
            {"phase": p, "count": int(round(dist[p] * nctx)), "share": dist[p]}
            for p in ranked if dist[p] > 0
        ],
        "scenario_candidate18": scenarios,
        "pairwise_candidate_overlap": pairwise,
    }


def null_compare(actual, nulls, metric_path):
    parts = metric_path.split(".")
    def get(x):
        for p in parts:
            x = x[p]
        return float(x)
    a = get(actual)
    vals = np.asarray([get(n) for n in nulls], dtype=float)
    return {
        "actual": a,
        "B_mean": float(vals.mean()),
        "B_q025": float(np.quantile(vals, .025)),
        "B_q50": float(np.quantile(vals, .5)),
        "B_q975": float(np.quantile(vals, .975)),
        "actual_percentile": float(np.mean(vals <= a)),
    }


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    draws = np.asarray([x for _, x in rows], dtype=int)

    rng = np.random.default_rng(SEED)
    cal = calibrate_shape(rng)

    actual = evaluate(draws, cal)
    current = current_scenarios(draws, rounds, cal)

    nulls = []
    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(draws))
        nulls.append(evaluate(world, cal))

    watched = {}
    for path in [
        "sharp18_auto_top_route.mean_hits",
        "sharp18_auto_top_route.p_ge_3",
        "sharp18_if_correct_route_oracle.mean_hits",
        "sharp18_if_correct_route_oracle.p_ge_3",
        "auto_route_top1_accuracy",
    ]:
        watched[path] = null_compare(actual, nulls, path)

    out = {
        "probe": "LOTO6 SHARP18 v0 — Path-Conditioned Shape18",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "pool_size": POOL,
        "question": "If the user chooses one of the next Phase roads, does conditioning Shape18 on that road create a meaningfully different candidate space, and what happens to historical extraction performance?",
        "definition": {
            "SHARP18": "Route hypothesis first, then Shape18 inside prior examples that took that same next-Phase road.",
            "route_context": "current Phase + previous Phase + previous Shape Motion, fallback only below 8 examples.",
            "shape": "Within the chosen-road examples, use the 15 nearest prior states in centroid x variance Shape space; count the transition-causing draw's numbers.",
            "oracle": "Historical diagnostic only: uses the actually realized next Phase to show the ceiling if the road choice were correct. It is not operationally available before a draw.",
            "auto": "Reference only: choose the most frequent next Phase from prior same-context transitions, then build SHARP18 for that road.",
        },
        "random18_exact": exact_random18(),
        "historical_strict_forward": actual,
        "B_world_comparison": watched,
        "current_2141_scenarios": current,
        "null": {
            "worlds": NULL_WORLDS,
            "seed": SEED,
            "generator": "independent exact-uniform 6-of-43 worlds",
        },
        "boundary": [
            "This is a research probe; no Floot app changes were made.",
            "Historical A is exploratory and not a pristine holdout.",
            "Oracle SHARP18 intentionally conditions on the realized next Phase and is only a diagnostic of 'if the chosen road is correct'.",
            "A user-selected road is a hypothesis, not an estimated winning probability.",
            "Do not retune route weights or rescue a failing branch after seeing these results."
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
