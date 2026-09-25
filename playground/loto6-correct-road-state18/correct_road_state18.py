from __future__ import annotations

import csv, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_correct_road_state18_v0.json")

SEED = 20260925
BURN = 150
POOL = 18
K = 15
NULL_WORLDS = 12

CENTER = 37 / 6
CENTER_BAND = 0.25

# Keep shape calibration aligned with the established Map Play work.
MU_MEAN = 22.0
MU_SD = 4.755114206198086
VAR_MEAN = 131.38888888573075
VAR_SD = 54.75268470147889


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X correct-road-state18"})
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


def build_raw_states(draws):
    draws = np.asarray(draws, dtype=int)
    shapes = [draw_shape(d) for d in draws]
    ages = np.zeros(43, dtype=float)
    prev_A = None
    prev_phase = "Uncertainty"
    states = []

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        dA = 0.0 if prev_A is None else A - prev_A
        phase = classify_phase(A, prev_A, prev_phase)

        if i == 0:
            mu, vv = MU_MEAN, VAR_MEAN
            zmu, zv = 0.0, 0.0
        else:
            mu, vv = shapes[i-1]
            zmu = (mu - MU_MEAN) / MU_SD
            zv = (vv - VAR_MEAN) / VAR_SD

        if states:
            dzmu = zmu - states[-1]["zmu"]
            dzv = zv - states[-1]["zv"]
            prev_phase_label = states[-1]["phase"]
        else:
            dzmu, dzv = 0.0, 0.0
            prev_phase_label = "None"

        states.append({
            "phase": phase,
            "prev_phase": prev_phase_label,
            "A": A,
            "dA": dA,
            "zmu": zmu,
            "zv": zv,
            "dzmu": dzmu,
            "dzv": dzv,
            "motion": motion_family(dzmu, dzv) if i else "None",
        })

        ages += 1
        ages[np.asarray(draw, dtype=int)-1] = 0
        prev_A = A
        prev_phase = phase

    # next not-yet-observed state after the latest draw
    A = float(ages.mean())
    dA = A - prev_A
    phase = classify_phase(A, prev_A, prev_phase)
    mu, vv = shapes[-1]
    zmu = (mu - MU_MEAN) / MU_SD
    zv = (vv - VAR_MEAN) / VAR_SD
    dzmu = zmu - states[-1]["zmu"]
    dzv = zv - states[-1]["zv"]
    next_state = {
        "phase": phase,
        "prev_phase": states[-1]["phase"],
        "A": A,
        "dA": dA,
        "zmu": zmu,
        "zv": zv,
        "dzmu": dzmu,
        "dzv": dzv,
        "motion": motion_family(dzmu, dzv),
    }
    return states, next_state


def raw_feature(s):
    # Exact continuous map-state coordinates, not coarse labels.
    return np.array([s["A"], s["dA"], s["zmu"], s["zv"], s["dzmu"], s["dzv"]], dtype=float)


def calibrate_feature_scale(rng):
    world = random_world(rng, 30000)
    states, _ = build_raw_states(world)
    X = np.asarray([raw_feature(s) for s in states[500:]], dtype=float)
    mean = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd <= 1e-12] = 1.0
    return mean, sd


def feature_matrix(states, mean, sd):
    return np.asarray([(raw_feature(s)-mean)/sd for s in states], dtype=float)


def top18_from_neighbors(draws, js, salt):
    counts = np.zeros(43, dtype=float)
    for j in js:
        for n in draws[j]:
            counts[int(n)-1] += 1
    rng = np.random.default_rng(SEED + 1000003*(salt+1))
    jitter = rng.random(43) * 1e-9
    order = np.argsort(-(counts + jitter), kind="stable")
    return [int(x+1) for x in order[:POOL]]


def nearest_prior(X, i, candidate_js, k=K):
    if len(candidate_js) == 0:
        return []
    js = np.asarray(candidate_js, dtype=int)
    d = np.sqrt(np.sum((X[js] - X[i])**2, axis=1))
    kk = min(k, len(js))
    order = np.argsort(d, kind="stable")[:kk]
    return [int(js[x]) for x in order]


def nearest_shape(states, i, candidate_js, k=K):
    if len(candidate_js) == 0:
        return []
    js = np.asarray(candidate_js, dtype=int)
    cur = np.array([states[i]["zmu"], states[i]["zv"]], dtype=float)
    arr = np.asarray([[states[j]["zmu"], states[j]["zv"]] for j in js], dtype=float)
    d = np.sqrt(np.sum((arr-cur)**2, axis=1))
    kk = min(k, len(js))
    order = np.argsort(d, kind="stable")[:kk]
    return [int(js[x]) for x in order]


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def summarize(hits):
    a = np.asarray(hits, dtype=float)
    return {
        "n": int(len(a)),
        "mean_hits": float(a.mean()),
        "p_ge_3": float(np.mean(a >= 3)),
        "p_ge_4": float(np.mean(a >= 4)),
        "p_ge_5": float(np.mean(a >= 5)),
        "hist": {str(k): int(np.sum(a == k)) for k in range(7)},
    }


def evaluate(draws, mean, sd):
    draws = np.asarray(draws, dtype=int)
    states0, next_state = build_raw_states(draws)
    states = states0 + [next_state]
    X = feature_matrix(states, mean, sd)

    shape_hits = []
    exact_state_hits = []
    oracle_hits = []
    oracle_by_phase = defaultdict(list)

    for i in range(BURN, len(draws)):
        prior = list(range(1, i))

        # existing shape-only reference
        shape_js = nearest_shape(states, i, prior, K)
        shape_pool = top18_from_neighbors(draws, shape_js, i)

        # same exact continuous state, no road conditioning
        state_js = nearest_prior(X, i, prior, K)
        state_pool = top18_from_neighbors(draws, state_js, i+10000)

        # diagnostic requested by user:
        # assume the next road is chosen correctly, then find prior states nearest
        # to the current continuous state among examples that took that same road.
        actual_road = states[i+1]["phase"]
        road_prior = [j for j in prior if states[j+1]["phase"] == actual_road]
        oracle_js = nearest_prior(X, i, road_prior, K)
        oracle_pool = top18_from_neighbors(draws, oracle_js, i+20000)

        hs = overlap(shape_pool, draws[i])
        he = overlap(state_pool, draws[i])
        ho = overlap(oracle_pool, draws[i])

        shape_hits.append(hs)
        exact_state_hits.append(he)
        oracle_hits.append(ho)
        oracle_by_phase[actual_road].append(ho)

    return {
        "shape18_reference": summarize(shape_hits),
        "exact_state18_unconditioned": summarize(exact_state_hits),
        "correct_road_exact_state18_oracle": summarize(oracle_hits),
        "oracle_by_actual_road": {k: summarize(v) for k, v in sorted(oracle_by_phase.items())},
    }


def current_scenarios(draws, rounds, mean, sd):
    draws = np.asarray(draws, dtype=int)
    states0, next_state = build_raw_states(draws)
    states = states0 + [next_state]
    X = feature_matrix(states, mean, sd)
    i = len(draws)

    # Same coarse road shares as the readability probe, but candidate numbers
    # are now created from exact continuous state matches within each road.
    full_ctx = [
        j for j in range(1, i)
        if states[j]["phase"] == next_state["phase"]
        and states[j]["prev_phase"] == next_state["prev_phase"]
        and states[j]["motion"] == next_state["motion"]
    ]
    c = Counter(states[j+1]["phase"] for j in full_ctx)
    total = sum(c.values())

    scenarios = {}
    for road, count in sorted(c.items(), key=lambda kv:(-kv[1], kv[0])):
        road_prior = [j for j in range(1, i) if states[j+1]["phase"] == road]
        near = nearest_prior(X, i, road_prior, K)
        pool = top18_from_neighbors(draws, near, i + 30000 + len(scenarios))
        scenarios[road] = {
            "traffic_count_in_same_coarse_intersection": int(count),
            "traffic_share_in_same_coarse_intersection": float(count/total) if total else 0.0,
            "all_prior_examples_that_took_this_road": len(road_prior),
            "nearest_exact_state_matches_used": len(near),
            "candidate18_score_order": pool,
            "candidate18_sorted": sorted(pool),
            "nearest_rounds": [int(rounds[j]) for j in near],
            "nearest_distances": [float(np.linalg.norm(X[j]-X[i])) for j in near],
        }

    names = list(scenarios)
    pairwise = []
    for a in range(len(names)):
        for b in range(a+1, len(names)):
            A, B = names[a], names[b]
            sa = set(scenarios[A]["candidate18_sorted"])
            sb = set(scenarios[B]["candidate18_sorted"])
            pairwise.append({
                "a": A, "b": B,
                "overlap": len(sa & sb),
                "only_a": sorted(sa-sb),
                "only_b": sorted(sb-sa),
            })

    return {
        "latest_observed_round": int(rounds[-1]),
        "target_round": int(rounds[-1] + 1),
        "current_state_exact": {
            "phase": next_state["phase"],
            "previous_phase": next_state["prev_phase"],
            "motion": next_state["motion"],
            "A": float(next_state["A"]),
            "dA": float(next_state["dA"]),
            "zmu": float(next_state["zmu"]),
            "zvar": float(next_state["zv"]),
            "dzmu": float(next_state["dzmu"]),
            "dzvar": float(next_state["dzv"]),
        },
        "same_coarse_intersection_examples": total,
        "road_scenarios": scenarios,
        "pairwise_candidate_overlap": pairwise,
    }


def null_compare(actual, nulls, path):
    parts = path.split(".")
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
    mean, sd = calibrate_feature_scale(rng)

    actual = evaluate(draws, mean, sd)
    current = current_scenarios(draws, rounds, mean, sd)

    nulls = []
    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(draws))
        nulls.append(evaluate(world, mean, sd))

    watched = {}
    for path in [
        "exact_state18_unconditioned.mean_hits",
        "exact_state18_unconditioned.p_ge_3",
        "correct_road_exact_state18_oracle.mean_hits",
        "correct_road_exact_state18_oracle.p_ge_3",
    ]:
        watched[path] = null_compare(actual, nulls, path)

    out = {
        "probe": "LOTO6 Correct-Road Exact-State18 v0",
        "source": URL,
        "rounds": [int(rounds[0]), int(rounds[-1])],
        "draws": len(rows),
        "burn": BURN,
        "pool_size": POOL,
        "question": "If the user chooses the correct next road, what does an 18-number pool look like when matching the current continuous map state rather than only coarse Phase labels?",
        "exact_state_vector": [
            "mean age A",
            "delta A",
            "shape centroid z",
            "shape variance z",
            "delta shape centroid z",
            "delta shape variance z"
        ],
        "method": {
            "historical_evaluation": "strict-forward; target i uses only j < i",
            "correct_road_oracle": "for diagnosis only, filter prior j to those whose realized next Phase equals the target's realized next Phase, then take 15 nearest exact-state matches",
            "numbers": "count the actual 6 numbers in those matched transition-causing draws, then take top18",
            "comparison": "same exact-state nearest neighbors without road condition, plus established shape-only reference",
        },
        "random18_exact": exact_random18(),
        "historical_strict_forward": actual,
        "B_world_comparison": watched,
        "current_2141_scenarios": current,
        "feature_calibration_from_independent_B_world": {
            "mean": [float(x) for x in mean],
            "sd": [float(x) for x in sd],
        },
        "null": {
            "worlds": NULL_WORLDS,
            "seed": SEED,
            "generator": "independent exact-uniform 6-of-43 worlds",
        },
        "boundary": [
            "No Floot app changes were made.",
            "Correct-road oracle uses future road identity and therefore is not operational before the draw; it measures the payoff if the road hypothesis is right.",
            "Because next Phase is itself produced by the transition-causing draw, conditioning on the correct road can mechanically reveal information about that draw. B-world comparison is therefore essential.",
            "Historical A is exploratory and not a pristine holdout.",
            "Do not rescue or retune after seeing this run."
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
