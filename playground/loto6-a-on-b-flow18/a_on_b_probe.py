from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_a_on_b_flow18_v0.json")
SEED = 20260924
NULL_WORLDS = 500
BURN = 100
POOL = 18
CENTER = 37 / 6
STRONG_FLOW_DISTANCE = 0.75


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X a-on-b-flow18 probe"})
    r.raise_for_status()
    try:
        text = r.content.decode("cp932")
    except UnicodeDecodeError:
        text = r.content.decode("utf-8")

    rows = []
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    for row in reader:
        try:
            round_no = int(row[0])
            nums = [int(row[2 + i]) for i in range(6)]
        except Exception:
            continue
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 43 for n in nums):
            rows.append((round_no, sorted(nums)))
    rows.sort(key=lambda x: x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few parsed draws: {len(rows)}")
    return rows


def random_world(rng, n):
    u = rng.random((n, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
    return np.sort(idx, axis=1)


def overlap(pool, draw):
    return len(set(pool).intersection(draw))


def pool_youngest(ages):
    return (np.argsort(ages, kind="stable")[:POOL] + 1).tolist()


def pool_oldest(ages):
    return (np.argsort(-ages, kind="stable")[:POOL] + 1).tolist()


def hypergeom_probs():
    den = math.comb(43, 6)
    probs = {}
    for k in range(7):
        if k <= POOL and 6 - k <= 43 - POOL:
            probs[k] = math.comb(POOL, k) * math.comb(43 - POOL, 6 - k) / den
        else:
            probs[k] = 0.0
    return {
        "mean_hits": 6 * POOL / 43,
        "p_ge_3": sum(v for k, v in probs.items() if k >= 3),
        "p_ge_4": sum(v for k, v in probs.items() if k >= 4),
        "p_ge_5": sum(v for k, v in probs.items() if k >= 5),
        "p_6": probs[6],
        "pmf": probs,
    }


def summarize_hits(hits):
    arr = np.asarray(hits, dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean_hits": float(arr.mean()),
        "p_ge_3": float(np.mean(arr >= 3)),
        "p_ge_4": float(np.mean(arr >= 4)),
        "p_ge_5": float(np.mean(arr >= 5)),
        "max_hits": int(arr.max()),
        "hist": {str(k): int(np.sum(arr == k)) for k in range(7)},
    }


def evaluate(draws, rounds=None):
    ages = np.zeros(43, dtype=float)
    residual_prev = 0.0

    flow_hits = []
    antiflow_hits = []
    residual_hits = []
    antiresidual_hits = []
    strong_flow_hits = []
    strong_antiflow_hits = []
    residuals = []
    eval_rounds = []
    rows = []

    for i, draw in enumerate(draws):
        draw = [int(x) for x in draw]

        if i >= BURN:
            A = float(ages.mean())
            if A >= CENTER:
                flow_pool = pool_oldest(ages)
                antiflow_pool = pool_youngest(ages)
            else:
                flow_pool = pool_youngest(ages)
                antiflow_pool = pool_oldest(ages)

            if residual_prev >= 0:
                residual_pool = pool_youngest(ages)
                antiresidual_pool = pool_oldest(ages)
            else:
                residual_pool = pool_oldest(ages)
                antiresidual_pool = pool_youngest(ages)

            fh = overlap(flow_pool, draw)
            afh = overlap(antiflow_pool, draw)
            rh = overlap(residual_pool, draw)
            arh = overlap(antiresidual_pool, draw)

            flow_hits.append(fh)
            antiflow_hits.append(afh)
            residual_hits.append(rh)
            antiresidual_hits.append(arh)
            if abs(A - CENTER) >= STRONG_FLOW_DISTANCE:
                strong_flow_hits.append(fh)
                strong_antiflow_hits.append(afh)

        A_before = float(ages.mean())
        expected_next_A = A_before + (37 - 6 * A_before) / 43

        next_ages = ages + 1
        for n in draw:
            next_ages[n - 1] = 0
        A_after = float(next_ages.mean())
        residual = A_after - expected_next_A

        if i >= BURN:
            residuals.append(float(residual))
            eval_rounds.append(int(rounds[i]) if rounds is not None else i + 1)
            rows.append({
                "round": int(rounds[i]) if rounds is not None else i + 1,
                "A_before": A_before,
                "distance_from_center": abs(A_before - CENTER),
                "expected_next_A_B": expected_next_A,
                "actual_next_A": A_after,
                "residual": residual,
                "flow_hits": flow_hits[-1],
                "antiflow_hits": antiflow_hits[-1],
                "residual18_hits": residual_hits[-1],
                "antiresidual18_hits": antiresidual_hits[-1],
            })

        ages = next_ages
        residual_prev = residual

    r = np.asarray(residuals, dtype=float)
    if len(r) > 1 and np.std(r[:-1]) > 0 and np.std(r[1:]) > 0:
        lag1_corr = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    else:
        lag1_corr = float("nan")
    sign_nonzero = (r[:-1] != 0) & (r[1:] != 0)
    same_sign = float(np.mean(np.sign(r[:-1][sign_nonzero]) == np.sign(r[1:][sign_nonzero]))) if np.any(sign_nonzero) else float("nan")

    n = len(flow_hits)
    cuts = [0, n // 3, 2 * n // 3, n]
    slices = []
    for j in range(3):
        lo, hi = cuts[j], cuts[j + 1]
        slices.append({
            "part": j + 1,
            "round_range": [eval_rounds[lo], eval_rounds[hi - 1]],
            "flow18": summarize_hits(flow_hits[lo:hi]),
            "residual18": summarize_hits(residual_hits[lo:hi]),
        })

    return {
        "flow18": summarize_hits(flow_hits),
        "antiflow18": summarize_hits(antiflow_hits),
        "residual18": summarize_hits(residual_hits),
        "antiresidual18": summarize_hits(antiresidual_hits),
        "strong_flow18": summarize_hits(strong_flow_hits),
        "strong_antiflow18": summarize_hits(strong_antiflow_hits),
        "residual_motion": {
            "n": int(len(r)),
            "mean": float(r.mean()),
            "sd": float(r.std(ddof=1)),
            "lag1_corr": lag1_corr,
            "same_sign_rate": same_sign,
        },
        "slices": slices,
        "rows": rows,
    }


def compare_null(actual, null_values, high_is_interesting=True):
    arr = np.asarray(null_values, dtype=float)
    if high_is_interesting:
        p = (1 + np.sum(arr >= actual)) / (len(arr) + 1)
    else:
        p = (1 + np.sum(arr <= actual)) / (len(arr) + 1)
    return {
        "actual": float(actual),
        "null_mean": float(arr.mean()),
        "null_median": float(np.median(arr)),
        "null_q025": float(np.quantile(arr, 0.025)),
        "null_q975": float(np.quantile(arr, 0.975)),
        "actual_percentile": float(np.mean(arr <= actual)),
        "one_sided_p": float(p),
    }


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    actual_draws = np.asarray([x for _, x in rows], dtype=int)

    actual = evaluate(actual_draws, rounds)

    rng = np.random.default_rng(SEED)
    null = {
        "flow18_mean_hits": [],
        "residual18_mean_hits": [],
        "strong_flow18_mean_hits": [],
        "residual_lag1_corr": [],
        "residual_same_sign_rate": [],
    }

    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(rows))
        m = evaluate(world)
        null["flow18_mean_hits"].append(m["flow18"]["mean_hits"])
        null["residual18_mean_hits"].append(m["residual18"]["mean_hits"])
        null["strong_flow18_mean_hits"].append(m["strong_flow18"]["mean_hits"])
        null["residual_lag1_corr"].append(m["residual_motion"]["lag1_corr"])
        null["residual_same_sign_rate"].append(m["residual_motion"]["same_sign_rate"])

    comparison = {
        "flow18_mean_hits": compare_null(actual["flow18"]["mean_hits"], null["flow18_mean_hits"]),
        "residual18_mean_hits": compare_null(actual["residual18"]["mean_hits"], null["residual18_mean_hits"]),
        "strong_flow18_mean_hits": compare_null(actual["strong_flow18"]["mean_hits"], null["strong_flow18_mean_hits"]),
        "residual_lag1_corr": compare_null(actual["residual_motion"]["lag1_corr"], null["residual_lag1_corr"]),
        "residual_same_sign_rate": compare_null(actual["residual_motion"]["same_sign_rate"], null["residual_same_sign_rate"]),
    }

    compact_actual = dict(actual)
    compact_actual.pop("rows")

    out = {
        "probe": "LOTO6 A-on-B Flow18 v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "pool_size": POOL,
        "B_equilibrium_center_A": CENTER,
        "strong_flow_distance": STRONG_FLOW_DISTANCE,
        "definitions": {
            "state": "ages of 43 numbers after the previous draw; age=draws since last appearance",
            "B_expected_motion": "E[A_next-A | A] = (37 - 6A)/43",
            "flow18": "if A is above B equilibrium, choose 18 oldest; if below, choose 18 youngest",
            "residual": "actual A_next minus B-expected A_next",
            "residual18": "use previous residual sign: positive -> 18 youngest, negative -> 18 oldest; this tests persistence of A-minus-local-B motion",
            "strong_flow": f"|A - 37/6| >= {STRONG_FLOW_DISTANCE}; fixed before reading the result",
        },
        "random18_exact": hypergeom_probs(),
        "actual": compact_actual,
        "null": {
            "worlds": NULL_WORLDS,
            "generator": "independent exact-uniform 6-of-43 worlds; same state reader and selection rules recomputed recursively inside each world",
            "seed": SEED,
        },
        "comparison": comparison,
        "boundary": [
            "Exploratory historical probe. Rounds 1501-2139 are already research-contaminated and are not a pristine holdout.",
            "No claim of predictive signal unless the effect is directionally coherent, beats the B-world cloud, and survives a future untouched interval.",
            "Flow18 tests whether the B recovery geometry itself maps to numbers. Residual18 tests whether A-minus-B deviations persist enough to map to numbers.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
