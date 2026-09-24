from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_residual_local_v0.json")
SEED = 20260924
BURN = 100
POOL = 18
CENTER = 37 / 6
CAL_WORLDS = 200
NULL_WORLDS = 400


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X residual-local probe"})
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


def pool_youngest(ages):
    return (np.argsort(ages, kind="stable")[:POOL] + 1).tolist()


def pool_oldest(ages):
    return (np.argsort(-ages, kind="stable")[:POOL] + 1).tolist()


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def trajectory(draws):
    ages = np.zeros(43, dtype=float)
    residual_prev = 0.0
    records = []
    residual_magnitudes = []

    for i, draw in enumerate(draws):
        draw = [int(x) for x in draw]

        if i >= BURN:
            if residual_prev >= 0:
                pool = pool_youngest(ages)
                anti = pool_oldest(ages)
            else:
                pool = pool_oldest(ages)
                anti = pool_youngest(ages)

            records.append({
                "abs_residual_prev": abs(float(residual_prev)),
                "residual_prev": float(residual_prev),
                "hits": overlap(pool, draw),
                "anti_hits": overlap(anti, draw),
            })

        A_before = float(ages.mean())
        expected_next_A = A_before + (37 - 6 * A_before) / 43
        next_ages = ages + 1
        for n in draw:
            next_ages[n - 1] = 0
        A_after = float(next_ages.mean())
        residual = A_after - expected_next_A

        if i >= BURN:
            residual_magnitudes.append(abs(float(residual)))

        ages = next_ages
        residual_prev = residual

    return records, residual_magnitudes


def bin_index(x, q50, q75, q90):
    if x <= q50:
        return 0
    if x <= q75:
        return 1
    if x <= q90:
        return 2
    return 3


def summarize(records, thresholds):
    q50, q75, q90 = thresholds
    bins = [[] for _ in range(4)]
    for r in records:
        bins[bin_index(r["abs_residual_prev"], q50, q75, q90)].append(r)

    out_bins = []
    labels = ["<=B-q50", "B-q50..q75", "B-q75..q90", ">B-q90"]
    for label, rows in zip(labels, bins):
        h = np.array([r["hits"] for r in rows], dtype=float)
        a = np.array([r["anti_hits"] for r in rows], dtype=float)
        out_bins.append({
            "bin": label,
            "n": int(len(rows)),
            "mean_hits": float(h.mean()) if len(h) else None,
            "p_ge_3": float(np.mean(h >= 3)) if len(h) else None,
            "anti_mean_hits": float(a.mean()) if len(a) else None,
            "directional_advantage": float((h - a).mean()) if len(h) else None,
        })

    x = np.array([r["abs_residual_prev"] for r in records], dtype=float)
    y = np.array([r["hits"] for r in records], dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")

    low = np.array([r["hits"] for r in bins[0]], dtype=float)
    high = np.array([r["hits"] for r in bins[3]], dtype=float)
    high_anti = np.array([r["anti_hits"] for r in bins[3]], dtype=float)

    return {
        "bins": out_bins,
        "corr_abs_residual_vs_next_hits": corr,
        "high_minus_low_mean_hits": float(high.mean() - low.mean()),
        "high_tail_directional_advantage": float((high - high_anti).mean()),
    }


def compare_null(actual, vals, high_is_interesting=True):
    arr = np.asarray(vals, dtype=float)
    if high_is_interesting:
        p = (1 + np.sum(arr >= actual)) / (len(arr) + 1)
    else:
        p = (1 + np.sum(arr <= actual)) / (len(arr) + 1)
    return {
        "actual": float(actual),
        "null_mean": float(arr.mean()),
        "null_q025": float(np.quantile(arr, 0.025)),
        "null_q975": float(np.quantile(arr, 0.975)),
        "actual_percentile": float(np.mean(arr <= actual)),
        "one_sided_p": float(p),
    }


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    actual_draws = np.asarray([x for _, x in rows], dtype=int)

    rng = np.random.default_rng(SEED)

    calibration_abs = []
    for _ in range(CAL_WORLDS):
        world = random_world(rng, len(rows))
        _, mags = trajectory(world)
        calibration_abs.extend(mags)

    thresholds = tuple(float(x) for x in np.quantile(np.asarray(calibration_abs), [0.50, 0.75, 0.90]))

    actual_records, _ = trajectory(actual_draws)
    actual = summarize(actual_records, thresholds)

    null_metrics = {
        "corr": [],
        "high_minus_low": [],
        "high_tail_advantage": [],
        "bin_mean_hits": [[], [], [], []],
        "bin_directional_advantage": [[], [], [], []],
    }

    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(rows))
        rec, _ = trajectory(world)
        s = summarize(rec, thresholds)
        null_metrics["corr"].append(s["corr_abs_residual_vs_next_hits"])
        null_metrics["high_minus_low"].append(s["high_minus_low_mean_hits"])
        null_metrics["high_tail_advantage"].append(s["high_tail_directional_advantage"])
        for j, b in enumerate(s["bins"]):
            null_metrics["bin_mean_hits"][j].append(b["mean_hits"])
            null_metrics["bin_directional_advantage"][j].append(b["directional_advantage"])

    comparisons = {
        "corr_abs_residual_vs_next_hits": compare_null(
            actual["corr_abs_residual_vs_next_hits"], null_metrics["corr"]
        ),
        "high_minus_low_mean_hits": compare_null(
            actual["high_minus_low_mean_hits"], null_metrics["high_minus_low"]
        ),
        "high_tail_directional_advantage": compare_null(
            actual["high_tail_directional_advantage"], null_metrics["high_tail_advantage"]
        ),
        "bins": [],
    }

    for j, b in enumerate(actual["bins"]):
        comparisons["bins"].append({
            "bin": b["bin"],
            "mean_hits": compare_null(b["mean_hits"], null_metrics["bin_mean_hits"][j]),
            "directional_advantage": compare_null(
                b["directional_advantage"], null_metrics["bin_directional_advantage"][j]
            ),
        })

    random18_mean = 6 * POOL / 43

    out = {
        "probe": "LOTO6 Residual Local Shape v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "pool_size": POOL,
        "B_center": CENTER,
        "question": "Does the small Residual18 uplift concentrate when |A-local-B residual| is large?",
        "fixed_rule": "Same Residual18 rule as prior probe: previous residual positive -> 18 youngest; negative -> 18 oldest. No rescue rule added.",
        "B_calibration": {
            "worlds": CAL_WORLDS,
            "thresholds_abs_residual": {
                "q50": thresholds[0],
                "q75": thresholds[1],
                "q90": thresholds[2],
            },
        },
        "random18_exact_mean_hits": random18_mean,
        "actual": actual,
        "null": {
            "worlds": NULL_WORLDS,
            "generator": "independent exact-uniform 6-of-43 worlds with identical recursive state reader and fixed thresholds from separate B calibration worlds",
            "seed": SEED,
        },
        "comparison": comparisons,
        "boundary": [
            "Observation-only historical probe; no new selector is adopted from bins after seeing the result.",
            "Rounds 1501-2139 are research-contaminated and not a pristine holdout.",
            "A local bump is not predictive evidence unless it exceeds the B-world cloud and later survives an untouched future interval.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
