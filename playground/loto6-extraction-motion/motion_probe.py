from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_extraction_motion_v0.json")
SEED = 20260924
NULL_WORLDS = 1000


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X extraction-motion probe"})
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
            nums = sorted(int(row[2 + i]) for i in range(6))
        except Exception:
            continue
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 43 for n in nums):
            rows.append((round_no, nums))
    rows.sort(key=lambda x: x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few parsed draws: {len(rows)}")
    return rows


def random_draws(rng, n):
    u = rng.random((n, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
    return np.sort(idx, axis=1).astype(float)


def scale_from_uniform(rng):
    sample = random_draws(rng, 200000)
    return sample.std(axis=0, ddof=1)


def motion_metrics(states, sd):
    s = states / sd
    v = np.diff(s, axis=0)
    a = v[:-1]
    b = v[1:]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    cos = np.divide((a * b).sum(axis=1), denom, out=np.zeros_like(denom), where=denom > 0)

    positive = cos > 0
    runs = []
    cur = 0
    for x in positive:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    pp_den = np.sum(positive[:-1])
    bp_den = np.sum(~positive[:-1])
    p_after_pos = np.sum(positive[:-1] & positive[1:]) / pp_den if pp_den else float("nan")
    p_after_break = np.sum((~positive[:-1]) & positive[1:]) / bp_den if bp_den else float("nan")

    pred = 2 * s[1:-1] - s[:-2]
    inertial_err = np.linalg.norm(s[2:] - pred, axis=1)

    return {
        "mean_cos": float(cos.mean()),
        "median_cos": float(np.median(cos)),
        "positive_rate": float(positive.mean()),
        "longest_positive_run": int(max(runs) if runs else 0),
        "mean_positive_run": float(np.mean(runs) if runs else 0.0),
        "p_continue_after_positive": float(p_after_pos),
        "p_continue_after_break": float(p_after_break),
        "mean_inertial_error": float(inertial_err.mean()),
        "q90_inertial_error": float(np.quantile(inertial_err, 0.90)),
    }


def summarize_null(values, actual, lower_is_better=False):
    arr = np.asarray(values, dtype=float)
    percentile = float(np.mean(arr <= actual))
    if lower_is_better:
        p_one = float((1 + np.sum(arr <= actual)) / (len(arr) + 1))
    else:
        p_one = float((1 + np.sum(arr >= actual)) / (len(arr) + 1))
    return {
        "actual": actual,
        "null_mean": float(arr.mean()),
        "null_median": float(np.median(arr)),
        "null_q025": float(np.quantile(arr, 0.025)),
        "null_q975": float(np.quantile(arr, 0.975)),
        "actual_percentile": percentile,
        "one_sided_p": p_one,
    }


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    actual_states = np.asarray([x for _, x in rows], dtype=float)

    rng = np.random.default_rng(SEED)
    sd = scale_from_uniform(rng)
    actual = motion_metrics(actual_states, sd)

    null = {k: [] for k in actual}
    for _ in range(NULL_WORLDS):
        world = random_draws(rng, len(rows))
        m = motion_metrics(world, sd)
        for k, v in m.items():
            null[k].append(v)

    compare = {
        k: summarize_null(
            null[k], v,
            lower_is_better=(k in {"mean_inertial_error", "q90_inertial_error"})
        )
        for k, v in actual.items()
    }

    out = {
        "probe": "LOTO6 Extraction Motion v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "state_definition": "sorted six-number vector [n1..n6], scaled only by coordinate SD learned from a uniform 6-of-43 simulation",
        "motion_definition": "V_t = S_t - S_(t-1)",
        "continuity_definition": "cosine(V_(t-1), V_t); positive means successive extraction-state motion points to the same half-space",
        "break_definition": "cosine <= 0, descriptive only",
        "null": {
            "worlds": NULL_WORLDS,
            "generator": "independent exact-uniform 6-of-43 draws",
            "seed": SEED
        },
        "actual_metrics": actual,
        "comparison": compare,
        "boundary": "Exploratory motion probe only. Any real-only deviation needs a separate probe before calling it flow."
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
