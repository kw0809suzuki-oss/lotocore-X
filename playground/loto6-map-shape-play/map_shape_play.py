from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

URL = "https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv"
OUT = Path("results/loto6_map_shape_play_v0.json")

SEED = 20260924
BURN = 100
POOL = 18
CENTER = 37 / 6
CENTER_BAND = 0.25
RECENT_W = 8
K = 15
NULL_WORLDS = 50


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X map-shape-play"})
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
            rnd = int(row[0])
            nums = [int(row[2 + i]) for i in range(6)]
        except Exception:
            continue
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 43 for n in nums):
            rows.append((rnd, sorted(nums)))
    rows.sort(key=lambda x: x[0])
    if len(rows) < 1000:
        raise RuntimeError(f"too few parsed draws: {len(rows)}")
    return rows


def random_world(rng, n):
    u = rng.random((n, 43))
    idx = np.argpartition(u, 6, axis=1)[:, :6] + 1
    return np.sort(idx, axis=1)


def draw_shape(draw):
    a = np.asarray(draw, dtype=float)
    return np.array([a.mean(), a.var()], dtype=float)


def overlap(pool, draw):
    return len(set(pool).intersection(int(x) for x in draw))


def normalize(v):
    a = np.asarray(v, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def deterministic_noise(step):
    rng = np.random.default_rng(SEED + 1000003 * (step + 1))
    return rng.random(43)


def top18(score, step):
    jitter = deterministic_noise(step) * 1e-9
    order = np.argsort(-(np.asarray(score, dtype=float) + jitter), kind="stable")
    return (order[:POOL] + 1).tolist()


def exact_random18():
    den = math.comb(43, 6)
    pmf = {
        k: (math.comb(POOL, k) * math.comb(43 - POOL, 6 - k) / den)
        if k <= POOL and 6-k <= 43-POOL else 0.0
        for k in range(7)
    }
    return {
        "mean_hits": 6 * POOL / 43,
        "p_ge_3": sum(v for k, v in pmf.items() if k >= 3),
        "p_ge_4": sum(v for k, v in pmf.items() if k >= 4),
        "p_ge_5": sum(v for k, v in pmf.items() if k >= 5),
        "p_6": pmf[6],
    }


def classify_state(A, prev_A, prev_label):
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


def shape_score(shapes, draws, i):
    # Target draw is i. Current observed shape is draw i-1.
    # Analogues j must satisfy j+1 <= i-1, so j <= i-2.
    if i < 3:
        return np.zeros(43, dtype=float)
    cur = shapes[i - 1]
    hist = shapes[: i - 1]  # states j=0..i-2
    scale = hist.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-9, scale, 1.0)
    d = np.sqrt(np.sum(((hist - cur) / scale) ** 2, axis=1))
    k = min(K, len(d))
    near = np.argpartition(d, k - 1)[:k]
    counts = np.zeros(43, dtype=float)
    for j in near:
        # transition j -> j+1 is known before target i because j+1 <= i-1
        for n in draws[j + 1]:
            counts[int(n) - 1] += 1
    return normalize(counts)


def recent_score(recent_draws):
    counts = np.zeros(43, dtype=float)
    for d in recent_draws[-RECENT_W:]:
        for n in d:
            counts[int(n) - 1] += 1
    return normalize(counts)


def old_score(ages):
    return normalize(ages)


def map_only_pool(label, recent, old, step):
    if label == "Flow":
        return top18(recent, step)
    if label == "Break":
        recent_rank = list(np.argsort(-recent, kind="stable") + 1)
        old_rank = list(np.argsort(-old, kind="stable") + 1)
        out = []
        for n in recent_rank[:9]:
            if n not in out:
                out.append(int(n))
        for n in old_rank:
            if int(n) not in out:
                out.append(int(n))
            if len(out) == POOL:
                break
        return out
    if label == "Uncertainty":
        return top18(deterministic_noise(step), step)
    if label == "Re-formation":
        return top18(0.5 * recent + 0.5 * old, step)
    raise ValueError(label)


def hybrid_pool(label, shape, recent, old, step):
    # Deliberately coarse. LOTO7-like shape analogue supplies half the score;
    # the LOTO6 movement phase decides the other half.
    if label == "Flow":
        score = 0.50 * shape + 0.50 * recent
    elif label == "Break":
        score = 0.50 * shape + 0.50 * old
    elif label == "Uncertainty":
        score = 0.50 * shape + 0.50 * deterministic_noise(step)
    elif label == "Re-formation":
        score = 0.50 * shape + 0.25 * recent + 0.25 * old
    else:
        raise ValueError(label)
    return top18(score, step)


def summarize(hits):
    a = np.asarray(hits, dtype=float)
    return {
        "n": int(len(a)),
        "mean_hits": float(a.mean()),
        "p_ge_3": float(np.mean(a >= 3)),
        "p_ge_4": float(np.mean(a >= 4)),
        "p_ge_5": float(np.mean(a >= 5)),
        "max_hits": int(a.max()),
        "hist": {str(k): int(np.sum(a == k)) for k in range(7)},
    }


def evaluate(draws, rounds=None):
    draws = np.asarray(draws, dtype=int)
    shapes = np.asarray([draw_shape(d) for d in draws], dtype=float)

    ages = np.zeros(43, dtype=float)
    recent_draws = []
    prev_A = None
    prev_label = "Uncertainty"

    hit = {k: [] for k in ["hybrid", "map_only", "shape_only", "recent", "old", "random"]}
    by_state = defaultdict(list)
    labels = []
    transitions = Counter()
    prev_eval = None

    for i, draw in enumerate(draws):
        A = float(ages.mean())
        label = classify_state(A, prev_A, prev_label)

        if i >= BURN:
            sh = shape_score(shapes, draws, i)
            re = recent_score(recent_draws)
            old = old_score(ages)
            pools = {
                "hybrid": hybrid_pool(label, sh, re, old, i),
                "map_only": map_only_pool(label, re, old, i),
                "shape_only": top18(sh, i),
                "recent": top18(re, i),
                "old": top18(old, i),
                "random": top18(deterministic_noise(i), i),
            }
            for name, p in pools.items():
                h = overlap(p, draw)
                hit[name].append(h)
                if name == "hybrid":
                    by_state[label].append(h)
            labels.append(label)
            if prev_eval is not None:
                transitions[(prev_eval, label)] += 1
            prev_eval = label

        next_ages = ages + 1
        for n in draw:
            next_ages[int(n) - 1] = 0
        ages = next_ages

        recent_draws.append([int(x) for x in draw])
        if len(recent_draws) > RECENT_W:
            recent_draws = recent_draws[-RECENT_W:]

        prev_A = A
        prev_label = label

    n = len(labels)
    occ = {s: labels.count(s) / n for s in ["Flow", "Break", "Uncertainty", "Re-formation"]}
    return {
        "metrics": {k: summarize(v) for k, v in hit.items()},
        "state_occupancy": occ,
        "hybrid_by_state": {s: summarize(by_state[s]) if by_state[s] else {"n": 0}
                            for s in ["Flow", "Break", "Uncertainty", "Re-formation"]},
        "transitions": {f"{a}->{b}": c for (a,b), c in sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))},
    }


def compare_null(actual, vals):
    a = np.asarray(vals, dtype=float)
    return {
        "actual": float(actual),
        "null_mean": float(a.mean()),
        "null_q025": float(np.quantile(a, 0.025)),
        "null_q975": float(np.quantile(a, 0.975)),
        "actual_percentile": float(np.mean(a <= actual)),
        "one_sided_p": float((1 + np.sum(a >= actual)) / (len(a) + 1)),
    }


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    actual_draws = np.asarray([x for _, x in rows], dtype=int)

    actual = evaluate(actual_draws, rounds)

    rng = np.random.default_rng(SEED)
    null = {k: {"mean_hits": [], "p_ge_3": []}
            for k in ["hybrid", "map_only", "shape_only"]}

    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(rows))
        m = evaluate(world)
        for k in null:
            null[k]["mean_hits"].append(m["metrics"][k]["mean_hits"])
            null[k]["p_ge_3"].append(m["metrics"][k]["p_ge_3"])

    comparison = {}
    for k in null:
        comparison[k] = {
            "mean_hits": compare_null(actual["metrics"][k]["mean_hits"], null[k]["mean_hits"]),
            "p_ge_3": compare_null(actual["metrics"][k]["p_ge_3"], null[k]["p_ge_3"]),
        }

    out = {
        "probe": "LOTO6 Map x LOTO7-Shape Play v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "pool_size": POOL,
        "random18_exact": exact_random18(),
        "map": {
            "phase": "same coarse Flow / Break / Uncertainty / Re-formation classifier as State Map Play v0",
            "shape": "LOTO7-style local analogue in (draw centroid, draw variance); K=15 prior states, candidate score from their known next draws",
            "hybrid": {
                "Flow": "0.50 shape + 0.50 recent-8 continuity",
                "Break": "0.50 shape + 0.50 age/old side",
                "Uncertainty": "0.50 shape + 0.50 deterministic random spread",
                "Re-formation": "0.50 shape + 0.25 recent + 0.25 age",
            },
        },
        "actual": actual,
        "null": {
            "worlds": NULL_WORLDS,
            "seed": SEED,
            "generator": "independent exact-uniform 6-of-43 worlds; same strict-forward shape analogue, phase classifier, and hybrid rule",
        },
        "comparison": comparison,
        "boundary": [
            "Play probe, not a fitted predictor.",
            "Weights and K were fixed before reading the run result; no post-result rescue.",
            "Historical rounds are exploratory and partly research-contaminated.",
            "The question is only whether combining the large movement map with LOTO7-style shape changes equal-size 18-number hit metrics at all.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
