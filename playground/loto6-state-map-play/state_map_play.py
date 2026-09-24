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
OUT = Path("results/loto6_state_map_play_v0.json")

SEED = 20260924
NULL_WORLDS = 100
BURN = 100
POOL = 18
CENTER = 37 / 6
CENTER_BAND = 0.25
RECENT_W = 8


def fetch_history():
    r = requests.get(URL, timeout=30, headers={"User-Agent": "lotocore-X state-map-play probe"})
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
    return len(set(pool).intersection(int(x) for x in draw))


def exact_random18():
    den = math.comb(43, 6)
    pmf = {}
    for k in range(7):
        if k <= POOL and 6 - k <= 43 - POOL:
            pmf[k] = math.comb(POOL, k) * math.comb(43 - POOL, 6 - k) / den
        else:
            pmf[k] = 0.0
    return {
        "mean_hits": 6 * POOL / 43,
        "p_ge_3": sum(v for k, v in pmf.items() if k >= 3),
        "p_ge_4": sum(v for k, v in pmf.items() if k >= 4),
        "p_ge_5": sum(v for k, v in pmf.items() if k >= 5),
        "p_6": pmf[6],
    }


def ranks_recent(recent_draws):
    counts = np.zeros(43, dtype=float)
    for d in recent_draws[-RECENT_W:]:
        for n in d:
            counts[int(n) - 1] += 1
    # Primary: recent hit count. Secondary: lower number only for deterministic tie-breaking.
    order = sorted(range(43), key=lambda i: (-counts[i], i))
    return [i + 1 for i in order], counts


def ranks_old(ages):
    order = sorted(range(43), key=lambda i: (-ages[i], i))
    return [i + 1 for i in order]


def pool_break(recent_rank, old_rank):
    out = []
    for n in recent_rank[:9]:
        if n not in out:
            out.append(n)
    for n in old_rank[:18]:
        if n not in out:
            out.append(n)
        if len(out) == POOL:
            break
    if len(out) < POOL:
        for n in recent_rank:
            if n not in out:
                out.append(n)
            if len(out) == POOL:
                break
    return out


def pool_blend(counts, ages):
    # Coarse "old flow + new shape" blend. Normalize both to [0,1] in the current state.
    cmax = max(float(counts.max()), 1.0)
    recent = counts / cmax
    amin = float(ages.min())
    amax = float(ages.max())
    if amax > amin:
        old = (ages - amin) / (amax - amin)
    else:
        old = np.zeros_like(ages)
    score = 0.5 * recent + 0.5 * old
    order = sorted(range(43), key=lambda i: (-score[i], i))
    return [i + 1 for i in order[:POOL]]


def deterministic_random_pool(step):
    rng = np.random.default_rng(SEED + 1000003 * (step + 1))
    return sorted((rng.choice(np.arange(1, 44), size=POOL, replace=False)).tolist())


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


def choose_pool(label, ages, recent_draws, step):
    recent_rank, counts = ranks_recent(recent_draws)
    old_rank = ranks_old(ages)
    if label == "Flow":
        return recent_rank[:POOL]
    if label == "Break":
        return pool_break(recent_rank, old_rank)
    if label == "Uncertainty":
        return deterministic_random_pool(step)
    if label == "Re-formation":
        return pool_blend(counts, ages)
    raise ValueError(label)


def choose_primitive(name, ages, recent_draws, step):
    recent_rank, counts = ranks_recent(recent_draws)
    old_rank = ranks_old(ages)
    if name == "recent":
        return recent_rank[:POOL]
    if name == "old":
        return old_rank[:POOL]
    if name == "blend":
        return pool_blend(counts, ages)
    if name == "random":
        return deterministic_random_pool(step)
    raise ValueError(name)


def summarize_hits(hits):
    a = np.asarray(hits, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean_hits": float(a.mean()),
        "p_ge_3": float(np.mean(a >= 3)),
        "p_ge_4": float(np.mean(a >= 4)),
        "p_ge_5": float(np.mean(a >= 5)),
        "max_hits": int(a.max()),
    }


def evaluate(draws, rounds=None):
    ages = np.zeros(43, dtype=float)
    recent_draws = []
    prev_A = None
    prev_label = "Uncertainty"

    labels = []
    map_hits = []
    by_state = defaultdict(list)
    primitive_hits = {k: [] for k in ["recent", "old", "blend", "random"]}
    transitions = Counter()
    prev_eval_label = None

    state_rows = []

    for i, draw in enumerate(draws):
        draw = [int(x) for x in draw]
        A = float(ages.mean())
        label = classify_state(A, prev_A, prev_label)

        if i >= BURN:
            pool = choose_pool(label, ages, recent_draws, i)
            h = overlap(pool, draw)
            labels.append(label)
            map_hits.append(h)
            by_state[label].append(h)

            for name in primitive_hits:
                primitive_hits[name].append(overlap(choose_primitive(name, ages, recent_draws, i), draw))

            if prev_eval_label is not None:
                transitions[(prev_eval_label, label)] += 1
            prev_eval_label = label

            state_rows.append({
                "round": int(rounds[i]) if rounds is not None else i + 1,
                "A": A,
                "x": A - CENTER,
                "label": label,
                "hits": h,
            })

        next_ages = ages + 1
        for n in draw:
            next_ages[n - 1] = 0
        ages = next_ages

        recent_draws.append(draw)
        if len(recent_draws) > RECENT_W:
            recent_draws = recent_draws[-RECENT_W:]

        prev_A = A
        prev_label = label

    n = len(labels)
    occ = {s: labels.count(s) / n for s in ["Flow", "Break", "Uncertainty", "Re-formation"]}
    state_perf = {s: summarize_hits(by_state[s]) for s in ["Flow", "Break", "Uncertainty", "Re-formation"]}

    trans = {
        f"{a}->{b}": c
        for (a, b), c in sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    return {
        "map_policy": summarize_hits(map_hits),
        "state_occupancy": occ,
        "state_performance": state_perf,
        "transitions": trans,
        "primitives": {k: summarize_hits(v) for k, v in primitive_hits.items()},
        "rows": state_rows,
    }


def compare_null(actual, arr, high_is_interesting=True):
    a = np.asarray(arr, dtype=float)
    if high_is_interesting:
        p = (1 + np.sum(a >= actual)) / (len(a) + 1)
    else:
        p = (1 + np.sum(a <= actual)) / (len(a) + 1)
    return {
        "actual": float(actual),
        "null_mean": float(a.mean()),
        "null_q025": float(np.quantile(a, 0.025)),
        "null_q975": float(np.quantile(a, 0.975)),
        "actual_percentile": float(np.mean(a <= actual)),
        "one_sided_p": float(p),
    }


def main():
    rows = fetch_history()
    rounds = [r for r, _ in rows]
    actual_draws = np.asarray([x for _, x in rows], dtype=int)
    actual = evaluate(actual_draws, rounds)

    rng = np.random.default_rng(SEED)
    null = {
        "map_mean_hits": [],
        "map_p_ge_3": [],
        "occupancy": {s: [] for s in ["Flow", "Break", "Uncertainty", "Re-formation"]},
    }

    for _ in range(NULL_WORLDS):
        world = random_world(rng, len(rows))
        m = evaluate(world)
        null["map_mean_hits"].append(m["map_policy"]["mean_hits"])
        null["map_p_ge_3"].append(m["map_policy"]["p_ge_3"])
        for s in null["occupancy"]:
            null["occupancy"][s].append(m["state_occupancy"][s])

    comparison = {
        "map_mean_hits": compare_null(actual["map_policy"]["mean_hits"], null["map_mean_hits"]),
        "map_p_ge_3": compare_null(actual["map_policy"]["p_ge_3"], null["map_p_ge_3"]),
        "state_occupancy": {
            s: compare_null(actual["state_occupancy"][s], null["occupancy"][s])
            for s in null["occupancy"]
        },
    }

    compact_actual = dict(actual)
    compact_actual.pop("rows")

    out = {
        "probe": "LOTO6 Random Sequence State Map Play v0",
        "source": URL,
        "rounds": [rounds[0], rounds[-1]],
        "draws": len(rows),
        "burn": BURN,
        "pool_size": POOL,
        "map": {
            "center_A": CENTER,
            "center_band": CENTER_BAND,
            "recent_window": RECENT_W,
            "classification": [
                "Uncertainty: |A-center| <= 0.25",
                "Re-formation: outside center band, moving centerward, previous label Break/Uncertainty",
                "Flow: outside center band and moving centerward",
                "Break: outside center band and not moving centerward",
            ],
            "play_policy": {
                "Flow": "18 numbers with highest occurrence count in previous 8 draws (continuity)",
                "Break": "9 continuity-side plus old-number side until 18 (spread)",
                "Uncertainty": "deterministic random 18 independent of actual future data",
                "Re-formation": "18 by equal blend of recent-8 frequency and current age rank",
            },
        },
        "random18_exact": exact_random18(),
        "actual": compact_actual,
        "null": {
            "worlds": NULL_WORLDS,
            "seed": SEED,
            "generator": "independent exact-uniform 6-of-43 worlds; same state classifier and same phase-switching policy",
        },
        "comparison": comparison,
        "boundary": [
            "This is a coarse play probe, not a fitted predictor.",
            "The four phase labels and extraction personalities were fixed before reading this run.",
            "Historical rounds are exploratory and partly research-contaminated; any uplift is not confirmation.",
            "The goal is to see whether the state-map idea produces a coherent large-scale shape and whether a crude phase switch moves equal-size 18-number hit metrics at all.",
        ],
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
