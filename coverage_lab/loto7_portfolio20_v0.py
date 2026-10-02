#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import lotocore

DRAW_SIZE = 7
POOL_K = 18
A_TICKETS = 20
B_CORE = 10
B_DIVERSIFY = 6
B_FALSIFY = 4
MODEL_WINDOW = 100
LOW_CUTOFF = 18
BANDS = ((1, 9), (10, 18), (19, 27), (28, 37))


def main_numbers(row) -> set[int]:
    return {int(row[f"n{i}"]) for i in range(1, 8)}


def rank_degrees(pool_order: list[int], n_tickets: int) -> Counter:
    total_slots = n_tickets * DRAW_SIZE
    if len(pool_order) < DRAW_SIZE:
        raise ValueError("candidate pool must contain at least 7 numbers")
    if len(pool_order) > total_slots:
        raise ValueError("pool too large for one-touch coverage")

    deg = Counter({n: 1 for n in pool_order})
    remaining = total_slots - len(pool_order)
    rank = {n: i for i, n in enumerate(pool_order)}
    weight = {n: len(pool_order) - i for i, n in enumerate(pool_order)}

    for _ in range(remaining):
        candidates = [n for n in pool_order if deg[n] < n_tickets]
        pick = max(
            candidates,
            key=lambda n: (weight[n] / (deg[n] + 1.0), -rank[n]),
        )
        deg[pick] += 1

    if sum(deg.values()) != total_slots:
        raise RuntimeError("degree allocation failed")
    return deg


def choose_ticket(
    remaining: Counter,
    tickets_left: int,
    rank: dict[int, int],
    pair_count: defaultdict,
    rng: random.Random,
) -> tuple[int, ...]:
    chosen = [n for n, c in remaining.items() if c == tickets_left]
    if len(chosen) > DRAW_SIZE:
        raise RuntimeError("infeasible degree schedule")

    while len(chosen) < DRAW_SIZE:
        candidates = [n for n, c in remaining.items() if c > 0 and n not in chosen]
        if not candidates:
            raise RuntimeError("ran out of candidates")
        rng.shuffle(candidates)
        pick = min(
            candidates,
            key=lambda n: (
                sum(pair_count[tuple(sorted((n, x)))] for x in chosen),
                -remaining[n],
                rank[n],
            ),
        )
        chosen.append(pick)

    for n in chosen:
        remaining[n] -= 1
    return tuple(sorted(chosen))


def structured_bundle(pool_order: list[int], n_tickets: int, seed: int) -> list[tuple[int, ...]]:
    degrees = rank_degrees(pool_order, n_tickets)
    remaining = Counter(degrees)
    pair_count = defaultdict(int)
    rank = {n: i for i, n in enumerate(pool_order)}
    rng = random.Random(seed)
    out = []

    for i in range(n_tickets):
        ticket = choose_ticket(
            remaining,
            n_tickets - i,
            rank,
            pair_count,
            rng,
        )
        for a, b in itertools.combinations(ticket, 2):
            pair_count[(a, b)] += 1
        out.append(ticket)

    if any(remaining.values()):
        raise RuntimeError(f"nonzero degrees after bundle: {remaining}")
    return out


def raw_structure(ticket: tuple[int, ...]) -> tuple[float, ...]:
    s = set(ticket)
    odd = sum(n % 2 for n in ticket)
    low = sum(n <= LOW_CUTOFF for n in ticket)
    total = sum(ticket)
    adjacent = sum(1 for n in ticket if n + 1 in s)
    bands = [sum(lo <= n <= hi for n in ticket) for lo, hi in BANDS]
    return (float(odd), float(low), float(total), float(adjacent), *map(float, bands))


def fixed_structure(ticket: tuple[int, ...]) -> tuple[float, ...]:
    r = raw_structure(ticket)
    return (
        r[0] / 7.0,
        r[1] / 7.0,
        r[2] / (7.0 * 37.0),
        r[3] / 6.0,
        r[4] / 7.0,
        r[5] / 7.0,
        r[6] / 7.0,
        r[7] / 7.0,
    )


def euclid(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def standardized_vectors(combos: list[tuple[int, ...]]) -> dict[tuple[int, ...], tuple[float, ...]]:
    raw = [raw_structure(c) for c in combos]
    cols = list(zip(*raw))
    means = [sum(c) / len(c) for c in cols]
    stds = []
    for col, mean in zip(cols, means):
        var = sum((x - mean) ** 2 for x in col) / len(col)
        stds.append(math.sqrt(var) or 1.0)
    return {
        combo: tuple((x - m) / s for x, m, s in zip(vec, means, stds))
        for combo, vec in zip(combos, raw)
    }


def max_overlap(ticket: tuple[int, ...], selected: list[tuple[int, ...]]) -> int:
    if not selected:
        return 0
    st = set(ticket)
    return max(len(st & set(x)) for x in selected)


def avg_candidate_score(ticket: tuple[int, ...], scores: dict[int, float]) -> float:
    return sum(scores[n] for n in ticket) / DRAW_SIZE


def portfolio_bundle(
    pool_order: list[int],
    scores: dict[int, float],
    seed: int,
) -> tuple[list[tuple[int, ...]], dict]:
    core = structured_bundle(pool_order, B_CORE, seed)
    combos = list(itertools.combinations(sorted(pool_order), DRAW_SIZE))
    z = standardized_vectors(combos)
    combo_score = {c: avg_candidate_score(c, scores) for c in combos}
    median_score = sorted(combo_score.values())[len(combo_score) // 2]

    selected = list(core)
    remaining = [c for c in combos if c not in set(selected)]

    diversify_candidates = [c for c in remaining if combo_score[c] >= median_score]
    diversify = []
    for _ in range(B_DIVERSIFY):
        best = max(
            diversify_candidates,
            key=lambda c: (
                min(euclid(z[c], z[s]) for s in selected),
                -max_overlap(c, selected),
                combo_score[c],
                tuple(-n for n in c),
            ),
        )
        diversify.append(best)
        selected.append(best)
        diversify_candidates.remove(best)

    core_centroid = tuple(
        sum(z[t][i] for t in core) / len(core)
        for i in range(len(z[core[0]]))
    )
    remaining_set = set(selected)
    falsify_candidates = [
        c for c in combos
        if c not in remaining_set and combo_score[c] <= median_score
    ]
    falsify = []
    for _ in range(B_FALSIFY):
        best = max(
            falsify_candidates,
            key=lambda c: (
                euclid(z[c], core_centroid),
                min(euclid(z[c], z[s]) for s in selected),
                -max_overlap(c, selected),
                -combo_score[c],
                tuple(-n for n in c),
            ),
        )
        falsify.append(best)
        selected.append(best)
        falsify_candidates.remove(best)

    if len(selected) != 20 or len(set(selected)) != 20:
        raise RuntimeError("portfolio did not produce 20 unique tickets")

    trace = {
        "core": [list(t) for t in core],
        "diversify": [list(t) for t in diversify],
        "falsify": [list(t) for t in falsify],
        "median_combo_score": median_score,
    }
    return selected, trace


def pairwise_overlap_mean(tickets: list[tuple[int, ...]]) -> float:
    vals = []
    for a, b in itertools.combinations(tickets, 2):
        vals.append(len(set(a) & set(b)))
    return sum(vals) / len(vals)


def structure_distance_stats(tickets: list[tuple[int, ...]]) -> tuple[float, float]:
    vecs = [fixed_structure(t) for t in tickets]
    vals = [euclid(a, b) for a, b in itertools.combinations(vecs, 2)]
    return sum(vals) / len(vals), min(vals)


def winning_subset_coverage(
    tickets: list[tuple[int, ...]], actual: set[int], subset_size: int
) -> int:
    if len(actual) < subset_size:
        return 0
    ticket_sets = [set(t) for t in tickets]
    return sum(
        any(set(sub) <= t for t in ticket_sets)
        for sub in itertools.combinations(sorted(actual), subset_size)
    )


def bundle_metrics(tickets: list[tuple[int, ...]], actual: set[int]) -> dict:
    hits = [len(set(t) & actual) for t in tickets]
    union = set().union(*(set(t) for t in tickets))
    avg_dist, min_dist = structure_distance_stats(tickets)
    usage = Counter(n for t in tickets for n in t)
    usage_mean = sum(usage.values()) / len(usage)
    usage_var = sum((v - usage_mean) ** 2 for v in usage.values()) / len(usage)

    return {
        "max_hits": max(hits),
        "mean_hits": sum(hits) / len(hits),
        "tickets_ge3": sum(h >= 3 for h in hits),
        "tickets_ge4": sum(h >= 4 for h in hits),
        "tickets_ge5": sum(h >= 5 for h in hits),
        "tickets_ge6": sum(h >= 6 for h in hits),
        "tickets_ge7": sum(h >= 7 for h in hits),
        "unique_numbers": len(union),
        "winning_unique_contact": len(union & actual),
        "winning_pair_coverage": winning_subset_coverage(tickets, actual, 2),
        "winning_triple_coverage": winning_subset_coverage(tickets, actual, 3),
        "mean_pairwise_overlap": pairwise_overlap_mean(tickets),
        "mean_structure_distance": avg_dist,
        "min_structure_distance": min_dist,
        "number_usage_std": math.sqrt(usage_var),
        "tickets": ";".join("-".join(f"{n:02d}" for n in t) for t in tickets),
    }


def summarize(res: pd.DataFrame) -> dict:
    out = {
        "experiment": "loto7_portfolio20_v0",
        "evaluation_rounds": int(res["round"].nunique()),
        "strategies": {},
        "paired": {},
        "quarter_means": {},
        "boundary": [
            "Same pre-draw CORE18 candidate pool is used for A and B within each round.",
            "Winning unique contact is expected to be identical when both methods touch all CORE18 numbers; it is retained as a control.",
            "No future result is used to construct either bundle.",
            "This experiment tests ticket placement, not a change in the underlying lottery probability.",
        ],
    }

    metrics = [
        "max_hits",
        "tickets_ge3",
        "tickets_ge4",
        "tickets_ge5",
        "winning_unique_contact",
        "winning_pair_coverage",
        "winning_triple_coverage",
        "mean_pairwise_overlap",
        "mean_structure_distance",
        "min_structure_distance",
        "number_usage_std",
    ]
    for strategy in ("structured20", "portfolio20"):
        g = res[res.strategy == strategy]
        out["strategies"][strategy] = {
            m: float(g[m].mean()) for m in metrics
        }
        out["strategies"][strategy]["p_max_ge4"] = float((g.max_hits >= 4).mean())
        out["strategies"][strategy]["p_max_ge5"] = float((g.max_hits >= 5).mean())
        out["strategies"][strategy]["best_max_hits"] = int(g.max_hits.max())

    pivot = res.pivot(index="round", columns="strategy")
    for metric in ("max_hits", "tickets_ge3", "tickets_ge4", "tickets_ge5", "winning_pair_coverage", "winning_triple_coverage"):
        a = pivot[metric]["structured20"]
        b = pivot[metric]["portfolio20"]
        out["paired"][metric] = {
            "B_gt_A": int((b > a).sum()),
            "B_eq_A": int((b == a).sum()),
            "B_lt_A": int((b < a).sum()),
            "mean_delta_B_minus_A": float((b - a).mean()),
        }

    rounds = sorted(res["round"].unique())
    n = len(rounds)
    for qi in range(4):
        lo = qi * n // 4
        hi = (qi + 1) * n // 4
        qrounds = set(rounds[lo:hi])
        q = res[res["round"].isin(qrounds)]
        out["quarter_means"][f"Q{qi+1}"] = {}
        for strategy in ("structured20", "portfolio20"):
            g = q[q.strategy == strategy]
            out["quarter_means"][f"Q{qi+1}"][strategy] = {
                "max_hits": float(g.max_hits.mean()),
                "tickets_ge4": float(g.tickets_ge4.mean()),
                "tickets_ge5": float(g.tickets_ge5.mean()),
                "winning_pair_coverage": float(g.winning_pair_coverage.mean()),
                "winning_triple_coverage": float(g.winning_triple_coverage.mean()),
                "mean_structure_distance": float(g.mean_structure_distance.mean()),
            }
    return out


def run(data_path: Path, model_window: int = MODEL_WINDOW) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    rows = []
    first_trace = None

    for idx in range(model_window, len(df)):
        history = df.iloc[idx - model_window:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))
        pool = ranked[:POOL_K]

        seed = rnd * 100_003 + 20261002
        a = structured_bundle(pool, A_TICKETS, seed + 1)
        b, trace = portfolio_bundle(pool, scores, seed + 2)

        if set().union(*(set(t) for t in a)) != set(pool):
            raise RuntimeError(f"A does not cover CORE18 at round {rnd}")
        if set().union(*(set(t) for t in b)) != set(pool):
            raise RuntimeError(f"B does not cover CORE18 at round {rnd}")

        if first_trace is None:
            first_trace = {
                "round": rnd,
                "history_end_round": int(history.iloc[-1]["round"]),
                "candidate_pool": pool,
                "portfolio_roles": trace,
            }

        for strategy, tickets in (("structured20", a), ("portfolio20", b)):
            rec = {
                "round": rnd,
                "date": target.get("date", ""),
                "strategy": strategy,
                "candidate_pool": "-".join(f"{n:02d}" for n in pool),
            }
            rec.update(bundle_metrics(tickets, actual))
            rows.append(rec)

    res = pd.DataFrame(rows)
    return res, {"summary": summarize(res), "first_trace": first_trace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/loto7.csv"))
    p.add_argument("--out", type=Path, default=Path("results/loto7_portfolio20_v0.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/loto7_portfolio20_v0_summary.json"))
    p.add_argument("--model-window", type=int, default=MODEL_WINDOW)
    a = p.parse_args()

    res, payload = run(a.data, a.model_window)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    a.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    s = payload["summary"]
    print("=== LOTO7 PORTFOLIO20 v0 ===")
    print(f"evaluation_rounds={s['evaluation_rounds']}")
    for name, m in s["strategies"].items():
        print(name, json.dumps(m, ensure_ascii=False, sort_keys=True))
    for name, m in s["paired"].items():
        print("PAIRED", name, json.dumps(m, ensure_ascii=False, sort_keys=True))
    print("QUARTERS", json.dumps(s["quarter_means"], ensure_ascii=False, sort_keys=True))
    print("BOUNDARY", " | ".join(s["boundary"]))
    print(f"saved -> {a.out}")
    print(f"saved -> {a.summary}")


if __name__ == "__main__":
    main()
