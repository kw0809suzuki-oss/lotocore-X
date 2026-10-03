#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW,
    POOL_K,
    bundle_metrics,
    main_numbers,
    portfolio_bundle,
)

DATA = Path("coverage_lab/fixtures/loto7_1_696.csv")
OUT = Path("results/generation_freedom_controller_v1_vs_random_exact.json")
TICKETS = 20
THRESHOLDS = (3, 4, 5)


def favorable_ticket_count(space_n: int, target_hits_available: int, threshold: int) -> int:
    # A 7-number ticket from a space of space_n numbers. target_hits_available of
    # those numbers are winning numbers. Count tickets with >= threshold hits.
    total = 0
    misses_available = space_n - target_hits_available
    for k in range(threshold, 8):
        if k <= target_hits_available and 7 - k <= misses_available:
            total += math.comb(target_hits_available, k) * math.comb(misses_available, 7 - k)
    return total


def bundle_success_probability(space_n: int, favorable: int, tickets: int = TICKETS) -> float:
    universe = math.comb(space_n, 7)
    if favorable <= 0:
        return 0.0
    if favorable >= universe:
        return 1.0
    # Exactly tickets unique tickets sampled uniformly without replacement
    # from the legal 7-number ticket universe.
    if universe - favorable < tickets:
        return 1.0
    return 1.0 - math.comb(universe - favorable, tickets) / math.comb(universe, tickets)


def poisson_binomial_tail(ps: list[float], observed: int) -> float:
    # DP distribution for sum of independent Bernoulli variables with varying p.
    dist = [0.0] * (len(ps) + 1)
    dist[0] = 1.0
    used = 0
    for p in ps:
        used += 1
        for k in range(used, 0, -1):
            dist[k] = dist[k] * (1.0 - p) + dist[k - 1] * p
        dist[0] *= (1.0 - p)
    return sum(dist[observed:])


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx - MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))
        core18 = ranked[:POOL_K]

        seed = rnd * 100_003 + 20261002
        machine20, _ = portfolio_bundle(core18, scores, seed + 2)
        mm = bundle_metrics(machine20, actual)

        h_core = len(set(core18) & actual)
        rec = {
            "round": rnd,
            "core18_actual_hits": h_core,
        }

        for threshold in THRESHOLDS:
            rec[f"machine_ge{threshold}"] = int(mm["max_hits"] >= threshold)

            full_fav = favorable_ticket_count(37, 7, threshold)
            rec[f"full_random20_p_ge{threshold}"] = bundle_success_probability(37, full_fav)

            core_fav = favorable_ticket_count(18, h_core, threshold)
            rec[f"same_core18_random20_p_ge{threshold}"] = bundle_success_probability(18, core_fav)

        rows.append(rec)

    summary = {
        "experiment": "generation_freedom_controller_v1_vs_random_exact",
        "evaluation_rounds": len(rows),
        "machine": "Current Portfolio20 = Controller v1 Diversify6 + existing Core10 + existing Falsify4",
        "ticket_budget": 20,
        "comparators": {
            "full_space_random20": "20 unique legal 7-of-37 tickets sampled uniformly without replacement",
            "same_core18_random20": "20 unique 7-number tickets sampled uniformly without replacement from the exact same pre-draw CORE18 for that round",
        },
        "thresholds": {},
        "boundary": [
            "No Monte Carlo is used for the random baselines; probabilities are exact combinatorial values.",
            "Full-space Random20 tests the end-to-end machine against ordinary uniform random ticketing.",
            "Same-CORE18 Random20 holds candidate material fixed and tests whether 20-ticket placement adds value beyond random placement inside the same CORE18.",
            "Machine tickets use only pre-draw history, but rounds 101-696 are the historical development sample; this comparison is descriptive and is not independent OOS evidence.",
            "Future OOS freezes remain necessary for a claim about forward performance.",
            "No threshold, seed, role ratio, or generator parameter is tuned in this comparison.",
        ],
    }

    for threshold in THRESHOLDS:
        observed = sum(r[f"machine_ge{threshold}"] for r in rows)

        full_ps = [r[f"full_random20_p_ge{threshold}"] for r in rows]
        core_ps = [r[f"same_core18_random20_p_ge{threshold}"] for r in rows]

        summary["thresholds"][f"{threshold}plus"] = {
            "machine_success_rounds": observed,
            "machine_rate": observed / len(rows),
            "full_space_random20": {
                "expected_success_rounds": sum(full_ps),
                "expected_rate": sum(full_ps) / len(full_ps),
                "one_sided_tail_P_random_ge_machine": poisson_binomial_tail(full_ps, observed),
            },
            "same_core18_random20": {
                "expected_success_rounds": sum(core_ps),
                "expected_rate": sum(core_ps) / len(core_ps),
                "one_sided_tail_P_random_ge_machine": poisson_binomial_tail(core_ps, observed),
            },
        }

    payload = {
        **summary,
        "rows": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== GENERATION FREEDOM CONTROLLER v1 VS RANDOM EXACT ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
