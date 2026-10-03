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
    MODEL_WINDOW, POOL_K, A_TICKETS,
    main_numbers, unique_structured_bundle, portfolio_bundle, bundle_metrics,
)
from coverage_lab.loto7_hybrid_ratio_sweep_v0 import select, merge_unique

DATA = Path("coverage_lab/fixtures/loto7_1_696.csv")
OUT = Path("results/generation_freedom_hybrid10_10_vs_random_world_a.json")
STRUCTURED_N = 10
FREEDOM_N = 10
THRESHOLDS = (3, 4, 5)


def favorable_ticket_count(space_n: int, winning_in_space: int, threshold: int) -> int:
    total = 0
    misses = space_n - winning_in_space
    for k in range(threshold, 8):
        if k <= winning_in_space and 7-k <= misses:
            total += math.comb(winning_in_space, k) * math.comb(misses, 7-k)
    return total


def unique_random20_success_probability(space_n: int, favorable: int) -> float:
    universe = math.comb(space_n, 7)
    tickets = 20
    if favorable <= 0:
        return 0.0
    if favorable >= universe:
        return 1.0
    if universe - favorable < tickets:
        return 1.0
    return 1.0 - math.comb(universe-favorable, tickets) / math.comb(universe, tickets)


def poisson_binomial_tail(ps: list[float], observed: int) -> float:
    dist = [0.0] * (len(ps)+1)
    dist[0] = 1.0
    used = 0
    for p in ps:
        used += 1
        for k in range(used, 0, -1):
            dist[k] = dist[k]*(1-p) + dist[k-1]*p
        dist[0] *= (1-p)
    return sum(dist[observed:])


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx-MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k):int(v) for k,v in snap["ranks"].items()}
        scores = {int(k):float(v) for k,v in snap["scores"].items()}
        core18 = sorted(range(1,38), key=lambda n:(ranks[n],n))[:POOL_K]

        seed = rnd*100_003 + 20261002
        structured20, _ = unique_structured_bundle(core18, A_TICKETS, seed+1)
        freedom20, _ = portfolio_bundle(core18, scores, seed+2)

        hybrid20, repairs = merge_unique(
            select(structured20, STRUCTURED_N),
            select(freedom20, FREEDOM_N),
            freedom20,
        )

        hm = bundle_metrics(hybrid20, actual)
        h_core = len(set(core18) & actual)

        rec = {
            "round": rnd,
            "core18_actual_hits": h_core,
            "collision_repairs": repairs,
        }

        for threshold in THRESHOLDS:
            rec[f"hybrid_ge{threshold}"] = int(hm["max_hits"] >= threshold)

            full_fav = favorable_ticket_count(37, 7, threshold)
            rec[f"full_random20_p_ge{threshold}"] = unique_random20_success_probability(37, full_fav)

            core_fav = favorable_ticket_count(18, h_core, threshold)
            rec[f"same_core18_random20_p_ge{threshold}"] = unique_random20_success_probability(18, core_fav)

        rows.append(rec)

    summary = {
        "experiment": "generation_freedom_hybrid10_10_vs_random_world_a",
        "evaluation_rounds": len(rows),
        "hybrid": {
            "structured_n": STRUCTURED_N,
            "freedom_n": FREEDOM_N,
            "selection_rule": "Existing spread-index selection + existing collision repair.",
            "why_fixed": "10+10 is the symmetric equal-budget composition. No ratio search is performed in this run.",
        },
        "thresholds": {},
        "collision_repairs_total": sum(r["collision_repairs"] for r in rows),
        "boundary": [
            "This run is World A descriptive evidence, not independent OOS.",
            "The 10+10 ratio is fixed for this run; no best-of-five ratio selection is allowed.",
            "Random baselines are exact combinatorial expectations for 20 unique tickets, not Monte Carlo.",
            "Full-space Random20 asks whether the Hybrid20 machine beats ordinary random ticket placement.",
            "Same-CORE18 Random20 asks whether the Hybrid20 placement adds value beyond random placement from the same candidate material.",
            "Because 10+10 performance on World A was already observed historically, World A cannot by itself validate the ratio independently.",
            "Independent worlds and future OOS are required before promotion.",
        ],
    }

    for threshold in THRESHOLDS:
        observed = sum(r[f"hybrid_ge{threshold}"] for r in rows)
        full_ps = [r[f"full_random20_p_ge{threshold}"] for r in rows]
        core_ps = [r[f"same_core18_random20_p_ge{threshold}"] for r in rows]

        summary["thresholds"][f"{threshold}plus"] = {
            "hybrid_success_rounds": observed,
            "hybrid_rate": observed/len(rows),
            "full_space_random20": {
                "expected_success_rounds": sum(full_ps),
                "expected_rate": sum(full_ps)/len(full_ps),
                "one_sided_tail_P_random_ge_hybrid": poisson_binomial_tail(full_ps, observed),
            },
            "same_core18_random20": {
                "expected_success_rounds": sum(core_ps),
                "expected_rate": sum(core_ps)/len(core_ps),
                "one_sided_tail_P_random_ge_hybrid": poisson_binomial_tail(core_ps, observed),
            },
        }

    payload = {**summary, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    print("=== GENERATION FREEDOM HYBRID10+10 VS RANDOM — WORLD A ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
