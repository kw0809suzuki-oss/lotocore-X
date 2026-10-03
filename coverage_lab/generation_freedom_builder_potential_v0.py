#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    B_CORE,
    B_DIVERSIFY,
    B_FALSIFY,
    MODEL_WINDOW,
    POOL_K,
    avg_candidate_score,
    bundle_metrics,
    euclid,
    main_numbers,
    portfolio_bundle,
    sampled_combinations,
    standardized_vectors,
    unique_structured_bundle,
)


DATA = Path("coverage_lab/fixtures/loto7_1_696.csv")
OUT = Path("results/generation_freedom_builder_potential_v0.json")


def generation_freedom_builder(
    pool_order: list[int],
    scores: dict[int, float],
    seed: int,
) -> tuple[list[tuple[int, ...]], dict]:
    # Same Core10 and same 2500 candidate world as current Portfolio20.
    core, _ = unique_structured_bundle(pool_order, B_CORE, seed)
    combos = sampled_combinations(pool_order, seed + 5000, core)
    z = standardized_vectors(combos)
    combo_score = {c: avg_candidate_score(c, scores) for c in combos}
    median_score = sorted(combo_score.values())[len(combo_score) // 2]

    core_centroid = tuple(
        sum(z[t][i] for t in core) / len(core)
        for i in range(len(z[core[0]]))
    )
    d = {c: euclid(z[c], core_centroid) for c in combos}

    selected = list(core)
    selected_set = set(core)
    remaining = [c for c in combos if c not in selected_set]

    high_m = [c for c in remaining if combo_score[c] >= median_score]
    low_m = [c for c in remaining if combo_score[c] < median_score]

    if len(high_m) < B_DIVERSIFY or len(low_m) < B_FALSIFY:
        raise RuntimeError("insufficient candidates for fixed 10/6/4 builder")

    # d only. Existing tuple(-n) rule is retained as deterministic tie-break.
    high_m_sorted = sorted(
        high_m,
        key=lambda c: (d[c], tuple(-n for n in c)),
        reverse=True,
    )
    displacement = high_m_sorted[:B_DIVERSIFY]
    selected.extend(displacement)
    selected_set.update(displacement)

    low_m = [c for c in low_m if c not in selected_set]
    low_m_sorted = sorted(
        low_m,
        key=lambda c: (d[c], tuple(-n for n in c)),
        reverse=True,
    )
    exit_unclassified = low_m_sorted[:B_FALSIFY]
    selected.extend(exit_unclassified)

    if len(selected) != 20 or len(set(selected)) != 20:
        raise RuntimeError("builder did not produce 20 unique tickets")

    trace = {
        "core": core,
        "displacement": displacement,
        "exit_unclassified": exit_unclassified,
        "median_score": median_score,
        "d": {c: d[c] for c in selected},
        "m": {c: combo_score[c] for c in selected},
    }
    return selected, trace


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def observe() -> dict:
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
        pool = ranked[:POOL_K]

        seed = rnd * 100_003 + 20261002
        portfolio, _ = portfolio_bundle(pool, scores, seed + 2)
        builder, trace = generation_freedom_builder(pool, scores, seed + 2)

        if portfolio[:B_CORE] != builder[:B_CORE]:
            raise RuntimeError(f"Core10 mismatch at round {rnd}")

        a_set = set(portfolio)
        b_set = set(builder)
        overlap = len(a_set & b_set)
        union = len(a_set | b_set)
        jaccard = overlap / union

        ma = bundle_metrics(portfolio, actual)
        mb = bundle_metrics(builder, actual)
        a_success = ma["tickets_ge5"] > 0
        b_success = mb["tickets_ge5"] > 0

        core = trace["core"]
        displacement = trace["displacement"]
        exit_u = trace["exit_unclassified"]

        rows.append(
            {
                "round": rnd,
                "overlap_tickets": overlap,
                "union_tickets": union,
                "jaccard": jaccard,
                "exact_same_bundle": a_set == b_set,
                "portfolio_success_5plus": a_success,
                "builder_success_5plus": b_success,
                "core_mean_d": mean([trace["d"][t] for t in core]),
                "displacement_mean_d": mean([trace["d"][t] for t in displacement]),
                "exit_unclassified_mean_d": mean([trace["d"][t] for t in exit_u]),
                "core_mean_m": mean([trace["m"][t] for t in core]),
                "displacement_mean_m": mean([trace["m"][t] for t in displacement]),
                "exit_unclassified_mean_m": mean([trace["m"][t] for t in exit_u]),
                "displacement_all_m_ge_median": all(
                    trace["m"][t] >= trace["median_score"] for t in displacement
                ),
                "exit_all_m_below_median": all(
                    trace["m"][t] < trace["median_score"] for t in exit_u
                ),
            }
        )

    a_only = [r["round"] for r in rows if r["portfolio_success_5plus"] and not r["builder_success_5plus"]]
    b_only = [r["round"] for r in rows if r["builder_success_5plus"] and not r["portfolio_success_5plus"]]
    both = [r["round"] for r in rows if r["portfolio_success_5plus"] and r["builder_success_5plus"]]
    neither = [r["round"] for r in rows if not r["portfolio_success_5plus"] and not r["builder_success_5plus"]]

    payload = {
        "experiment": "generation_freedom_builder_potential_v0",
        "question": "Are d/m only explanatory coordinates for current Portfolio20, or can they independently recompose a 20-ticket bundle with different reachable success regions?",
        "fixed_conditions": {
            "same_round": True,
            "same_core18": True,
            "same_2500_candidates": True,
            "same_seed": True,
            "same_core10": True,
            "tickets": 20,
            "ratio": "10/6/4 reused",
            "future_info_in_generation": False,
            "threshold_search": False,
            "q": "unknown",
        },
        "builder_rule": {
            "core": "same existing Core10",
            "structural_displacement": "select 6 largest d among candidates with m >= round median",
            "exit_unclassified": "select 4 largest d among candidates with m < round median",
            "tie_break": "existing deterministic tuple(-n) final ordering",
        },
        "gate1_recomposition": {
            "evaluation_rounds": len(rows),
            "exact_same_bundle_rounds": sum(r["exact_same_bundle"] for r in rows),
            "mean_jaccard": mean([r["jaccard"] for r in rows]),
            "median_jaccard": float(pd.Series([r["jaccard"] for r in rows]).median()),
            "min_jaccard": min(r["jaccard"] for r in rows),
            "max_jaccard": max(r["jaccard"] for r in rows),
            "all_displacement_m_ge_median": all(r["displacement_all_m_ge_median"] for r in rows),
            "all_exit_m_below_median": all(r["exit_all_m_below_median"] for r in rows),
            "mean_d_by_slice": {
                "core": mean([r["core_mean_d"] for r in rows]),
                "structural_displacement": mean([r["displacement_mean_d"] for r in rows]),
                "exit_unclassified": mean([r["exit_unclassified_mean_d"] for r in rows]),
            },
            "mean_m_by_slice": {
                "core": mean([r["core_mean_m"] for r in rows]),
                "structural_displacement": mean([r["displacement_mean_m"] for r in rows]),
                "exit_unclassified": mean([r["exit_unclassified_mean_m"] for r in rows]),
            },
        },
        "gate2_success_sets": {
            "portfolio_success_rounds": sum(r["portfolio_success_5plus"] for r in rows),
            "builder_success_rounds": sum(r["builder_success_5plus"] for r in rows),
            "portfolio_only_count": len(a_only),
            "builder_only_count": len(b_only),
            "both_count": len(both),
            "neither_count": len(neither),
            "union_success_rounds": len(a_only) + len(b_only) + len(both),
            "portfolio_only_rounds": a_only,
            "builder_only_rounds": b_only,
            "both_rounds": both,
        },
        "boundary": [
            "This probe does not add a predictor, score, threshold search, or q implementation.",
            "Generation differs only in how the final 10 non-Core tickets are selected from the same existing candidate world.",
            "Builder-only success is interpreted only as nonredundant reach, not as superiority.",
            "Role labels are not used as selection criteria in Builder v0.",
        ],
        "rounds": rows,
    }
    return payload


def main():
    payload = observe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== GENERATION FREEDOM BUILDER POTENTIAL v0 ===")
    print("GATE1", json.dumps(payload["gate1_recomposition"], ensure_ascii=False, sort_keys=True))
    print("GATE2", json.dumps(payload["gate2_success_sets"], ensure_ascii=False, sort_keys=True))
    print("BOUNDARY", " | ".join(payload["boundary"]))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
