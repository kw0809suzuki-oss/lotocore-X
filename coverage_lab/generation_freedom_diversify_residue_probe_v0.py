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
    B_CORE,
    B_DIVERSIFY,
    CANDIDATE_SAMPLE,
    MODEL_WINDOW,
    POOL_K,
    avg_candidate_score,
    euclid,
    main_numbers,
    max_overlap,
    sampled_combinations,
    standardized_vectors,
    unique_structured_bundle,
)
from coverage_lab.generation_freedom_builder_potential_v0 import (
    generation_freedom_builder,
)

DATA = Path("coverage_lab/fixtures/loto7_1_696.csv")
OUT = Path("results/generation_freedom_diversify_residue_probe_v0.json")
LOST_ROUNDS = {314, 359, 374, 413, 440}


def hit_count(ticket: tuple[int, ...], actual: set[int]) -> int:
    return len(set(ticket) & actual)


def replay_diversify(
    pool: list[int],
    scores: dict[int, float],
    seed: int,
    actual: set[int],
) -> tuple[list[dict], dict]:
    core, _ = unique_structured_bundle(pool, B_CORE, seed)
    combos = sampled_combinations(pool, seed + 5000, core, target_size=CANDIDATE_SAMPLE)
    z = standardized_vectors(combos)
    combo_score = {c: avg_candidate_score(c, scores) for c in combos}
    median_score = sorted(combo_score.values())[len(combo_score) // 2]

    core_centroid = tuple(
        sum(z[t][i] for t in core) / len(core)
        for i in range(len(z[core[0]]))
    )
    centroid_d = {c: euclid(z[c], core_centroid) for c in combos}

    selected = list(core)
    selected_set = set(selected)
    remaining = [c for c in combos if c not in selected_set]

    min_dist = {c: min(euclid(z[c], z[s]) for s in selected) for c in remaining}
    overlap = {c: max_overlap(c, selected) for c in remaining}

    initial_min_dist = dict(min_dist)
    initial_overlap = dict(overlap)

    diversify_candidates = [c for c in remaining if combo_score[c] >= median_score]
    steps = []

    for step in range(1, B_DIVERSIFY + 1):
        max_md = max(min_dist[c] for c in diversify_candidates)
        # Exact float equality is appropriate here because the actual selector compares
        # the stored float values directly before any secondary key is considered.
        distance_ties = [c for c in diversify_candidates if min_dist[c] == max_md]

        best = max(
            diversify_candidates,
            key=lambda c: (
                min_dist[c],
                -overlap[c],
                combo_score[c],
                tuple(-n for n in c),
            ),
        )

        steps.append(
            {
                "step": step,
                "ticket": list(best),
                "hit_count": hit_count(best, actual),
                "is_5plus": hit_count(best, actual) >= 5,
                "centroid_d": centroid_d[best],
                "m": combo_score[best],
                "initial_min_dist_to_core": initial_min_dist[best],
                "initial_overlap_to_core": initial_overlap[best],
                "selection_time_min_dist": min_dist[best],
                "selection_time_overlap": overlap[best],
                "distance_tie_count": len(distance_ties),
                "overlap_could_affect_selection": len(distance_ties) > 1,
            }
        )

        selected.append(best)
        selected_set.add(best)
        diversify_candidates.remove(best)

        best_set = set(best)
        for c in remaining:
            if c in selected_set:
                continue
            min_dist[c] = min(min_dist[c], euclid(z[c], z[best]))
            overlap[c] = max(overlap[c], len(set(c) & best_set))

    high_m = [c for c in remaining if combo_score[c] >= median_score]
    by_centroid_d = sorted(
        high_m,
        key=lambda c: (centroid_d[c], tuple(-n for n in c)),
        reverse=True,
    )
    centroid_rank = {c: i + 1 for i, c in enumerate(by_centroid_d)}

    return steps, {
        "core": core,
        "median_score": median_score,
        "centroid_rank": centroid_rank,
        "centroid_d": centroid_d,
        "combo_score": combo_score,
        "initial_min_dist": initial_min_dist,
        "initial_overlap": initial_overlap,
    }


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    all_steps = []
    lost_details = []

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
        steps, obs = replay_diversify(pool, scores, seed + 2, actual)
        builder, builder_trace = generation_freedom_builder(pool, scores, seed + 2)
        builder_disp = {tuple(t) for t in builder_trace["displacement"]}

        for s in steps:
            rec = {"round": rnd, **s, "selected_by_builder_displacement": tuple(s["ticket"]) in builder_disp}
            all_steps.append(rec)

        if rnd in LOST_ROUNDS:
            winners = [s for s in steps if s["is_5plus"]]
            if not winners:
                raise RuntimeError(f"lost round {rnd} has no 5+ diversify ticket")
            for w in winners:
                t = tuple(w["ticket"])
                lost_details.append(
                    {
                        "round": rnd,
                        **w,
                        "selected_by_builder_displacement": t in builder_disp,
                        "centroid_d_rank_among_high_m": obs["centroid_rank"][t],
                    }
                )

    tie_steps = [s for s in all_steps if s["distance_tie_count"] > 1]
    unique_distance_steps = [s for s in all_steps if s["distance_tie_count"] == 1]

    lost_tie = [s for s in lost_details if s["distance_tie_count"] > 1]

    payload = {
        "experiment": "generation_freedom_diversify_residue_probe_v0",
        "question": "In the five Diversify-only successes lost by the d/m Builder, did existing min_dist or overlap carry the selection information that centroid distance alone discarded?",
        "lost_rounds": sorted(LOST_ROUNDS),
        "selection_order_fact": {
            "current_diversify_key": [
                "min_dist primary",
                "-overlap secondary",
                "combo_score tertiary",
                "deterministic tuple ordering final",
            ],
            "interpretation_boundary": "overlap can affect a choice only when candidates tie on the stored primary min_dist value",
        },
        "all_596_rounds": {
            "diversify_selection_steps": len(all_steps),
            "steps_with_unique_primary_min_dist": len(unique_distance_steps),
            "steps_with_primary_min_dist_tie": len(tie_steps),
            "rounds_with_any_primary_min_dist_tie": len({s["round"] for s in tie_steps}),
            "selected_diversify_ticket_also_in_builder_displacement": sum(
                s["selected_by_builder_displacement"] for s in all_steps
            ),
        },
        "lost_five_rounds": {
            "winning_diversify_ticket_count": len(lost_details),
            "winning_tickets_selected_by_builder": sum(
                s["selected_by_builder_displacement"] for s in lost_details
            ),
            "winning_steps_with_primary_min_dist_tie": len(lost_tie),
            "details": lost_details,
        },
        "boundary": [
            "No new feature, score, threshold, predictor, world, or generator is introduced.",
            "This replays the existing Diversify lexicographic selection and records values already present in that selection.",
            "centroid_d_rank is descriptive only and uses the Builder's already-existing d ordering.",
            "A unique primary min_dist at a step means overlap could not have changed that step's selected ticket under the existing selector.",
            "This probe does not claim causal importance beyond the observed selection mechanism and the five lost success rounds.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== GENERATION FREEDOM DIVERSIFY RESIDUE PROBE v0 ===")
    print("ALL", json.dumps(payload["all_596_rounds"], ensure_ascii=False, sort_keys=True))
    print("LOST5", json.dumps(payload["lost_five_rounds"], ensure_ascii=False, sort_keys=True))
    print("BOUNDARY", " | ".join(payload["boundary"]))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
