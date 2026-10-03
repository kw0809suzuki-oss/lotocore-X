#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K, A_TICKETS,
    main_numbers, unique_structured_bundle, portfolio_bundle,
)
from coverage_lab.loto7_naive_parallel_hybrid20_v0 import A_IDX, merge_unique

EVENTS = {
    134: "added",
    249: "added",
    374: "added",
    413: "added",
    440: "added",
    518: "added",
    217: "lost",
    438: "lost",
    676: "lost",
}
KS = (1, 2, 3, 4)


def atoms(ticket: tuple[int, ...], k: int) -> set[tuple[int, ...]]:
    return set(itertools.combinations(ticket, k))


def bundle_atom_counts(tickets: list[tuple[int, ...]], k: int) -> Counter:
    c = Counter()
    for t in tickets:
        c.update(atoms(t, k))
    return c


def unique_loss(full_counts: dict[int, Counter], ticket: tuple[int, ...]) -> dict[int, set[tuple[int, ...]]]:
    return {
        k: {a for a in atoms(ticket, k) if full_counts[k][a] == 1}
        for k in KS
    }


def replacement_count(
    pool: list[int],
    remaining: list[tuple[int, ...]],
    removed: tuple[int, ...],
    required_atoms: list[set[tuple[int, ...]]],
) -> tuple[int, list[int]]:
    required_numbers = sorted(set().union(*(set(x) for aset in required_atoms for x in aset)))
    if len(required_numbers) > 7:
        return 0, required_numbers

    required = set(required_numbers)
    free = [n for n in pool if n not in required]
    need = 7 - len(required)
    if need < 0:
        return 0, required_numbers

    blocked = set(remaining)
    blocked.add(removed)
    count = 0
    for extra in itertools.combinations(free, need):
        candidate = tuple(sorted(required | set(extra)))
        if candidate not in blocked:
            count += 1
    return count, required_numbers


def leave_one_out_map(
    tickets: list[tuple[int, ...]],
    pool: list[int],
    source: str,
    role_by_ticket: dict[tuple[int, ...], str] | None = None,
) -> list[dict]:
    full_counts = {k: bundle_atom_counts(tickets, k) for k in KS}
    rows = []
    for idx, t in enumerate(tickets):
        remaining = tickets[:idx] + tickets[idx + 1:]
        lost = unique_loss(full_counts, t)
        per_k = {}
        for k in KS:
            rc, req = replacement_count(pool, remaining, t, [lost[k]])
            per_k[str(k)] = {
                "lost_unique_atoms": len(lost[k]),
                "required_numbers_to_restore": req,
                "distinct_legal_replacements": rc,
                "irrecoverable_with_one_other_core18_ticket": bool(lost[k]) and rc == 0,
            }

        primary_lost = [lost[3], lost[4]]
        joint_count, joint_req = replacement_count(pool, remaining, t, primary_lost)
        rows.append({
            "source": source,
            "index0": idx,
            "ticket": list(t),
            "role": None if role_by_ticket is None else role_by_ticket.get(t),
            "function_delta": per_k,
            "primary_function": {
                "definition": "unique 3-subset and 4-subset coverage lost by removing this ticket",
                "lost_unique_triples": len(lost[3]),
                "lost_unique_quadruples": len(lost[4]),
                "required_numbers_to_restore_both": joint_req,
                "distinct_legal_core18_replacements_restoring_both": joint_count,
                "irrecoverable_with_one_other_core18_ticket": bool(lost[3] or lost[4]) and joint_count == 0,
            },
            "provenance": {
                "source_generator": source,
                "role": None if role_by_ticket is None else role_by_ticket.get(t),
                "position0": idx,
            },
            "residual": {
                "status": "unknown",
                "reason": "No existing implementation quantity is promoted to Residual in this probe.",
            },
        })

    # Pre-draw within-round ranks. Higher rank = larger unique loss.
    for field in ("lost_unique_triples", "lost_unique_quadruples"):
        vals = sorted((r["primary_function"][field], r["index0"]) for r in rows)
        rank = {idx: i + 1 for i, (_, idx) in enumerate(vals)}
        for r in rows:
            r["primary_function"][f"rank_low_to_high_{field}"] = rank[r["index0"]]
    return rows


def main() -> None:
    df = pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)
    rounds = []
    for rnd, event in EVENTS.items():
        hit = df.index[df["round"].astype(int) == rnd].tolist()
        if len(hit) != 1:
            raise RuntimeError(f"round {rnd} not found exactly once")
        idx = hit[0]
        history = df.iloc[idx - MODEL_WINDOW:idx]
        target = df.iloc[idx]
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        pool = sorted(range(1, 38), key=lambda n: (ranks[n], n))[:POOL_K]
        seed = rnd * 100_003 + 20261002

        structured20, _ = unique_structured_bundle(pool, A_TICKETS, seed + 1)
        portfolio20, trace = portfolio_bundle(pool, scores, seed + 2)
        role_by_ticket = {}
        for role in ("core", "diversify", "falsify"):
            for t in trace[role]:
                role_by_ticket[tuple(t)] = role

        structured10 = [structured20[i] for i in A_IDX]
        hybrid20, repairs = merge_unique(structured10, portfolio20)

        # Build maps BEFORE attaching any outcome labels.
        structured_map = leave_one_out_map(structured20, pool, "structured20")
        hybrid_map = leave_one_out_map(hybrid20, pool, "hybrid20", role_by_ticket)

        # Outcome overlay only after the pre-draw maps exist.
        s_winners = [i for i,t in enumerate(structured20) if len(set(t) & actual) >= 5]
        p_winners = [i for i,t in enumerate(portfolio20) if len(set(t) & actual) >= 5]
        h_winners = [i for i,t in enumerate(hybrid20) if len(set(t) & actual) >= 5]

        removed_structured = [i for i in range(20) if i not in A_IDX]
        lost_winning_structured = [i for i in s_winners if i in removed_structured]
        incoming_hybrid_winners = [
            i for i in h_winners
            if hybrid20[i] not in set(structured10)
        ]

        overlay = {
            "event": event,
            "structured_5plus_indices0": s_winners,
            "portfolio_5plus_indices0": p_winners,
            "hybrid_5plus_indices0": h_winners,
            "lost_winning_structured_indices0": lost_winning_structured,
            "incoming_hybrid_winner_indices0": incoming_hybrid_winners,
            "lost_winning_structured_metrics": [
                structured_map[i] for i in lost_winning_structured
            ],
            "incoming_hybrid_winner_metrics": [
                hybrid_map[i] for i in incoming_hybrid_winners
            ],
        }

        rounds.append({
            "round": rnd,
            "event": event,
            "candidate_pool": pool,
            "collision_repairs": repairs,
            "pre_draw_maps": {
                "structured20": structured_map,
                "hybrid20": hybrid_map,
            },
            "outcome_overlay": overlay,
        })

    payload = {
        "experiment": "irrecoverable_loss_observer_v0",
        "mode": "observation_only",
        "event_rounds": EVENTS,
        "function_definition": {
            "primary": "leave-one-out unique triple/quadruple coverage in the 20-ticket set",
            "diagnostic": "unique number/pair coverage also recorded",
            "why": "3+/4+ coverage is the existing evidence-backed function; no winning numbers are used to build the map.",
        },
        "recovery_definition": {
            "recover": "whether the remaining 19 already retain the atom (equivalent to zero unique loss)",
            "reconstruct": "whether one distinct legal 7-of-CORE18 ticket can restore every lost primary atom",
            "regenerate": "not operationalized in v0; generator-specific re-generation is left unknown rather than invented.",
        },
        "provenance_definition": "trace only: source generator / role / position; provenance is not scored as value.",
        "residual_definition": "unknown in v0; no new residual score is invented.",
        "rounds": rounds,
        "boundary": [
            "The nine rounds are outcome-conditioned event rounds, so this probe is explanatory and not predictive validation.",
            "All leave-one-out maps and recovery calculations use only pre-draw generated tickets and CORE18.",
            "Winning numbers are attached only in a later overlay step.",
            "No Preserve/Exchange/Unknown label is assigned by this observer.",
            "No composite score is created.",
            "Residual and generator-specific regeneration remain unknown rather than being backfilled by new features.",
        ],
    }

    out = Path("results/irrecoverable_loss_observer_v0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== IRRECOVERABLE LOSS OBSERVER v0 ===")
    for r in rounds:
        o = r["outcome_overlay"]
        print(
            "ROUND", r["round"], r["event"],
            "lost_structured", [
                {
                    "idx": x["index0"],
                    "triple_loss": x["primary_function"]["lost_unique_triples"],
                    "quad_loss": x["primary_function"]["lost_unique_quadruples"],
                    "replacements": x["primary_function"]["distinct_legal_core18_replacements_restoring_both"],
                }
                for x in o["lost_winning_structured_metrics"]
            ],
            "incoming_hybrid", [
                {
                    "idx": x["index0"],
                    "role": x["role"],
                    "triple_loss": x["primary_function"]["lost_unique_triples"],
                    "quad_loss": x["primary_function"]["lost_unique_quadruples"],
                    "replacements": x["primary_function"]["distinct_legal_core18_replacements_restoring_both"],
                }
                for x in o["incoming_hybrid_winner_metrics"]
            ],
        )
    print("saved ->", out)


if __name__ == "__main__":
    main()
