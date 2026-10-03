from __future__ import annotations

from itertools import combinations
import json

from finite_preservation import (
    CandidateState,
    PreservationRelation,
    PreservationState,
    validate_transition,
)


def exact_gap_candidates(gaps: tuple[int, ...]) -> list[tuple[int, ...]]:
    size = len(gaps) + 1
    out = []
    for combo in combinations(range(1, 38), size):
        if tuple(b - a for a, b in zip(combo, combo[1:])) == gaps:
            out.append(combo)
    return out


def run_probe() -> dict:
    # Consecutive committed states: draw 324 -> draw 325.
    # At candidate-generation time only draw 324's preserved local relation is used.
    draw324 = [9, 13, 16, 17, 18, 19, 20]
    preserved_before = (17, 18)
    gaps = (1,)

    candidates = exact_gap_candidates(gaps)

    # Freeze candidate set, then reveal draw 325.
    draw325 = [4, 14, 17, 18, 21, 31, 33]
    realized_candidates = [
        c for c in candidates if all(n in draw325 for n in c)
    ]

    before = CandidateState({
        "A": 17, "B": 18,
        "x9": 9, "x13": 13, "x16": 16, "x19": 19, "x20": 20,
    })
    after = CandidateState({
        "A2": 17, "B2": 18,
        "y4": 4, "y14": 14, "y21": 21, "y31": 31, "y33": 33,
    })
    p = PreservationState(
        status="active",
        members=("A", "B"),
        relation=PreservationRelation(consecutive=True, gaps=gaps),
        remaining=1,
    )
    v = validate_transition(before, after, p, {"A":"A2","B":"B2"})

    return {
        "case": "324 -> 325 consecutive future-blind replay",
        "candidate_generation": {
            "uses_draw325": False,
            "preserved_before": list(preserved_before),
            "gaps": list(gaps),
            "candidate_count_before_reveal": len(candidates),
        },
        "future_reveal": {
            "draw325": draw325,
            "realized_candidates": [list(x) for x in realized_candidates],
            "realized_count": len(realized_candidates),
        },
        "validator": {
            "accepted": v.accepted,
            "remaining": v.next_preservation.remaining,
            "status": v.next_preservation.status,
        },
        "boundary": (
            "This is a one-step historical replay. Candidate positions are generated "
            "from the pre-existing preservation relation before the next draw is revealed. "
            "It measures constraint-induced candidate restriction, not predictive skill."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
