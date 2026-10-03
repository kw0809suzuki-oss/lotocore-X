from __future__ import annotations

from itertools import combinations
import json

from finite_preservation import (
    CandidateState,
    PreservationRelation,
    PreservationState,
    validate_transition,
)


def candidate_subsets_from_constraint(
    available_numbers: list[int],
    gaps: tuple[int, ...],
) -> list[tuple[int, ...]]:
    size = len(gaps) + 1
    out: list[tuple[int, ...]] = []
    nums = sorted(set(int(n) for n in available_numbers))
    for combo in combinations(nums, size):
        actual = tuple(b - a for a, b in zip(combo, combo[1:]))
        if actual == gaps:
            out.append(combo)
    return out


def run_probe() -> dict:
    # Historical case used only as a replay:
    # draw 137 local structure [13,19,21,25] has gaps [6,2,4].
    # Candidate successor mappings are created WITHOUT using draw 142.
    before = CandidateState({
        "A": 13, "B": 19, "C": 21, "D": 25,
        "x1": 27, "x2": 32, "x3": 35,
    })
    preservation = PreservationState(
        status="active",
        members=("A", "B", "C", "D"),
        relation=PreservationRelation(consecutive=True, gaps=(6, 2, 4)),
        remaining=1,
    )

    # Future is unknown at candidate generation time.
    universe = list(range(1, 38))
    candidate_subsets = candidate_subsets_from_constraint(
        universe,
        preservation.relation.gaps,
    )

    # Reveal historical future only after candidate set is frozen.
    actual_future = [6, 12, 14, 18, 21, 27, 28]
    actual_subset = (6, 12, 14, 18)

    after = CandidateState({
        "A2": 6, "B2": 12, "C2": 14, "D2": 18,
        "y1": 21, "y2": 27, "y3": 28,
    })
    mapping = {"A":"A2","B":"B2","C":"C2","D":"D2"}
    validation = validate_transition(before, after, preservation, mapping)

    return {
        "case": "137 -> 142 replay with future-blind mapping candidates",
        "constraint": {
            "member_count": 4,
            "gaps": [6, 2, 4],
            "candidate_generation_uses_future": False,
        },
        "candidate_space": {
            "universe": "1..37",
            "count": len(candidate_subsets),
            "candidates": [list(x) for x in candidate_subsets],
        },
        "future_reveal": {
            "actual_draw": actual_future,
            "actual_subset": list(actual_subset),
            "actual_subset_was_in_candidate_space": actual_subset in candidate_subsets,
        },
        "validator_replay": {
            "accepted": validation.accepted,
            "reason": validation.reason,
            "remaining": validation.next_preservation.remaining,
            "status": validation.next_preservation.status,
        },
        "boundary": (
            "The candidate set is generated only from the previously declared exact-gap "
            "preservation constraint over 1..37. The historical future is revealed afterward. "
            "This tests candidate-space restriction, not predictive skill."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
