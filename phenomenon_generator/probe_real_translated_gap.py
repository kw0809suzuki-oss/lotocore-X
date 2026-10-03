from __future__ import annotations

import json

from finite_preservation import (
    CandidateState,
    PreservationRelation,
    PreservationState,
    validate_transition,
)


def run_probe() -> dict:
    # Observed local relation:
    # draw 137 subset [13,19,21,25] -> draw 142 subset [6,12,14,18]
    # exact translation -7, exact gaps [6,2,4].
    before = CandidateState({
        "A": 13, "B": 19, "C": 21, "D": 25,
        "x1": 27, "x2": 32, "x3": 35,
    })
    after = CandidateState({
        "A2": 6, "B2": 12, "C2": 14, "D2": 18,
        "y1": 21, "y2": 27, "y3": 28,
    })
    p = PreservationState(
        status="active",
        members=("A", "B", "C", "D"),
        relation=PreservationRelation(consecutive=True, gaps=(6, 2, 4)),
        remaining=1,
    )
    mapping = {"A":"A2","B":"B2","C":"C2","D":"D2"}
    r = validate_transition(before, after, p, mapping)
    return {
        "case": "137-142 local translated gap structure",
        "before_subset": [13,19,21,25],
        "after_subset": [6,12,14,18],
        "gaps": [6,2,4],
        "accepted": r.accepted,
        "reason": r.reason,
        "remaining": r.next_preservation.remaining,
        "status": r.next_preservation.status,
        "boundary": (
            "This only shows that v0.1 can represent an explicitly mapped "
            "translated local relation. It does not establish causal continuity."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
