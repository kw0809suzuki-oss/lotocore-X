from __future__ import annotations

import json

from finite_preservation import (
    CandidateState,
    PreservationRelation,
    PreservationState,
    validate_transition,
)


def run_probe() -> dict:
    # Observed 309 -> 310
    before = CandidateState({
        "A": 1, "B": 7, "C": 13, "D": 14, "E": 16, "F": 21, "G": 22
    })
    after = CandidateState({
        "A2": 17, "B2": 25, "C2": 26, "D2": 28, "E2": 31, "F2": 32, "G2": 37
    })

    # Rank-preserving explicit mapping is supplied deliberately.
    # v0.1 still requires exact adjacent gaps, so this case should reject.
    p = PreservationState(
        status="active",
        members=("A", "B", "C", "D", "E", "F", "G"),
        relation=PreservationRelation(consecutive=True, gaps=(6, 6, 1, 2, 5, 1)),
        remaining=1,
    )
    mapping = {
        "A":"A2","B":"B2","C":"C2","D":"D2","E":"E2","F":"F2","G":"G2"
    }
    r = validate_transition(before, after, p, mapping)

    return {
        "case": "309-310 width-preserving opposite-side jump",
        "before_width": 22 - 1,
        "after_width": 37 - 17,
        "before_gaps": [6, 6, 1, 2, 5, 1],
        "after_gaps": [8, 1, 2, 3, 1, 5],
        "explicit_rank_mapping": True,
        "finite_preservation_v0_1": {
            "accepted": r.accepted,
            "reason": r.reason,
            "remaining": r.next_preservation.remaining,
            "status": r.next_preservation.status,
        },
        "boundary": (
            "v0.1 is not a width-preservation validator. "
            "REJECT here is expected and preserves the declared scope."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
