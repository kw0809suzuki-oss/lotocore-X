from __future__ import annotations

from dataclasses import asdict
import json

from finite_preservation import (
    CandidateState,
    PreservationRelation,
    PreservationState,
    validate_transition,
)


def run_probe() -> dict:
    # Same observed sequence: 324 -> 325 -> 326
    s324 = CandidateState({
        "p17": 17,
        "p18": 18,
        "x9": 9,
        "x13": 13,
        "x16": 16,
        "x19": 19,
        "x20": 20,
    })
    s325 = CandidateState({
        "p17": 17,
        "p18": 18,
        "y4": 4,
        "y14": 14,
        "y21": 21,
        "y31": 31,
        "y33": 33,
    })
    s326 = CandidateState({
        "p17": 17,
        "p18": 18,
        "z1": 1,
        "z2": 2,
        "z24": 24,
        "z27": 27,
        "z29": 29,
    })

    preservation = PreservationState(
        status="active",
        members=("p17", "p18"),
        relation=PreservationRelation(consecutive=True, gaps=(1,)),
        remaining=2,
    )

    r1 = validate_transition(
        s324,
        s325,
        preservation,
        {"p17": "p17", "p18": "p18"},
    )
    r2 = validate_transition(
        s325,
        s326,
        r1.next_preservation,
        {"p17": "p17", "p18": "p18"},
    )

    # Counterfactual candidate from the same 325 state:
    # keep 17 but drift 18 -> 19. Old hand-written steps can simply omit this
    # candidate; Finite Preservation can explicitly reject it before commit.
    bad326 = CandidateState({
        "p17": 17,
        "p18": 19,
        "z1": 1,
        "z2": 2,
        "z24": 24,
        "z27": 27,
        "z29": 29,
    })
    rejected = validate_transition(
        s325,
        bad326,
        r1.next_preservation,
        {"p17": "p17", "p18": "p18"},
    )

    return {
        "case": "324-326 fixed pair",
        "old_steps_observation": {
            "can_reproduce_committed_sequence": True,
            "preserve_is_redeclared_per_step": True,
            "persistent_constraint_state": False,
            "rejected_candidate_is_first_class": False,
            "remaining_release_state": False,
        },
        "finite_preservation_observation": {
            "transition_324_325": {
                "accepted": r1.accepted,
                "remaining": r1.next_preservation.remaining,
                "status": r1.next_preservation.status,
            },
            "transition_325_326": {
                "accepted": r2.accepted,
                "remaining": r2.next_preservation.remaining,
                "status": r2.next_preservation.status,
            },
            "counterfactual_325_bad326": {
                "accepted": rejected.accepted,
                "reason": rejected.reason,
                "remaining": rejected.next_preservation.remaining,
                "status": rejected.next_preservation.status,
            },
        },
        "boundary": (
            "This probe compares representational behavior only. "
            "It does not establish predictive value, causal mechanism, or final design necessity."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
