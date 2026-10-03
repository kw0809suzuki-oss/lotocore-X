from __future__ import annotations

import unittest

from finite_preservation import (
    CandidateState,
    PreservationRelation,
    PreservationState,
    explicit_pass,
    validate_transition,
)


def pstate(remaining: int = 2) -> PreservationState:
    return PreservationState(
        status="active",
        members=("A", "B", "C"),
        relation=PreservationRelation(consecutive=True, gaps=(1, 2)),
        remaining=remaining,
    )


class FinitePreservationStateTest(unittest.TestCase):
    def test_t1_parallel_move_accept(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A2": 16, "B2": 17, "C2": 19})
        r = validate_transition(before, after, pstate(), {"A": "A2", "B": "B2", "C": "C2"})
        self.assertTrue(r.accepted)
        self.assertEqual(r.next_preservation.remaining, 1)

    def test_t2_single_member_drift_reject(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A": 12, "B": 14, "C": 15})
        r = validate_transition(before, after, pstate(), {"A": "A", "B": "B", "C": "C"})
        self.assertFalse(r.accepted)
        self.assertEqual(r.next_preservation.remaining, 2)

    def test_t3_outside_insert_accept(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A": 12, "B": 13, "C": 15, "D": 20})
        r = validate_transition(before, after, pstate(), {"A": "A", "B": "B", "C": "C"})
        self.assertTrue(r.accepted)

    def test_t4_inside_insert_reject(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A": 12, "B": 13, "C": 15, "D": 14})
        r = validate_transition(before, after, pstate(), {"A": "A", "B": "B", "C": "C"})
        self.assertFalse(r.accepted)

    def test_t5_scope_delete_reject(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A": 12, "C": 15})
        r = validate_transition(before, after, pstate(), {"A": "A", "C": "C"})
        self.assertFalse(r.accepted)

    def test_t6_unrelated_same_shape_reject(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"X": 20, "Y": 21, "Z": 23})
        r = validate_transition(before, after, pstate(), {})
        self.assertFalse(r.accepted)

    def test_t7_explicit_pass_accepts_and_consumes(self):
        state = CandidateState({"A": 12, "B": 13, "C": 15})
        r = explicit_pass(state, pstate())
        self.assertTrue(r.accepted)
        self.assertEqual(r.next_preservation.remaining, 1)

    def test_t8_reject_keeps_state_and_remaining(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A": 12, "B": 14, "C": 15})
        ps = pstate()
        r = validate_transition(before, after, ps, {"A": "A", "B": "B", "C": "C"})
        self.assertFalse(r.accepted)
        self.assertEqual(r.next_preservation, ps)

    def test_compound_move_and_outside_insert_accept(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A2": 16, "B2": 17, "C2": 19, "D": 25})
        r = validate_transition(before, after, pstate(), {"A": "A2", "B": "B2", "C": "C2"})
        self.assertTrue(r.accepted)

    def test_compound_move_and_inside_insert_reject(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A2": 16, "B2": 17, "C2": 19, "D": 18})
        r = validate_transition(before, after, pstate(), {"A": "A2", "B": "B2", "C": "C2"})
        self.assertFalse(r.accepted)

    def test_atomic_microstep_not_observed_final_state_only(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A": 12, "B": 13, "C": 15})
        r = validate_transition(before, after, pstate(), {"A": "A", "B": "B", "C": "C"})
        self.assertTrue(r.accepted)

    def test_release_after_last_accepted_transition(self):
        before = CandidateState({"A": 12, "B": 13, "C": 15})
        after = CandidateState({"A2": 16, "B2": 17, "C2": 19})
        r = validate_transition(before, after, pstate(1), {"A": "A2", "B": "B2", "C": "C2"})
        self.assertTrue(r.accepted)
        self.assertEqual(r.next_preservation.status, "released")
        self.assertEqual(r.next_preservation.remaining, 0)


if __name__ == "__main__":
    unittest.main()
