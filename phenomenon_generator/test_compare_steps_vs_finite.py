from __future__ import annotations

import unittest

from compare_steps_vs_finite import run_probe


class StepsVsFiniteProbeTest(unittest.TestCase):
    def test_same_committed_sequence_and_explicit_reject_difference(self):
        out = run_probe()
        finite = out["finite_preservation_observation"]

        self.assertTrue(finite["transition_324_325"]["accepted"])
        self.assertEqual(finite["transition_324_325"]["remaining"], 1)
        self.assertEqual(finite["transition_324_325"]["status"], "active")

        self.assertTrue(finite["transition_325_326"]["accepted"])
        self.assertEqual(finite["transition_325_326"]["remaining"], 0)
        self.assertEqual(finite["transition_325_326"]["status"], "released")

        self.assertFalse(finite["counterfactual_325_bad326"]["accepted"])
        self.assertEqual(finite["counterfactual_325_bad326"]["remaining"], 1)
        self.assertEqual(finite["counterfactual_325_bad326"]["status"], "active")


if __name__ == "__main__":
    unittest.main()
