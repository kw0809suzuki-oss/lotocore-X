from __future__ import annotations

import unittest

from probe_consecutive_future_blind import run_probe


class ConsecutiveFutureBlindProbeTest(unittest.TestCase):
    def test_324_325_candidate_set_is_frozen_before_reveal(self):
        out = run_probe()
        self.assertFalse(out["candidate_generation"]["uses_draw325"])
        self.assertEqual(out["candidate_generation"]["candidate_count_before_reveal"], 36)
        self.assertIn([17, 18], out["future_reveal"]["realized_candidates"])
        self.assertTrue(out["validator"]["accepted"])
        self.assertEqual(out["validator"]["status"], "released")


if __name__ == "__main__":
    unittest.main()
