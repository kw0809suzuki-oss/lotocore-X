from __future__ import annotations

import unittest

from probe_width_jump_boundary import run_probe


class WidthJumpBoundaryProbeTest(unittest.TestCase):
    def test_309_310_is_rejected_by_exact_gap_v0_1(self):
        out = run_probe()
        self.assertEqual(out["before_width"], 21)
        self.assertEqual(out["after_width"], 20)
        self.assertTrue(out["explicit_rank_mapping"])
        self.assertFalse(out["finite_preservation_v0_1"]["accepted"])
        self.assertIn("gap mismatch", out["finite_preservation_v0_1"]["reason"])
        self.assertEqual(out["finite_preservation_v0_1"]["remaining"], 1)
        self.assertEqual(out["finite_preservation_v0_1"]["status"], "active")


if __name__ == "__main__":
    unittest.main()
