from __future__ import annotations

import unittest

from probe_real_translated_gap import run_probe


class RealTranslatedGapProbeTest(unittest.TestCase):
    def test_real_local_translation_accepts(self):
        out = run_probe()
        self.assertEqual(out["before_subset"], [13,19,21,25])
        self.assertEqual(out["after_subset"], [6,12,14,18])
        self.assertEqual(out["gaps"], [6,2,4])
        self.assertTrue(out["accepted"])
        self.assertEqual(out["remaining"], 0)
        self.assertEqual(out["status"], "released")


if __name__ == "__main__":
    unittest.main()
