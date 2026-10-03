from __future__ import annotations

import unittest

from probe_future_blind_mapping import run_probe


class FutureBlindMappingProbeTest(unittest.TestCase):
    def test_future_is_not_used_to_generate_candidate_space(self):
        out = run_probe()
        self.assertFalse(out["constraint"]["candidate_generation_uses_future"])
        self.assertGreater(out["candidate_space"]["count"], 0)
        self.assertTrue(out["future_reveal"]["actual_subset_was_in_candidate_space"])
        self.assertTrue(out["validator_replay"]["accepted"])
        self.assertEqual(out["validator_replay"]["status"], "released")


if __name__ == "__main__":
    unittest.main()
