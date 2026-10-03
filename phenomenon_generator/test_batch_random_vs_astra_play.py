from __future__ import annotations

import unittest

from batch_random_vs_astra_play import SEEDS


class BatchPlayTest(unittest.TestCase):
    def test_has_six_fresh_seeds(self):
        self.assertEqual(len(SEEDS), 6)
        self.assertEqual(len(set(SEEDS)), 6)


if __name__ == "__main__":
    unittest.main()
