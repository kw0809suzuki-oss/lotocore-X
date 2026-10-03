from __future__ import annotations

import unittest
import pandas as pd

from compare_random_vs_astra_play import run


class RandomVsAstraPlayTest(unittest.TestCase):
    def test_runs_on_small_fixture(self):
        rows = []
        for r in range(1, 8):
            vals = list(range(r, r + 7))
            vals = [((x - 1) % 37) + 1 for x in vals]
            rows.append({"round": r, **{f"n{i+1}": vals[i] for i in range(7)}})
        df = pd.DataFrame(rows)
        out = run(df, last_n=5, seed=1)
        self.assertEqual(out["window"]["rounds"], 5)
        self.assertEqual(out["random20"]["rounds"], 5)
        self.assertEqual(out["astra20"]["rounds"], 5)


if __name__ == "__main__":
    unittest.main()
