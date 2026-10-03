from __future__ import annotations

import unittest

from play_run_695 import generate


class PlayRun695Test(unittest.TestCase):
    def test_generates_twenty_valid_layouts_with_an_adjacent_pair(self):
        out = generate()
        self.assertEqual(out["candidate_count"], 20)
        self.assertEqual(len({tuple(x) for x in out["candidates"]}), 20)
        for layout in out["candidates"]:
            self.assertEqual(len(layout), 7)
            self.assertEqual(len(set(layout)), 7)
            self.assertTrue(all(1 <= n <= 37 for n in layout))
            self.assertTrue(any(b - a == 1 for a, b in zip(layout, layout[1:])))


if __name__ == "__main__":
    unittest.main()
