from __future__ import annotations

import unittest

from compare_flow_astra_random import (
    detect_flow_persistence,
    detect_flow_return,
    detect_flow_width_jump,
)


class FlowAstraRandomTest(unittest.TestCase):
    def test_return_after_absence(self):
        d = detect_flow_return(
            [2,7,10,20,24,29,34],
            [1,8,9,16,27,32,37],
            [2,7,10,12,17,20,34],
        )
        self.assertIsNotNone(d)
        self.assertEqual(d.carriers, (2,7,10,20,34))

    def test_point_pair_persistence(self):
        d = detect_flow_persistence(
            [9,13,16,17,18,19,20],
            [4,14,17,18,21,31,33],
            [1,2,17,18,24,27,29],
        )
        self.assertIsNotNone(d)
        self.assertEqual(d.carriers, (17,18))

    def test_width_jump(self):
        d = detect_flow_width_jump(
            [1,7,13,14,16,21,22],
            [17,25,26,28,31,32,37],
        )
        self.assertIsNotNone(d)


if __name__ == "__main__":
    unittest.main()
