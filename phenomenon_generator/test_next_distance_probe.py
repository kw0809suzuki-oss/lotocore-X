from __future__ import annotations

import unittest

from next_distance_probe import (
    detect_s1,
    detect_s2,
    detect_s3,
    detect_s4,
    detect_s5,
    distance_metrics,
)


class NextDistanceProbeTest(unittest.TestCase):
    def test_s1_exact_signature(self):
        self.assertIsNotNone(
            detect_s1(
                [1, 2, 3, 10, 20, 30, 37],
                [5, 6, 7, 11, 21, 31, 36],
            )
        )

    def test_s2_exact_signature(self):
        d = detect_s2(
            [2, 28, 30, 31, 32, 33, 34],
            [30, 31, 32, 33, 34, 35, 36],
            [2, 28, 30, 31, 32, 33, 37],
        )
        self.assertIsNotNone(d)
        self.assertIn(2, d.carriers)
        self.assertIn(30, d.carriers)

    def test_s3_exact_signature(self):
        d = detect_s3(
            [1, 10, 24, 25, 26, 30, 37],
            [2, 20, 23, 24, 29, 31, 36],
            [3, 21, 24, 28, 32, 34, 35],
        )
        self.assertIsNotNone(d)
        self.assertIn(24, d.carriers)

    def test_s4_exact_signature(self):
        d = detect_s4(
            [1, 12, 20, 25, 32, 33, 37],
            [1, 4, 7, 8, 32, 33, 37],
            [2, 11, 21, 24, 28, 33, 36],
        )
        self.assertIsNotNone(d)
        self.assertIn(33, d.carriers)

    def test_s5_exact_signature(self):
        d = detect_s5(
            [7, 9, 15, 18, 20, 28, 31],
            [16, 17, 22, 23, 25, 33, 35],
        )
        self.assertIsNotNone(d)
        self.assertEqual(d.carriers, (23, 25, 33))

    def test_distance_direction(self):
        near = distance_metrics([10, 20], [9, 10, 11, 18, 20, 21, 30])
        far = distance_metrics([1, 37], [9, 10, 11, 18, 20, 21, 30])
        self.assertLess(near["mean_min_distance"], far["mean_min_distance"])


if __name__ == "__main__":
    unittest.main()
