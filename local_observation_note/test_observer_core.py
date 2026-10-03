from __future__ import annotations

import unittest
from io import StringIO

import pandas as pd

from observer_core import interval_frame, point_frame, validate_and_normalize


class ObserverCoreTest(unittest.TestCase):
    def test_normalize_sorts_without_bonus(self):
        df = pd.read_csv(
            StringIO(
                "round,n1,n2,n3,n4,n5,n6,n7,b1,b2\n"
                "2,35,20,34,24,29,31,33,12,32\n"
                "1,34,7,28,10,23,12,17,3,15\n"
            )
        )
        normalized, errors = validate_and_normalize(df)
        self.assertEqual(errors, [])
        self.assertEqual(normalized["round"].tolist(), [1, 2])
        self.assertEqual(
            normalized.loc[0, [f"n{i}" for i in range(1, 8)]].tolist(),
            [7, 10, 12, 17, 23, 28, 34],
        )

    def test_detects_gap_duplicate_and_out_of_range(self):
        df = pd.read_csv(
            StringIO(
                "round,n1,n2,n3,n4,n5,n6,n7\n"
                "1,1,2,3,4,5,6,7\n"
                "3,1,1,3,4,5,6,38\n"
            )
        )
        _, errors = validate_and_normalize(df)
        joined = "\n".join(errors)
        self.assertIn("欠回", joined)
        self.assertIn("重複", joined)
        self.assertIn("範囲外", joined)

    def test_interval_and_points_preserve_time_and_positions(self):
        df = pd.DataFrame(
            [
                {"round": 1, "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "n7": 7},
                {"round": 2, "n1": 8, "n2": 9, "n3": 10, "n4": 11, "n5": 12, "n6": 13, "n7": 14},
            ]
        )
        selected = interval_frame(df, 2, 1)
        points = point_frame(selected)
        self.assertEqual(selected["round"].tolist(), [1, 2])
        self.assertEqual(len(points), 14)
        self.assertEqual(points.iloc[0].to_dict(), {"round": 1, "number": 1})


if __name__ == "__main__":
    unittest.main()
