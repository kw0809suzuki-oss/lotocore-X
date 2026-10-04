from __future__ import annotations

import unittest

from structure_lifetime import observe_partial_gap_lifetime


class StructureLifetimeTest(unittest.TestCase):
    def test_astra_41_45_gap_1_1_lifetime(self):
        rounds = [41, 42, 43, 44, 45]
        draws = [
            [2, 8, 14, 20, 34, 35, 36],
            [3, 9, 15, 16, 17, 29, 37],
            [1, 6, 7, 8, 18, 26, 35],
            [4, 11, 19, 27, 28, 29, 34],
            [2, 9, 14, 20, 25, 31, 37],
        ]

        life = observe_partial_gap_lifetime(rounds, draws, [1, 1])

        self.assertEqual(life.consecutive_presence, 4)
        self.assertEqual(life.break_round, 45)
        self.assertEqual(
            [o.present for o in life.observations],
            [True, True, True, True, False],
        )

    def test_absolute_position_is_not_required_to_persist(self):
        rounds = [1, 2]
        draws = [
            [1, 2, 3, 10, 20, 30, 37],
            [5, 6, 7, 11, 21, 31, 36],
        ]
        life = observe_partial_gap_lifetime(rounds, draws, [1, 1])

        self.assertEqual(life.consecutive_presence, 2)
        self.assertNotEqual(
            life.observations[0].matches,
            life.observations[1].matches,
        )


if __name__ == "__main__":
    unittest.main()
