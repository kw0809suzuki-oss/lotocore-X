from __future__ import annotations

import unittest

from temporal_paths import (
    TemporalPathError,
    match_required_paths,
    observe_paths,
)


class TemporalPathTest(unittest.TestCase):
    def test_astra_round_8_10_paths(self):
        # Minimal fixture preserves only the observed membership facts needed
        # for Astra S2. Other draw members are intentionally irrelevant here.
        obs = observe_paths(
            rounds=[8, 9, 10],
            draws=[
                {2, 28, 30},
                {30},
                {2, 28, 30},
            ],
            members=[2, 28, 30],
        )

        self.assertEqual(
            obs.path_types(),
            (
                "present_absent_present",
                "present_absent_present",
                "present_present_present",
            ),
        )
        self.assertTrue(obs.has_coexisting_path_types())
        self.assertTrue(
            match_required_paths(
                obs,
                {
                    2: [True, False, True],
                    28: [True, False, True],
                    30: [True, True, True],
                },
            )
        )

    def test_order_matters(self):
        obs = observe_paths(
            rounds=[8, 9, 10],
            draws=[{2}, set(), {2}],
            members=[2],
        )
        self.assertFalse(match_required_paths(obs, {2: [True, True, False]}))

    def test_path_length_matches_round_window(self):
        with self.assertRaises(TemporalPathError):
            observe_paths(
                rounds=[8, 9, 10],
                draws=[{2}, set()],
                members=[2],
            )


if __name__ == "__main__":
    unittest.main()
