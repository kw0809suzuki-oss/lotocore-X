from __future__ import annotations

import unittest

from blank_lifecycle import (
    BlankLifecycleError,
    observe_blank_then_reentry,
)


class BlankLifecycleTest(unittest.TestCase):
    def test_astra_472_473_blank_reentry_and_33_persistence(self):
        # Minimal fixtures preserve only the reported facts required for S4.
        r472 = [1, 4, 7, 8, 32, 33, 37]
        r473 = [2, 11, 21, 24, 28, 33, 36]

        obs = observe_blank_then_reentry(
            blank_layout=r472,
            reentry_layout=r473,
            lo=9,
            hi=31,
            persistent_candidates=[33],
        )

        self.assertTrue(obs.blank_state.is_blank)
        self.assertEqual(obs.reentry_numbers, (11, 21, 24, 28))
        self.assertEqual(obs.persistent_numbers, (33,))

    def test_blank_and_persistence_are_separate_components(self):
        r472 = [1, 4, 7, 8, 32, 33, 37]
        r473 = [2, 11, 21, 24, 28, 33, 36]

        obs = observe_blank_then_reentry(r472, r473, 9, 31, [33])
        self.assertNotIn(33, obs.reentry_numbers)
        self.assertIn(33, obs.persistent_numbers)

    def test_reject_when_first_state_not_blank(self):
        with self.assertRaises(BlankLifecycleError):
            observe_blank_then_reentry(
                [1, 10, 33, 34, 35, 36, 37],
                [2, 11, 21, 24, 28, 33, 36],
                9,
                31,
                [33],
            )


if __name__ == "__main__":
    unittest.main()
