from __future__ import annotations

import unittest

from partial_gap_scope import (
    find_partial_gap_matches,
    translated_subset,
)


class PartialGapScopeTest(unittest.TestCase):
    def test_astra_692_693_subset(self):
        r692 = [7, 9, 15, 18, 20, 28, 31]
        r693 = [16, 17, 22, 23, 25, 33, 35]

        m692 = find_partial_gap_matches(r692, [2, 8])
        m693 = find_partial_gap_matches(r693, [2, 8])

        self.assertIn((18, 20, 28), [m.members for m in m692])
        self.assertIn((23, 25, 33), [m.members for m in m693])

        ok, delta = translated_subset([18, 20, 28], [23, 25, 33])
        self.assertTrue(ok)
        self.assertEqual(delta, 5)

    def test_scope_is_partial_not_whole_layout(self):
        r692 = [7, 9, 15, 18, 20, 28, 31]
        self.assertNotEqual(
            [b - a for a, b in zip(r692, r692[1:])],
            [2, 8],
        )

    def test_non_uniform_translation_rejected(self):
        ok, delta = translated_subset([1, 3, 7], [6, 8, 13])
        self.assertFalse(ok)
        self.assertIsNone(delta)


if __name__ == "__main__":
    unittest.main()
