from __future__ import annotations

import unittest

from point_context import context_changed, observe_point_context


class PointContextTest(unittest.TestCase):
    def test_astra_694_696_point_24(self):
        r694 = [3, 13, 24, 26, 30, 31, 36]
        r695 = [5, 6, 11, 16, 22, 24, 34]
        r696 = [1, 2, 9, 19, 23, 24, 34]

        c694 = observe_point_context(r694, 24)
        c695 = observe_point_context(r695, 24)
        c696 = observe_point_context(r696, 24)

        self.assertEqual((c694.left_neighbor, c694.right_neighbor), (13, 26))
        self.assertEqual((c694.left_gap, c694.right_gap), (11, 2))

        self.assertEqual((c695.left_neighbor, c695.right_neighbor), (22, 34))
        self.assertEqual((c695.left_gap, c695.right_gap), (2, 10))

        self.assertEqual((c696.left_neighbor, c696.right_neighbor), (23, 34))
        self.assertEqual((c696.left_gap, c696.right_gap), (1, 10))

        self.assertTrue(c694.present)
        self.assertTrue(c695.present)
        self.assertTrue(c696.present)
        self.assertTrue(context_changed(c694, c695))
        self.assertTrue(context_changed(c695, c696))

    def test_absent_member_has_no_context(self):
        c = observe_point_context([1, 2, 3, 4, 5, 6, 7], 24)
        self.assertFalse(c.present)
        self.assertIsNone(c.left_neighbor)
        self.assertIsNone(c.right_neighbor)


if __name__ == "__main__":
    unittest.main()
