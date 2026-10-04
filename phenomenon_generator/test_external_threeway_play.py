from __future__ import annotations

import random
import unittest

from external_threeway_play import _ticket_from_carriers


class ExternalThreewayPlayTest(unittest.TestCase):
    def test_ticket_keeps_carriers(self):
        t = _ticket_from_carriers([17, 18], random.Random(1))
        self.assertEqual(len(t), 7)
        self.assertEqual(len(set(t)), 7)
        self.assertIn(17, t)
        self.assertIn(18, t)

    def test_full_seven_carriers_are_ticket(self):
        c = [1, 7, 13, 14, 16, 21, 22]
        self.assertEqual(_ticket_from_carriers(c, random.Random(1)), c)


if __name__ == "__main__":
    unittest.main()
