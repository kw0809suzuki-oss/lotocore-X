from __future__ import annotations

import json
import unittest
from pathlib import Path

from audit_examples import audit_recipe
from generator import run_recipe

HERE = Path(__file__).resolve().parent
EX = HERE / "examples"


class BreadthProbeTest(unittest.TestCase):
    def test_additional_examples_exact(self):
        expected = {
            "93_95_anchor30.json": [1, 8, 13, 14, 18, 30, 34],
            "269_271_anchor1_15.json": [1, 9, 15, 16, 19, 32, 35],
            "343_345_pair31_32.json": [6, 9, 10, 18, 28, 31, 32],
            "570_573_anchor26.json": [11, 12, 18, 19, 23, 26, 31],
        }
        for filename, final in expected.items():
            recipe = json.loads((EX / filename).read_text(encoding="utf-8"))
            result = run_recipe(recipe)
            self.assertEqual(result["final"], final, filename)

    def test_no_added_example_uses_full_seven_point_rewrite(self):
        for filename in [
            "93_95_anchor30.json",
            "269_271_anchor1_15.json",
            "343_345_pair31_32.json",
            "570_573_anchor26.json",
        ]:
            report = audit_recipe(EX / filename)
            self.assertLess(report["max_changed_points"], 7, filename)


if __name__ == "__main__":
    unittest.main()
