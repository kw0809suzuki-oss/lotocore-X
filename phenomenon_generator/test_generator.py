from __future__ import annotations

import json
import unittest
from pathlib import Path

from generator import GenerationError, run_recipe

HERE = Path(__file__).resolve().parent


def load_example(name: str):
    return json.loads((HERE / "examples" / name).read_text(encoding="utf-8"))


class PhenomenonGeneratorTest(unittest.TestCase):
    def test_324_326_exact_reproduction(self):
        result = run_recipe(load_example("324_326.json"))
        self.assertEqual(result["steps"][0]["after"], [4, 14, 17, 18, 21, 31, 33])
        self.assertEqual(result["final"], [1, 2, 17, 18, 24, 27, 29])

    def test_287_289_rereference_exact_reproduction(self):
        result = run_recipe(load_example("287_289.json"))
        self.assertEqual(result["steps"][0]["after"], [1, 8, 9, 16, 27, 32, 37])
        self.assertEqual(result["final"], [2, 7, 10, 12, 16, 17, 34])
        reref = result["steps"][1]["operations"][0]["operation"]
        self.assertEqual(reref["type"], "rereference")
        self.assertEqual(reref["source_index"], 0)

    def test_309_310_jump_exact_reproduction(self):
        result = run_recipe(load_example("309_310.json"))
        self.assertEqual(result["final"], [17, 25, 26, 28, 31, 32, 37])
        self.assertTrue(result["steps"][0]["preserve_checks"][0]["ok"])

    def test_conflict_is_rejected_without_silent_fix(self):
        recipe = {
            "initial": [1, 7, 13, 14, 16, 21, 22],
            "steps": [
                {
                    "preserve": [{"type": "width", "value": 21}],
                    "operations": [
                        {
                            "type": "reallocate",
                            "remove": [1],
                            "add": [2]
                        }
                    ]
                }
            ]
        }
        with self.assertRaises(GenerationError):
            run_recipe(recipe)


if __name__ == "__main__":
    unittest.main()
