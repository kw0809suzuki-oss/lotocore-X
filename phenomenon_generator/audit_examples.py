from __future__ import annotations

import json
from pathlib import Path

from generator import run_recipe

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"


def changed_count(before: list[int], after: list[int]) -> int:
    return 7 - len(set(before) & set(after))


def audit_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    result = run_recipe(recipe)
    rows = []
    for step in result["steps"]:
        before = step["before"]
        after = step["after"]
        rows.append(
            {
                "label": step["label"],
                "changed_points": changed_count(before, after),
                "preserved_points": len(set(before) & set(after)),
                "operation_count": len(step["operations"]),
            }
        )
    return {
        "name": result["name"],
        "step_count": len(rows),
        "steps": rows,
        "max_changed_points": max((r["changed_points"] for r in rows), default=0),
        "mean_changed_points": (
            sum(r["changed_points"] for r in rows) / len(rows) if rows else 0.0
        ),
    }


def main() -> None:
    paths = sorted(EXAMPLES.glob("*.json"))
    reports = [audit_recipe(p) for p in paths]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
