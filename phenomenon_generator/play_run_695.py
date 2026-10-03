from __future__ import annotations

import json
import random

BASE = [5, 6, 11, 16, 22, 24, 34]


def generate(seed: int = 695, count: int = 20) -> dict:
    rng = random.Random(seed)

    # Play-run choice: preserve only the local relation [5,6] as an adjacent pair (gap=1).
    # The pair may translate anywhere in 1..37. The other five points are freely reallocated.
    pair_candidates = [(n, n + 1) for n in range(1, 37)]
    layouts = []
    seen = set()

    attempts = 0
    while len(layouts) < count and attempts < 10000:
        attempts += 1
        a, b = rng.choice(pair_candidates)
        remaining_pool = [n for n in range(1, 38) if n not in {a, b}]
        rest = rng.sample(remaining_pool, 5)
        layout = tuple(sorted([a, b, *rest]))
        if layout in seen:
            continue
        seen.add(layout)
        layouts.append(list(layout))

    return {
        "input_round": 695,
        "input_layout": BASE,
        "preserve": {
            "source_members": [5, 6],
            "relation": {"consecutive": True, "gaps": [1]},
            "duration": 1,
        },
        "generation_mode": "future-blind play run",
        "selection_note": (
            "This run intentionally fixes one observed local relation only. "
            "The other five points are reallocated without predictive scoring."
        ),
        "candidate_count": len(layouts),
        "candidates": layouts,
        "boundary": (
            "Generated placements are not predictions. This run only demonstrates "
            "that the preservation mechanism can produce concrete seven-number layouts."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
