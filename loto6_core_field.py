from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

NUMBERS = range(1, 44)
DRAW_SIZE = 6


@dataclass
class Prediction:
    numbers: tuple[int, ...]
    state: dict


def _draws(history: pd.DataFrame) -> list[list[int]]:
    cols = [f"n{i}" for i in range(1, DRAW_SIZE + 1)]
    return [sorted(map(int, row)) for row in history[cols].to_numpy()]


def predict(history: pd.DataFrame) -> Prediction:
    """LOTO6 Core without similar-state matching.

    Treat the latest 100 draws as one candidate field. Each number's density in
    that field is the only signal. No analog states, recent weighting, gap
    weighting, odd/even, adjacency, zones, or sum rules are used.
    """
    draws = _draws(history)
    if len(draws) < 20:
        raise ValueError("LOTO6 Field Core requires at least 20 historical draws")

    field = draws[-100:]
    counts = Counter(n for draw in field for n in draw)
    ranked = sorted(NUMBERS, key=lambda n: (-counts[n], n))
    chosen = tuple(sorted(ranked[:DRAW_SIZE]))

    all_values = [n for draw in field for n in draw]
    field_center = sum(all_values) / len(all_values)

    return Prediction(
        chosen,
        {
            "model": "loto6_core_field",
            "field_draws": len(field),
            "field_center": round(field_center, 4),
            "ranked_top12": "-".join(map(str, ranked[:12])),
        },
    )
