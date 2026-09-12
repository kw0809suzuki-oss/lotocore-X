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
    """Conservative LOTO6 Core baseline.

    Stable frequency/recency/gap mixture. This is intentionally less reactive
    than Dynamic X and does not use X Field-regime reconstruction.
    """
    draws = _draws(history)
    if len(draws) < 20:
        raise ValueError("LOTO6 Core requires at least 20 historical draws")

    long = draws[-100:]
    recent = draws[-20:]
    freq_long = Counter(n for d in long for n in d)
    freq_recent = Counter(n for d in recent for n in d)

    gap = {n: len(draws) for n in NUMBERS}
    for g, draw in enumerate(reversed(draws)):
        for n in draw:
            if gap[n] == len(draws):
                gap[n] = g

    scores: dict[int, float] = {}
    for n in NUMBERS:
        scores[n] = (
            0.60 * freq_long[n] / max(1, len(long))
            + 0.25 * freq_recent[n] / max(1, len(recent))
            + 0.15 * min(gap[n], 14) / 14.0
        )

    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], n))
    chosen = tuple(sorted(ranked[:DRAW_SIZE]))
    center = sum(chosen) / DRAW_SIZE
    variance = sum((n - center) ** 2 for n in chosen) / DRAW_SIZE
    return Prediction(
        chosen,
        {
            "model": "loto6_core",
            "center": round(center, 4),
            "variance": round(variance, 4),
            "ranked_top12": "-".join(map(str, ranked[:12])),
        },
    )
