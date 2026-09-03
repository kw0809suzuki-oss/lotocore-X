from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

NUMBERS = range(1, 38)


@dataclass
class Prediction:
    numbers: tuple[int, ...]
    state: dict


def _draws(history: pd.DataFrame) -> list[list[int]]:
    cols = [f"n{i}" for i in range(1, 8)]
    return [sorted(map(int, row)) for row in history[cols].to_numpy()]


def predict(history: pd.DataFrame) -> Prediction:
    """Conservative Loto Core proxy.

    Core idea: keep a comparatively stable structure from frequency, recency and
    positional center. This is intentionally less reactive than X.
    """
    draws = _draws(history)
    if len(draws) < 10:
        raise ValueError("Loto Core requires at least 10 historical draws")

    long = draws[-100:]
    recent = draws[-20:]
    freq_long = Counter(n for d in long for n in d)
    freq_recent = Counter(n for d in recent for n in d)

    # Last-seen gap: moderate preference for numbers not seen very recently.
    gap = {n: len(draws) for n in NUMBERS}
    for g, d in enumerate(reversed(draws), start=0):
        for n in d:
            if gap[n] == len(draws):
                gap[n] = g

    # Keep the model stable: long frequency dominates; recent frequency and gap
    # only adjust it rather than redefining it every draw.
    scores = {}
    for n in NUMBERS:
        scores[n] = (
            0.60 * freq_long[n] / max(1, len(long))
            + 0.25 * freq_recent[n] / max(1, len(recent))
            + 0.15 * min(gap[n], 12) / 12.0
        )

    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], n))
    chosen = tuple(sorted(ranked[:7]))
    center = sum(chosen) / 7.0
    spread = sum((n - center) ** 2 for n in chosen) / 7.0
    return Prediction(
        chosen,
        {
            "model": "lotocore",
            "center": round(center, 4),
            "variance": round(spread, 4),
            "top_score": round(scores[ranked[0]], 6),
        },
    )
