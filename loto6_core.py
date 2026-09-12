from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import pandas as pd

NUMBERS = range(1, 44)
DRAW_SIZE = 6
STATE_WINDOW = 20
ANALOGS = 12


@dataclass
class Prediction:
    numbers: tuple[int, ...]
    state: dict


def _draws(history: pd.DataFrame) -> list[list[int]]:
    cols = [f"n{i}" for i in range(1, DRAW_SIZE + 1)]
    return [sorted(map(int, row)) for row in history[cols].to_numpy()]


def _draw_center(draw: list[int]) -> float:
    return sum(draw) / DRAW_SIZE


def _draw_spread(draw: list[int]) -> float:
    center = _draw_center(draw)
    return sqrt(sum((n - center) ** 2 for n in draw) / DRAW_SIZE)


def _macro_state(draws: list[list[int]]) -> tuple[float, float]:
    """Observe one coarse LOTO6 state from a block of draws.

    center: where the block sits overall.
    amplitude: how widely each draw is spread on average.

    The Core intentionally stays coarse. It does not add odd/even, adjacency,
    zones, sums, individual gaps, or other local features.
    """
    centers = [_draw_center(d) for d in draws]
    spreads = [_draw_spread(d) for d in draws]
    return sum(centers) / len(centers), sum(spreads) / len(spreads)


def predict(history: pd.DataFrame) -> Prediction:
    """LOTO6 Core: macro state -> similar past states -> candidate set.

    The current state is observed first from the latest 20 draws. Earlier
    20-draw states inside the available history are then compared with it.
    Numbers from the draws that followed the most similar states form the
    candidate ranking.
    """
    draws = _draws(history)
    if len(draws) < STATE_WINDOW * 2 + 1:
        raise ValueError("LOTO6 Core requires at least 41 historical draws")

    current_center, current_amplitude = _macro_state(draws[-STATE_WINDOW:])

    # Build past state -> next draw examples. The latest current block itself
    # is never used as a labelled example because its next draw is unknown.
    examples: list[tuple[float, int, list[int], float, float]] = []
    for end in range(STATE_WINDOW, len(draws)):
        block = draws[end - STATE_WINDOW : end]
        center, amplitude = _macro_state(block)

        # Coarse normalized distance. The fixed scales only put center and
        # amplitude on comparable units; they are not fitted to outcomes.
        distance = sqrt(
            ((center - current_center) / 5.0) ** 2
            + ((amplitude - current_amplitude) / 3.0) ** 2
        )
        examples.append((distance, end, draws[end], center, amplitude))

    examples.sort(key=lambda x: (x[0], -x[1]))
    nearest = examples[: min(ANALOGS, len(examples))]

    candidate_counts = Counter(n for _, _, next_draw, _, _ in nearest for n in next_draw)
    long_counts = Counter(n for d in draws[-100:] for n in d)

    # Similar-state follow-up frequency is the Core signal. Long history is
    # used only as a deterministic tie-breaker, not as a second model.
    ranked = sorted(
        NUMBERS,
        key=lambda n: (-candidate_counts[n], -long_counts[n], n),
    )
    chosen = tuple(sorted(ranked[:DRAW_SIZE]))

    return Prediction(
        chosen,
        {
            "model": "loto6_core_macro_state",
            "state_center": round(current_center, 4),
            "state_amplitude": round(current_amplitude, 4),
            "analogs": len(nearest),
            "nearest_distance": round(nearest[0][0], 4) if nearest else None,
            "ranked_top12": "-".join(map(str, ranked[:12])),
        },
    )
