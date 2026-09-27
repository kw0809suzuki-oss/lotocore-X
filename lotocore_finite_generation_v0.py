from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import pandas as pd

import lotocore


BOUNDARY_K = 6


@dataclass
class Prediction:
    numbers: tuple[int, ...]
    state: dict


def _ranked(snapshot: dict) -> list[int]:
    ranks = {int(n): int(r) for n, r in snapshot["ranks"].items()}
    return sorted(ranks, key=lambda n: (ranks[n], n))


def _weighted_state(scores: dict) -> tuple[float, float]:
    vals = {int(n): max(0.0, float(v)) for n, v in scores.items()}
    total = sum(vals.values())
    if total <= 0:
        p = {n: 1.0 / 37.0 for n in range(1, 38)}
    else:
        p = {n: vals.get(n, 0.0) / total for n in range(1, 38)}
    center = sum(n * p[n] for n in p)
    variance = sum((n - center) ** 2 * p[n] for n in p)
    return center, math.sqrt(variance)


def _set_state(nums: tuple[int, ...]) -> tuple[float, float]:
    vals = list(nums)
    center = sum(vals) / len(vals)
    variance = sum((n - center) ** 2 for n in vals) / len(vals)
    return center, math.sqrt(variance)


def predict(history: pd.DataFrame) -> Prediction:
    """Finite-generation Core v0.

    Keep current Loto Core unchanged as the source model.
    Treat its 7 numbers as Core, ranks 8..13 outside the Core as Boundary,
    then regenerate one final 7-number set from that finite representation.

    Selection uses only current model state:
    1) minimize center+spread error to the full score field,
    2) retain score mass as tie-break.
    No target draw is used.
    """
    base = lotocore.predict(history)
    snap = lotocore.score_snapshot(history)

    core = tuple(sorted(base.numbers))
    core_set = set(core)
    ranked = _ranked(snap)
    boundary = tuple(n for n in ranked if n not in core_set)[:BOUNDARY_K]
    support = tuple(sorted(core_set | set(boundary)))

    scores = {int(n): float(v) for n, v in snap["scores"].items()}
    target_center, target_spread = _weighted_state(snap["scores"])

    best = None
    for combo in combinations(support, 7):
        combo = tuple(sorted(combo))
        center, spread = _set_state(combo)
        center_error = abs(center - target_center)
        spread_error = abs(spread - target_spread)
        shape_error = (center_error + spread_error) / max(target_spread, 1e-12)
        score_mass = sum(scores[n] for n in combo)

        key = (shape_error, -score_mass, combo)
        if best is None or key < best[0]:
            best = (
                key,
                combo,
                center,
                spread,
                center_error,
                spread_error,
                score_mass,
            )

    _, chosen, center, spread, center_error, spread_error, score_mass = best
    return Prediction(
        chosen,
        {
            "model": "lotocore_finite_generation_v0",
            "base_core7": list(core),
            "boundary6": list(boundary),
            "support13": list(support),
            "base_overlap": len(set(core) & set(chosen)),
            "target_center": round(target_center, 6),
            "target_spread": round(target_spread, 6),
            "center": round(center, 6),
            "spread": round(spread, 6),
            "center_error": round(center_error, 6),
            "spread_error": round(spread_error, 6),
            "score_mass": round(score_mass, 6),
        },
    )
