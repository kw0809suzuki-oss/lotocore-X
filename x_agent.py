from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

NUMBERS = range(1, 38)


@dataclass
class Prediction:
    numbers: tuple[int, ...]
    state: dict


def _draws(history: pd.DataFrame) -> list[list[int]]:
    cols = [f"n{i}" for i in range(1, 8)]
    return [sorted(map(int, row)) for row in history[cols].to_numpy()]


def _mean(draw):
    return sum(draw) / len(draw)


def _variance(draw):
    m = _mean(draw)
    return sum((x - m) ** 2 for x in draw) / len(draw)


def _outer_share(draws):
    vals = [n for d in draws for n in d]
    if not vals:
        return 0.0
    return sum(1 for n in vals if n <= 12 or n >= 26) / len(vals)


def _mean_gap(draw):
    if len(draw) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(draw, draw[1:])]
    return sum(gaps) / len(gaps)


def predict(history: pd.DataFrame, competition_gate: bool = True) -> Prediction:
    """Dynamic Structure X for a non-intervenable Field.

    Field -> Relation -> Weight -> Flow.  Shape and gap geometry relations are
    provisional and should survive only if walk-forward performance improves.
    """
    draws = _draws(history)
    if len(draws) < 20:
        raise ValueError("X requires at least 20 historical draws")

    recent = draws[-12:]
    prev = draws[-24:-12] if len(draws) >= 24 else draws[:-12]

    f_recent = Counter(n for d in recent for n in d)
    f_prev = Counter(n for d in prev for n in d)

    centers = [_mean(d) for d in recent]
    spreads = [_variance(d) for d in recent]
    center_now = sum(centers[-4:]) / 4.0
    center_before = sum(centers[:4]) / 4.0
    center_delta = center_now - center_before
    spread_now = sum(spreads[-4:]) / 4.0
    spread_before = sum(spreads[:4]) / 4.0
    spread_delta = spread_now - spread_before

    volatility = min(1.0, (abs(center_delta) / 5.0 + abs(spread_delta) / 35.0) / 2.0)
    w_persist = 0.50 * (1.0 - volatility) + 0.15
    w_reverse = 0.20 + 0.45 * volatility
    w_gap = 1.0 - w_persist - w_reverse
    if w_gap < 0.10:
        w_gap = 0.10
        total = w_persist + w_reverse + w_gap
        w_persist, w_reverse, w_gap = [x / total for x in (w_persist, w_reverse, w_gap)]

    # Relation 4: distribution shape (edge-heavy vs center-heavy).
    outer_recent = _outer_share(recent[-4:])
    outer_before = _outer_share(recent[:4])
    shape_delta = outer_recent - outer_before
    w_shape = 0.12 + 0.18 * min(1.0, abs(shape_delta) / 0.35)

    # Relation 5: gap geometry. Read whether consecutive-number spacing is
    # becoming wider or tighter, and softly prefer candidates compatible with it.
    gap_recent = sum(_mean_gap(d) for d in recent[-4:]) / 4.0
    gap_before = sum(_mean_gap(d) for d in recent[:4]) / 4.0
    gap_delta = gap_recent - gap_before
    w_geom = 0.08 + 0.12 * min(1.0, abs(gap_delta) / 2.5)

    # Competition Gate: shape and geometry currently share the same
    # outer/center axis. If they point the same way, geometry adds no new
    # information and is suppressed; when they conflict it remains observable.
    relation_conflict = (shape_delta >= 0) != (gap_delta >= 0)
    geom_active = (not competition_gate) or relation_conflict
    if not geom_active:
        w_geom = 0.0

    base_scale = 1.0 - w_shape - w_geom
    if base_scale < 0.55:
        base_scale = 0.55
        total_extra = w_shape + w_geom
        if total_extra > 0:
            extra_scale = (1.0 - base_scale) / total_extra
            w_shape *= extra_scale
            w_geom *= extra_scale
    w_persist *= base_scale
    w_reverse *= base_scale
    w_gap *= base_scale

    gap = {n: len(draws) for n in NUMBERS}
    for g, d in enumerate(reversed(draws)):
        for n in d:
            if gap[n] == len(draws):
                gap[n] = g

    scores = {}
    for n in NUMBERS:
        persist = f_recent[n] / max(1, len(recent))
        trend = persist - (f_prev[n] / max(1, len(prev)))
        reversal = max(0.0, -trend)
        gap_pressure = min(gap[n], 16) / 16.0

        is_outer = 1.0 if (n <= 12 or n >= 26) else 0.0
        shape = is_outer if shape_delta >= 0 else 1.0 - is_outer

        # Geometry score: wider spacing favors distance from the center; tighter
        # spacing favors the central band. This is deliberately soft.
        dist_center = abs(n - 19.0) / 18.0
        geometry = dist_center if gap_delta >= 0 else 1.0 - dist_center

        scores[n] = (
            w_persist * persist
            + w_reverse * reversal
            + w_gap * gap_pressure
            + w_shape * shape
            + w_geom * geometry
        )

    target_center = 19.0 + max(-3.0, min(3.0, center_delta * 0.35))
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], abs(n - target_center), n))

    chosen = []
    target_gap = max(2.5, min(7.5, 5.0 + gap_delta * 0.35))
    for n in ranked:
        if not chosen:
            chosen.append(n)
            continue
        if len(chosen) >= 7:
            break
        trial = sorted(chosen + [n])
        mean_gap = _mean_gap(trial)
        spread_ok = len(trial) <= 3 or (max(trial) - min(trial)) >= 3 * (len(trial) - 1)
        geom_ok = (not geom_active) or len(trial) <= 3 or abs(mean_gap - target_gap) <= 2.75
        if spread_ok and geom_ok:
            chosen.append(n)
    for n in ranked:
        if len(chosen) >= 7:
            break
        if n not in chosen:
            chosen.append(n)

    chosen = tuple(sorted(chosen[:7]))
    center = _mean(chosen)
    spread = _variance(chosen)
    return Prediction(
        chosen,
        {
            "model": "x",
            "center": round(center, 4),
            "variance": round(spread, 4),
            "field_center_delta": round(center_delta, 4),
            "field_spread_delta": round(spread_delta, 4),
            "shape_delta": round(shape_delta, 4),
            "gap_delta": round(gap_delta, 4),
            "target_gap": round(target_gap, 4),
            "volatility": round(volatility, 4),
            "w_persist": round(w_persist, 4),
            "w_reverse": round(w_reverse, 4),
            "w_gap": round(w_gap, 4),
            "w_shape": round(w_shape, 4),
            "w_geom": round(w_geom, 4),
            "geom_active": int(geom_active),
            "competition_gate": int(competition_gate),
        },
    )
