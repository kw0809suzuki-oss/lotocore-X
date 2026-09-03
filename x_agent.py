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


def predict(history: pd.DataFrame) -> Prediction:
    """Dynamic Structure X for a non-intervenable Field.

    X does not assume one stable winning structure. It reads the current Field,
    generates competing Relations, weights them from recent change, and rebuilds
    the 7-number Flow on every step.

    This is a research model, not a claim that Loto7 is predictable.
    """
    draws = _draws(history)
    if len(draws) < 20:
        raise ValueError("X requires at least 20 historical draws")

    recent = draws[-12:]
    mid = draws[-36:]
    prev = draws[-24:-12] if len(draws) >= 24 else draws[:-12]

    f_recent = Counter(n for d in recent for n in d)
    f_mid = Counter(n for d in mid for n in d)
    f_prev = Counter(n for d in prev for n in d)

    centers = [_mean(d) for d in recent]
    spreads = [_variance(d) for d in recent]
    center_now = sum(centers[-4:]) / 4.0
    center_before = sum(centers[:4]) / 4.0
    center_delta = center_now - center_before
    spread_now = sum(spreads[-4:]) / 4.0
    spread_before = sum(spreads[:4]) / 4.0
    spread_delta = spread_now - spread_before

    # Three Relations: persistence, reversal, and gap pressure.
    # Their Weight changes with the current Field rather than staying fixed.
    volatility = min(1.0, (abs(center_delta) / 5.0 + abs(spread_delta) / 35.0) / 2.0)
    w_persist = 0.50 * (1.0 - volatility) + 0.15
    w_reverse = 0.20 + 0.45 * volatility
    w_gap = 1.0 - w_persist - w_reverse
    if w_gap < 0.10:
        w_gap = 0.10
        total = w_persist + w_reverse + w_gap
        w_persist, w_reverse, w_gap = [x / total for x in (w_persist, w_reverse, w_gap)]

    gap = {n: len(draws) for n in NUMBERS}
    for g, d in enumerate(reversed(draws)):
        for n in d:
            if gap[n] == len(draws):
                gap[n] = g

    scores = {}
    for n in NUMBERS:
        persist = f_recent[n] / max(1, len(recent))
        trend = (f_recent[n] / max(1, len(recent))) - (f_prev[n] / max(1, len(prev)))
        reversal = max(0.0, -trend)
        gap_pressure = min(gap[n], 16) / 16.0
        scores[n] = w_persist * persist + w_reverse * reversal + w_gap * gap_pressure

    # Field geometry gently reshapes the candidate pool. When recent draws drift
    # upward/downward, X does not predict the same direction blindly; it applies
    # only a small coordinate bias while preserving uncertainty.
    target_center = 19.0 + max(-3.0, min(3.0, center_delta * 0.35))
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], abs(n - target_center), n))

    # Build Flow incrementally to avoid collapsing onto one cluster.
    chosen = []
    for n in ranked:
        if not chosen:
            chosen.append(n)
            continue
        if len(chosen) >= 7:
            break
        trial = sorted(chosen + [n])
        if len(trial) <= 3 or (max(trial) - min(trial)) >= 3 * (len(trial) - 1):
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
            "volatility": round(volatility, 4),
            "w_persist": round(w_persist, 4),
            "w_reverse": round(w_reverse, 4),
            "w_gap": round(w_gap, 4),
        },
    )
