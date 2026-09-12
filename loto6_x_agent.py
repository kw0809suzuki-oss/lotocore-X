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


def _mean(draw: list[int] | tuple[int, ...]) -> float:
    return sum(draw) / len(draw)


def _variance(draw: list[int] | tuple[int, ...]) -> float:
    m = _mean(draw)
    return sum((x - m) ** 2 for x in draw) / len(draw)


def _band(center: float) -> str:
    # Coarse observation only. LOTO6 is intentionally not over-segmented.
    if center < 20.0:
        return "low"
    if center > 25.0:
        return "high"
    return "mid"


def _field_features(draws: list[list[int]]) -> dict:
    recent = draws[-8:]
    centers = [_mean(d) for d in recent]
    bands = [_band(c) for c in centers]
    moves = [centers[i] - centers[i - 1] for i in range(1, len(centers))]
    mean_abs_move = sum(abs(v) for v in moves) / max(1, len(moves))
    reversals = sum(
        1 for a, b in zip(moves, moves[1:]) if a * b < 0 and abs(a) >= 4 and abs(b) >= 4
    )
    stay_high = len(bands) >= 2 and bands[-1] == bands[-2] == "high"
    stay_low = len(bands) >= 2 and bands[-1] == bands[-2] == "low"
    turbulent = mean_abs_move >= 5.0 or reversals >= 2
    if turbulent:
        regime = "turbulent"
    elif stay_high or stay_low:
        regime = "one_side_stay"
    else:
        regime = "quiet"
    return {
        "centers": centers,
        "bands": bands,
        "moves": moves,
        "mean_abs_move": mean_abs_move,
        "reversals": reversals,
        "stay_high": stay_high,
        "stay_low": stay_low,
        "regime": regime,
    }


def _analog_next_draws(draws: list[list[int]], features: dict, limit: int = 12) -> list[list[int]]:
    """Find prior coarse Field states and return their next draws.

    Similarity stays deliberately coarse: regime, final band, and short stay.
    This avoids turning random local geometry into a dense explanatory model.
    """
    target_regime = features["regime"]
    target_band = features["bands"][-1]
    target_stay_high = features["stay_high"]
    target_stay_low = features["stay_low"]

    matches: list[tuple[float, list[int]]] = []
    for end in range(20, len(draws) - 1):
        prefix = draws[:end]
        f = _field_features(prefix)
        score = 0.0
        if f["regime"] == target_regime:
            score += 3.0
        if f["bands"][-1] == target_band:
            score += 2.0
        if f["stay_high"] == target_stay_high:
            score += 1.0
        if f["stay_low"] == target_stay_low:
            score += 1.0
        score -= abs(f["mean_abs_move"] - features["mean_abs_move"]) / 8.0
        if score > 3.0:
            matches.append((score, draws[end]))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [draw for _, draw in matches[:limit]]


def score_candidates(history: pd.DataFrame) -> tuple[dict[int, float], dict]:
    draws = _draws(history)
    if len(draws) < 30:
        raise ValueError("LOTO6 X requires at least 30 historical draws")

    features = _field_features(draws)
    analogs = _analog_next_draws(draws, features)
    analog_counts = Counter(n for d in analogs for n in d)

    recent = draws[-12:]
    previous = draws[-24:-12]
    recent_counts = Counter(n for d in recent for n in d)
    previous_counts = Counter(n for d in previous for n in d)

    gap = {n: len(draws) for n in NUMBERS}
    for g, d in enumerate(reversed(draws)):
        for n in d:
            if gap[n] == len(draws):
                gap[n] = g

    # Relation weights emerge from the current Field rather than from a fixed nucleus.
    if features["regime"] == "turbulent":
        w_analog, w_persist, w_reverse, w_gap = 0.45, 0.18, 0.22, 0.15
    elif features["regime"] == "one_side_stay":
        w_analog, w_persist, w_reverse, w_gap = 0.35, 0.30, 0.15, 0.20
    else:
        w_analog, w_persist, w_reverse, w_gap = 0.30, 0.35, 0.10, 0.25

    scores: dict[int, float] = {}
    for n in NUMBERS:
        analog = analog_counts[n] / max(1, len(analogs))
        persist = recent_counts[n] / max(1, len(recent))
        prev = previous_counts[n] / max(1, len(previous))
        reversal = max(0.0, prev - persist)
        gap_pressure = min(gap[n], 18) / 18.0
        scores[n] = (
            w_analog * analog
            + w_persist * persist
            + w_reverse * reversal
            + w_gap * gap_pressure
        )

    state = {
        "model": "loto6_x",
        "field_regime": features["regime"],
        "field_band": features["bands"][-1],
        "field_bands": "->".join(features["bands"]),
        "field_mean_abs_move": round(features["mean_abs_move"], 4),
        "field_reversals": features["reversals"],
        "field_stay_high": int(features["stay_high"]),
        "field_stay_low": int(features["stay_low"]),
        "analog_count": len(analogs),
        "w_analog": w_analog,
        "w_persist": w_persist,
        "w_reverse": w_reverse,
        "w_gap": w_gap,
    }
    return scores, state


def predict(history: pd.DataFrame) -> Prediction:
    """Return one 6-number X candidate generated from the current LOTO6 Field."""
    scores, state = score_candidates(history)
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], n))
    chosen = tuple(sorted(ranked[:DRAW_SIZE]))
    state = dict(state)
    state.update(
        {
            "center": round(_mean(chosen), 4),
            "variance": round(_variance(chosen), 4),
            "ranked_top12": "-".join(map(str, ranked[:12])),
        }
    )
    return Prediction(chosen, state)


def generate_tickets(history: pd.DataFrame, count: int = 15) -> tuple[list[tuple[int, ...]], dict]:
    """Generate several local actions from one X Field without a hard-fixed nucleus."""
    scores, state = score_candidates(history)
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], n))
    pool = ranked[:18]

    tickets: list[tuple[int, ...]] = []
    for i in range(count * 4):
        # Deterministic rotation: reproducible before the result is known.
        offset = i % len(pool)
        rotated = pool[offset:] + pool[:offset]
        stride = 2 + (i % 4)
        candidate: list[int] = []
        for j in range(0, len(rotated), stride):
            n = rotated[j]
            if n not in candidate:
                candidate.append(n)
            if len(candidate) == DRAW_SIZE:
                break
        if len(candidate) < DRAW_SIZE:
            for n in rotated:
                if n not in candidate:
                    candidate.append(n)
                if len(candidate) == DRAW_SIZE:
                    break
        ticket = tuple(sorted(candidate[:DRAW_SIZE]))
        if ticket not in tickets:
            tickets.append(ticket)
        if len(tickets) >= count:
            break

    state = dict(state)
    state["ranked_top18"] = "-".join(map(str, pool))
    state["ticket_count"] = len(tickets)
    return tickets, state
