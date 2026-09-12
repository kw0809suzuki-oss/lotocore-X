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


def _field_features(draws: list[list[int]]) -> dict:
    """Observe LOTO6 as a large breathing motion around its center.

    The upper Field is intentionally coarse: expansion / contraction / neutral.
    High/low position is retained only as a secondary coordinate, not the regime itself.
    """
    recent = draws[-10:]
    centers = [_mean(d) for d in recent]
    moves = [centers[i] - centers[i - 1] for i in range(1, len(centers))]

    recent_amp = max(centers[-5:]) - min(centers[-5:])
    prior_amp = max(centers[:5]) - min(centers[:5])
    amp_delta = recent_amp - prior_amp

    # Soft direction: avoid forcing every small change into a state transition.
    if amp_delta >= 3.0:
        breathing = "expanding"
    elif amp_delta <= -3.0:
        breathing = "contracting"
    else:
        breathing = "neutral"

    long_centers = [_mean(d) for d in draws[-60:]]
    field_center = sum(long_centers) / len(long_centers)
    position = centers[-1] - field_center
    if position >= 4.0:
        side = "upper"
    elif position <= -4.0:
        side = "lower"
    else:
        side = "center"

    return {
        "centers": centers,
        "moves": moves,
        "prior_amplitude": prior_amp,
        "recent_amplitude": recent_amp,
        "amplitude_delta": amp_delta,
        "breathing": breathing,
        "field_center": field_center,
        "position": position,
        "side": side,
    }


def _analog_next_draws(draws: list[list[int]], features: dict, limit: int = 12) -> list[list[int]]:
    """Find prior states with a similar large-scale breathing relation."""
    matches: list[tuple[float, list[int]]] = []
    for end in range(30, len(draws) - 1):
        f = _field_features(draws[:end])
        score = 0.0
        if f["breathing"] == features["breathing"]:
            score += 4.0
        if f["side"] == features["side"]:
            score += 1.5
        score -= abs(f["recent_amplitude"] - features["recent_amplitude"]) / 10.0
        score -= abs(f["amplitude_delta"] - features["amplitude_delta"]) / 12.0
        if score > 2.5:
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

    # Weight changes follow the large breathing state, not a high/low prediction rule.
    if features["breathing"] == "expanding":
        w_analog, w_persist, w_reverse, w_gap = 0.50, 0.15, 0.20, 0.15
    elif features["breathing"] == "contracting":
        w_analog, w_persist, w_reverse, w_gap = 0.40, 0.30, 0.10, 0.20
    else:
        w_analog, w_persist, w_reverse, w_gap = 0.35, 0.30, 0.15, 0.20

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
        "model": "loto6_x_breathing",
        "field_breathing": features["breathing"],
        "field_side": features["side"],
        "field_center": round(features["field_center"], 4),
        "field_position": round(features["position"], 4),
        "field_prior_amplitude": round(features["prior_amplitude"], 4),
        "field_recent_amplitude": round(features["recent_amplitude"], 4),
        "field_amplitude_delta": round(features["amplitude_delta"], 4),
        "analog_count": len(analogs),
        "w_analog": w_analog,
        "w_persist": w_persist,
        "w_reverse": w_reverse,
        "w_gap": w_gap,
    }
    return scores, state


def predict(history: pd.DataFrame) -> Prediction:
    scores, state = score_candidates(history)
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], n))
    chosen = tuple(sorted(ranked[:DRAW_SIZE]))
    state = dict(state)
    state.update({
        "center": round(_mean(chosen), 4),
        "variance": round(_variance(chosen), 4),
        "ranked_top12": "-".join(map(str, ranked[:12])),
    })
    return Prediction(chosen, state)


def generate_tickets(history: pd.DataFrame, count: int = 15) -> tuple[list[tuple[int, ...]], dict]:
    scores, state = score_candidates(history)
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], n))
    pool = ranked[:18]

    tickets: list[tuple[int, ...]] = []
    for i in range(count * 4):
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
