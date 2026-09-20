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


def _spacing_features(draw):
    """Internal six-gap geometry, independent of absolute center/range."""
    gaps = [b - a for a, b in zip(draw, draw[1:])]
    if not gaps:
        return 0.0, 0.0
    mean_gap = sum(gaps) / len(gaps)
    irregularity = (
        sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    ) ** 0.5 / max(mean_gap, 1e-9)
    short_share = sum(g <= 2 for g in gaps) / len(gaps)
    return irregularity, short_share


def predict(
    history: pd.DataFrame,
    competition_gate: bool = True,
    spacing_v2: bool = False,
    action_point: bool = False,
) -> Prediction:
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

    recent_features = [_spacing_features(d) for d in recent[-4:]]
    before_features = [_spacing_features(d) for d in recent[:4]]
    irr_recent = sum(v[0] for v in recent_features) / 4.0
    irr_before = sum(v[0] for v in before_features) / 4.0
    short_recent = sum(v[1] for v in recent_features) / 4.0
    short_before = sum(v[1] for v in before_features) / 4.0
    irr_delta = irr_recent - irr_before
    short_delta = short_recent - short_before
    target_irregularity = max(0.0, min(1.5, irr_recent + 0.25 * irr_delta))
    target_short_share = max(0.0, min(1.0, short_recent + 0.25 * short_delta))

    spacing_strength = min(1.0, abs(irr_delta) / 0.35 + abs(short_delta) / 0.30)
    pre_weights = sorted((w_persist, w_reverse, w_gap, w_shape), reverse=True)
    competition_margin = pre_weights[0] - pre_weights[1]
    action_signal = spacing_strength * (
        1.0 - min(1.0, competition_margin / 0.20)
    )
    spacing_engaged = spacing_v2 and (
        (not action_point) or action_signal >= 0.45
    )

    if spacing_engaged:
        # A distinct local-spacing axis: no absolute-number or center signal.
        w_geom = 0.08 + 0.12 * spacing_strength
        relation_conflict = False
        geom_active = True
    else:
        w_geom = 0.08 + 0.12 * min(1.0, abs(gap_delta) / 2.5)
        # Legacy relation shares Shape's outer/center axis. Gate redundancy.
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
    score_components = {}
    for n in NUMBERS:
        persist = f_recent[n] / max(1, len(recent))
        trend = persist - (f_prev[n] / max(1, len(prev)))
        reversal = max(0.0, -trend)
        gap_pressure = min(gap[n], 16) / 16.0

        is_outer = 1.0 if (n <= 12 or n >= 26) else 0.0
        shape = is_outer if shape_delta >= 0 else 1.0 - is_outer

        if spacing_engaged:
            # Pair geometry cannot be assigned honestly to an isolated number.
            geometry = 0.0
        else:
            dist_center = abs(n - 19.0) / 18.0
            geometry = dist_center if gap_delta >= 0 else 1.0 - dist_center

        c_persist = w_persist * persist
        c_reverse = w_reverse * reversal
        c_gap = w_gap * gap_pressure
        c_shape = w_shape * shape
        c_geom = w_geom * geometry
        scores[n] = c_persist + c_reverse + c_gap + c_shape + c_geom
        score_components[n] = {
            "persist_raw": float(persist),
            "reversal_raw": float(reversal),
            "gap_pressure_raw": float(gap_pressure),
            "shape_raw": float(shape),
            "geometry_raw": float(geometry),
            "persist": float(c_persist),
            "reversal": float(c_reverse),
            "gap": float(c_gap),
            "shape": float(c_shape),
            "geometry": float(c_geom),
        }

    target_center = 19.0 + max(-3.0, min(3.0, center_delta * 0.35))
    ranked = sorted(NUMBERS, key=lambda n: (-scores[n], abs(n - target_center), n))

    chosen = []
    micro_applied = False
    micro_utility = 0.0
    target_gap = max(2.5, min(7.5, 5.0 + gap_delta * 0.35))
    if spacing_engaged and action_point:
        # Start from the gated Champion and make at most one local intervention.
        champion = list(
            predict(history, competition_gate=competition_gate).numbers
        )
        base_irr, base_short = _spacing_features(champion)
        base_error = (
            abs(base_irr - target_irregularity)
            + abs(base_short - target_short_share)
        )
        best = (0.0, tuple(champion))
        for removed in champion:
            for added in NUMBERS:
                if added in champion:
                    continue
                trial = tuple(sorted((set(champion) - {removed}) | {added}))
                irr, short = _spacing_features(trial)
                spacing_error = (
                    abs(irr - target_irregularity)
                    + abs(short - target_short_share)
                )
                spacing_gain = base_error - spacing_error
                relation_cost = max(0.0, scores[removed] - scores[added])
                utility = w_geom * spacing_gain - relation_cost
                candidate = (utility, tuple(-n for n in trial), trial)
                incumbent = (best[0], tuple(-n for n in best[1]), best[1])
                if candidate > incumbent:
                    best = (utility, trial)
        if best[0] > 0:
            chosen = list(best[1])
            micro_applied = True
            micro_utility = best[0]
        else:
            chosen = champion
    elif spacing_engaged:
        # Full Spacing v2 remains as an experimental comparison candidate.
        while len(chosen) < 7:
            candidates = []
            for n in NUMBERS:
                if n in chosen:
                    continue
                trial = sorted(chosen + [n])
                if len(trial) < 4:
                    spacing_error = 0.0
                else:
                    irr, short = _spacing_features(trial)
                    spacing_error = (
                        abs(irr - target_irregularity)
                        + abs(short - target_short_share)
                    )
                candidates.append((scores[n] - w_geom * spacing_error, scores[n], -n, n))
            chosen.append(max(candidates)[3])
    else:
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
            "spacing_irregularity_delta": round(irr_delta, 4),
            "spacing_short_delta": round(short_delta, 4),
            "target_irregularity": round(target_irregularity, 4),
            "target_short_share": round(target_short_share, 4),
            "spacing_v2": int(spacing_v2),
            "spacing_engaged": int(spacing_engaged),
            "action_point_mode": int(action_point),
            "spacing_strength": round(spacing_strength, 4),
            "competition_margin": round(competition_margin, 4),
            "action_signal": round(action_signal, 4),
            "micro_applied": int(micro_applied),
            "micro_utility": round(micro_utility, 6),
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


def score_snapshot(
    history: pd.DataFrame,
    competition_gate: bool = True,
    spacing_v2: bool = False,
    action_point: bool = False,
) -> dict:
    """Observation-only full 1..37 score/rank snapshot.

    This mirrors the pre-selection score construction used by predict() and
    does not modify predict() behavior. For spacing_v2/action_point, the
    snapshot still records the isolated-number score vector before any
    pair-geometry intervention.
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

    outer_recent = _outer_share(recent[-4:])
    outer_before = _outer_share(recent[:4])
    shape_delta = outer_recent - outer_before
    w_shape = 0.12 + 0.18 * min(1.0, abs(shape_delta) / 0.35)

    gap_recent = sum(_mean_gap(d) for d in recent[-4:]) / 4.0
    gap_before = sum(_mean_gap(d) for d in recent[:4]) / 4.0
    gap_delta = gap_recent - gap_before

    recent_features = [_spacing_features(d) for d in recent[-4:]]
    before_features = [_spacing_features(d) for d in recent[:4]]
    irr_recent = sum(v[0] for v in recent_features) / 4.0
    irr_before = sum(v[0] for v in before_features) / 4.0
    short_recent = sum(v[1] for v in recent_features) / 4.0
    short_before = sum(v[1] for v in before_features) / 4.0
    irr_delta = irr_recent - irr_before
    short_delta = short_recent - short_before
    target_irregularity = max(0.0, min(1.5, irr_recent + 0.25 * irr_delta))
    target_short_share = max(0.0, min(1.0, short_recent + 0.25 * short_delta))

    spacing_strength = min(1.0, abs(irr_delta) / 0.35 + abs(short_delta) / 0.30)
    pre_weights = sorted((w_persist, w_reverse, w_gap, w_shape), reverse=True)
    competition_margin = pre_weights[0] - pre_weights[1]
    action_signal = spacing_strength * (
        1.0 - min(1.0, competition_margin / 0.20)
    )
    spacing_engaged = spacing_v2 and (
        (not action_point) or action_signal >= 0.45
    )

    if spacing_engaged:
        w_geom = 0.08 + 0.12 * spacing_strength
        relation_conflict = False
        geom_active = True
    else:
        w_geom = 0.08 + 0.12 * min(1.0, abs(gap_delta) / 2.5)
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

        if spacing_engaged:
            geometry = 0.0
        else:
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
    ranks = {n: i + 1 for i, n in enumerate(ranked)}

    return {
        "scores": {str(n): float(scores[n]) for n in NUMBERS},
        "ranks": {str(n): int(ranks[n]) for n in NUMBERS},
        "components": {str(n): score_components[n] for n in NUMBERS},
        "state": {
            "model": "x",
            "snapshot_kind": "pre_selection_score",
            "competition_gate": int(competition_gate),
            "spacing_v2": int(spacing_v2),
            "action_point_mode": int(action_point),
            "spacing_engaged": int(spacing_engaged),
            "target_center": round(target_center, 6),
            "field_center_delta": round(center_delta, 6),
            "field_spread_delta": round(spread_delta, 6),
            "shape_delta": round(shape_delta, 6),
            "gap_delta": round(gap_delta, 6),
            "spacing_irregularity_delta": round(irr_delta, 6),
            "spacing_short_delta": round(short_delta, 6),
            "spacing_strength": round(spacing_strength, 6),
            "competition_margin": round(competition_margin, 6),
            "action_signal": round(action_signal, 6),
            "w_persist": round(w_persist, 6),
            "w_reverse": round(w_reverse, 6),
            "w_gap": round(w_gap, 6),
            "w_shape": round(w_shape, 6),
            "w_geom": round(w_geom, 6),
            "geom_active": int(geom_active),
        },
    }
