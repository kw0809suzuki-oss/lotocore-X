from __future__ import annotations

from pathlib import Path
import statistics

import pandas as pd

import x_agent
import probe_loto7_dynamic_band_online_replay_v0 as box
from probe_loto7_current_match_model_v0 import (
    WINDOW,
    proxy_model_metrics,
    state_features,
)

DATA = Path("data/loto7.csv")

FEATURES = (
    "field_center_delta",
    "field_spread_delta",
    "shape_delta",
    "gap_delta",
    "spacing_strength",
    "competition_margin",
    "action_signal",
    "volatility",
    "w_persist",
    "w_reverse",
    "w_gap",
    "w_shape",
    "w_geom",
)


def safe(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def effect_size(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    pooled = ((va + vb) / 2.0) ** 0.5
    return 0.0 if pooled < 1e-12 else (ma - mb) / pooled


def build_rows():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(WINDOW, len(df)):
        rnd = int(df.iloc[i]["round"])
        history = df.iloc[i-WINDOW:i]
        actual = box.truth(df.iloc[i])

        state = state_features(history)
        perf = proxy_model_metrics(actual, state, rnd)

        xs = x_agent.score_snapshot(history, competition_gate=True)
        xstate = xs.get("state", {})

        rec = {
            "round": rnd,
            "x_max": perf["x"]["max"],
            "blend_max": perf["blend"]["max"],
            "x_p4": perf["x"]["p4"],
            "blend_p4": perf["blend"]["p4"],
            "delta_max": perf["x"]["max"] - perf["blend"]["max"],
            "delta_p4": perf["x"]["p4"] - perf["blend"]["p4"],
        }
        for f in FEATURES:
            rec[f] = safe(xstate.get(f, 0.0))
        rows.append(rec)

    return rows


def summarize_groups(rows, label):
    # Meaningful outcome separation, fixed before inspection.
    win = [r for r in rows if r["delta_max"] >= 0.25]
    lose = [r for r in rows if r["delta_max"] <= -0.25]
    tie = [r for r in rows if -0.25 < r["delta_max"] < 0.25]

    print(f"=== {label} ===")
    print(f"X_SPIKE n={len(win)} TIE n={len(tie)} X_WEAK n={len(lose)}")
    print(
        f"mean delta_max spike={statistics.mean(r['delta_max'] for r in win) if win else 0:+.4f} "
        f"weak={statistics.mean(r['delta_max'] for r in lose) if lose else 0:+.4f}"
    )

    effects = []
    for f in FEATURES:
        a = [r[f] for r in win]
        b = [r[f] for r in lose]
        d = effect_size(a, b)
        effects.append((abs(d), d, f, statistics.mean(a) if a else 0, statistics.mean(b) if b else 0))

    effects.sort(reverse=True)
    for _, d, f, mw, ml in effects:
        print(f"{f:24s} effect={d:+.3f} spike_mean={mw:+.4f} weak_mean={ml:+.4f}")
    print()
    return effects


def holdout_one_feature(rows, feature, direction):
    mid = len(rows) // 2
    train = rows[:mid]
    test = rows[mid:]

    vals = [r[feature] for r in train]
    threshold = statistics.median(vals)

    # Direction comes only from train group observation.
    def choose(r):
        active = r[feature] >= threshold if direction > 0 else r[feature] <= threshold
        return "x" if active else "blend"

    chosen = []
    blend_static = []
    x_static = []
    active_n = 0
    for r in test:
        m = choose(r)
        if m == "x":
            active_n += 1
        chosen.append(r[f"{m}_max"])
        blend_static.append(r["blend_max"])
        x_static.append(r["x_max"])

    print("=== SIMPLE HOLDOUT GATE ===")
    print(
        f"train_rounds={train[0]['round']}..{train[-1]['round']} "
        f"test_rounds={test[0]['round']}..{test[-1]['round']}"
    )
    print(
        f"feature={feature} threshold(train median)={threshold:+.6f} "
        f"x_side={'high' if direction > 0 else 'low'}"
    )
    print(
        f"test X-selected={active_n}/{len(test)} "
        f"gate_mean_max={statistics.mean(chosen):.6f} "
        f"blend_static={statistics.mean(blend_static):.6f} "
        f"x_static={statistics.mean(x_static):.6f} "
        f"gate_minus_blend={statistics.mean(chosen)-statistics.mean(blend_static):+.6f}"
    )
    print()


def main():
    rows = build_rows()
    mid = len(rows) // 2
    train = rows[:mid]

    print("=== LOTO7 X SPIKE STATE v0 ===")
    print("Question: does X have a pre-draw State that separates spike rounds from weak rounds?")
    print("Outcome is used only to label historical X-vs-Blend result after prediction.")
    print("All FEATURES are captured before the target draw.")
    print("Spike/weak threshold is fixed at +/-0.25 expected max-hit difference.")
    print()

    train_effects = summarize_groups(train, "TRAIN HALF")
    summarize_groups(rows[mid:], "HOLDOUT HALF (OBSERVATION ONLY)")

    # Select exactly one feature using TRAIN only.
    _, d, best_feature, _, _ = train_effects[0]
    holdout_one_feature(rows, best_feature, 1 if d > 0 else -1)

    print("BOUNDARY: feature selection uses TRAIN only; holdout gate is not re-tuned.")
    print("DELETE RULE: if holdout gate does not beat static Blend, do not promote this X-state selector.")
    print("BOUNDARY: historical Floot AI outputs are not replayable; CORE/X/Blend are GitHub proxy packs.")


if __name__ == "__main__":
    main()
