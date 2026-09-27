from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

import lotocore
import x_agent

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_selector_state_grid_v0.csv")
WINDOW = 100
MIN_WIDTH_HISTORY = 60


def ranked(snapshot):
    ranks = {int(n): int(r) for n, r in snapshot["ranks"].items()}
    return sorted(ranks, key=lambda n: (ranks[n], n))


def top10_concentration(snapshot):
    scores = {int(n): float(v) for n, v in snapshot["scores"].items()}
    order = ranked(snapshot)
    total = sum(max(0.0, scores[n]) for n in scores)
    if total <= 0:
        return 0.0
    return sum(max(0.0, scores[n]) for n in order[:10]) / total


def relation_bin(overlap):
    if overlap <= 1:
        return "separate"
    if overlap >= 4:
        return "high_overlap"
    return "partial"


def width_bin(value, prior_values):
    if len(prior_values) < MIN_WIDTH_HISTORY:
        return None, None, None
    s = pd.Series(prior_values, dtype=float)
    q1 = float(s.quantile(1 / 3))
    q2 = float(s.quantile(2 / 3))
    if value <= q1:
        label = "wide"
    elif value >= q2:
        label = "narrow"
    else:
        label = "medium"
    return label, q1, q2


def state_snapshot(history):
    cs = lotocore.score_snapshot(history)
    xs = x_agent.score_snapshot(history, competition_gate=True)
    cr = ranked(cs)
    xr = ranked(xs)
    overlap = len(set(cr[:10]) & set(xr[:10]))
    core_conc = top10_concentration(cs)
    x_conc = top10_concentration(xs)
    width_value = (core_conc + x_conc) / 2.0
    return {
        "overlap": overlap,
        "relation": relation_bin(overlap),
        "core_concentration": core_conc,
        "x_concentration": x_conc,
        "width_value": width_value,
    }


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    prior_widths = []
    rows = []

    for i in range(WINDOW, len(df)):
        history = df.iloc[i-WINDOW:i]
        target_round = int(df.iloc[i]["round"])
        s = state_snapshot(history)

        wbin, q1, q2 = width_bin(s["width_value"], prior_widths)
        rows.append({
            "target_round": target_round,
            "history_end_round": int(df.iloc[i-1]["round"]),
            "overlap": s["overlap"],
            "relation": s["relation"],
            "core_top10_concentration": s["core_concentration"],
            "x_top10_concentration": s["x_concentration"],
            "width_value": s["width_value"],
            "prior_width_n": len(prior_widths),
            "prior_q1": q1,
            "prior_q2": q2,
            "width_bin": wbin,
            "state": None if wbin is None else f"{s['relation']}__{wbin}",
        })
        prior_widths.append(s["width_value"])

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    eligible = res.dropna(subset=["state"]).copy()
    print("=== LOTO7 SELECTOR STATE GRID v0 ===")
    print("No lottery target outcome is used.")
    print("Horizontal axis: CORE/X top10 overlap (same thresholds as current app).")
    print("Vertical axis: mean CORE/X top10 score concentration.")
    print("Width thresholds use only PRIOR observed width values; no future quantiles.")
    print(f"history_window={WINDOW} width_warmup={MIN_WIDTH_HISTORY}")
    print(f"all_state_rows={len(res)} eligible_rows={len(eligible)}")
    print()

    relations = ["separate", "partial", "high_overlap"]
    widths = ["wide", "medium", "narrow"]
    counts = Counter(eligible["state"])
    print("=== 9-STATE COUNTS ===")
    for w in widths:
        print(
            f"{w:6s}  "
            + "  ".join(
                f"{r}={counts[f'{r}__{w}']}" for r in relations
            )
        )
    print()

    sparse = [s for s, n in sorted(counts.items()) if n < 20]
    missing = [
        f"{r}__{w}"
        for w in widths for r in relations
        if counts[f"{r}__{w}"] == 0
    ]
    print(f"cells_below_20={len(sparse)} {sparse}")
    print(f"missing_cells={len(missing)} {missing}")
    print()

    print("=== RELATION COUNTS ===")
    print(eligible["relation"].value_counts().to_dict())
    print("=== WIDTH COUNTS ===")
    print(eligible["width_bin"].value_counts().to_dict())
    print()

    recent = eligible.tail(100)
    print("=== LAST100 STATE COUNTS ===")
    print(recent["state"].value_counts().to_dict())
    print()

    # Current pre-draw state for next round, using all available history.
    current_history = df.tail(WINDOW)
    cur = state_snapshot(current_history)
    wbin, q1, q2 = width_bin(cur["width_value"], prior_widths)
    next_round = int(df.iloc[-1]["round"]) + 1
    print("=== CURRENT PRE-DRAW STATE ===")
    print(
        f"history_end={int(df.iloc[-1]['round'])} next_round={next_round} "
        f"overlap={cur['overlap']} relation={cur['relation']} "
        f"core_conc={cur['core_concentration']:.6f} "
        f"x_conc={cur['x_concentration']:.6f} "
        f"width_value={cur['width_value']:.6f} "
        f"prior_q1={q1:.6f} prior_q2={q2:.6f} width={wbin} "
        f"state={cur['relation']}__{wbin}"
    )

    print()
    print("BOUNDARY: This probe validates the State partition only.")
    print("BOUNDARY: It does not claim that any model is better in any State.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
