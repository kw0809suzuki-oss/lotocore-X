from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

import lotocore
import x_agent

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_selector_state_grid_v1.csv")
WINDOW = 100
MIN_STATE_HISTORY = 60


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


def prior_tertile(value, prior):
    if len(prior) < MIN_STATE_HISTORY:
        return None, None, None
    s = pd.Series(prior, dtype=float)
    q1 = float(s.quantile(1 / 3))
    q2 = float(s.quantile(2 / 3))
    if value <= q1:
        label = "low"
    elif value >= q2:
        label = "high"
    else:
        label = "mid"
    return label, q1, q2


def snapshot(history):
    cs = lotocore.score_snapshot(history)
    xs = x_agent.score_snapshot(history, competition_gate=True)
    cr = ranked(cs)
    xr = ranked(xs)

    overlap = len(set(cr[:10]) & set(xr[:10]))
    core_conc = top10_concentration(cs)
    x_conc = top10_concentration(xs)
    concentration = (core_conc + x_conc) / 2.0

    return {
        "overlap": overlap,
        "core_concentration": core_conc,
        "x_concentration": x_conc,
        "concentration": concentration,
    }


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    prior_overlap = []
    prior_concentration = []
    rows = []

    for i in range(WINDOW, len(df)):
        history = df.iloc[i-WINDOW:i]
        s = snapshot(history)

        relation, oq1, oq2 = prior_tertile(s["overlap"], prior_overlap)
        conc_bin, cq1, cq2 = prior_tertile(
            s["concentration"], prior_concentration
        )

        # Low concentration = broad candidate field.
        width = {
            "low": "wide",
            "mid": "medium",
            "high": "narrow",
        }.get(conc_bin)

        state = (
            None
            if relation is None or width is None
            else f"{relation}_agreement__{width}"
        )

        rows.append({
            "target_round": int(df.iloc[i]["round"]),
            "history_end_round": int(df.iloc[i-1]["round"]),
            "overlap": s["overlap"],
            "prior_overlap_n": len(prior_overlap),
            "overlap_q1": oq1,
            "overlap_q2": oq2,
            "agreement_bin": relation,
            "core_top10_concentration": s["core_concentration"],
            "x_top10_concentration": s["x_concentration"],
            "concentration": s["concentration"],
            "concentration_q1": cq1,
            "concentration_q2": cq2,
            "width": width,
            "state": state,
        })

        # Strict chronology: current observation joins the reference only
        # after its State was classified.
        prior_overlap.append(s["overlap"])
        prior_concentration.append(s["concentration"])

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    eligible = res.dropna(subset=["state"]).copy()

    print("=== LOTO7 SELECTOR STATE GRID v1 ===")
    print("No lottery target outcome is used.")
    print("Horizontal: CORE/X agreement low/mid/high from PRIOR overlap distribution.")
    print("Vertical: candidate width wide/medium/narrow from PRIOR score-concentration distribution.")
    print("Both axes are strict walk-forward; no future quantiles.")
    print(f"history_window={WINDOW} state_warmup={MIN_STATE_HISTORY}")
    print(f"all_rows={len(res)} eligible_rows={len(eligible)}")
    print()

    agreements = ["low", "mid", "high"]
    widths = ["wide", "medium", "narrow"]
    counts = Counter(eligible["state"])

    print("=== 9-STATE COUNTS ===")
    for w in widths:
        print(
            f"{w:6s}  "
            + "  ".join(
                f"{a}={counts[f'{a}_agreement__{w}']}"
                for a in agreements
            )
        )

    sparse = [
        f"{a}_agreement__{w}"
        for w in widths for a in agreements
        if counts[f"{a}_agreement__{w}"] < 20
    ]
    missing = [
        f"{a}_agreement__{w}"
        for w in widths for a in agreements
        if counts[f"{a}_agreement__{w}"] == 0
    ]
    print()
    print(f"cells_below_20={len(sparse)} {sparse}")
    print(f"missing_cells={len(missing)} {missing}")
    print()

    print("=== AGREEMENT COUNTS ===")
    print(eligible["agreement_bin"].value_counts().to_dict())
    print("=== WIDTH COUNTS ===")
    print(eligible["width"].value_counts().to_dict())
    print()

    recent = eligible.tail(100)
    print("=== LAST100 STATE COUNTS ===")
    print(recent["state"].value_counts().to_dict())
    print()

    cur = snapshot(df.tail(WINDOW))
    relation, oq1, oq2 = prior_tertile(cur["overlap"], prior_overlap)
    conc_bin, cq1, cq2 = prior_tertile(cur["concentration"], prior_concentration)
    width = {"low": "wide", "mid": "medium", "high": "narrow"}[conc_bin]
    next_round = int(df.iloc[-1]["round"]) + 1

    print("=== CURRENT PRE-DRAW STATE ===")
    print(
        f"history_end={int(df.iloc[-1]['round'])} next_round={next_round} "
        f"overlap={cur['overlap']} overlap_q1={oq1:.3f} overlap_q2={oq2:.3f} "
        f"agreement={relation} "
        f"concentration={cur['concentration']:.6f} "
        f"conc_q1={cq1:.6f} conc_q2={cq2:.6f} "
        f"width={width} state={relation}_agreement__{width}"
    )

    print()
    print("BOUNDARY: State partition only. No model-performance claim.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
