from __future__ import annotations

from pathlib import Path
import statistics

import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, NUMBERS,
    draws_from_df, frequency_distribution, mix_distribution,
    distribution_state, rank_numbers, choose_shape_aware,
)

OUT = Path("results/loto7_c_auditor_a_to_b_separator_v0.csv")
BOUNDARY_K = 3
EPS = 1e-12


def truth(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def hits(actual, picks):
    return len(set(actual) & set(picks))


def sign(x):
    if x > EPS:
        return 1
    if x < -EPS:
        return -1
    return 0


def make_state(round_no, p, prev_p):
    ranked = rank_numbers(p)
    center, spread = distribution_state(p)
    core, *_ = choose_shape_aware(ranked, p, center, spread)
    core = tuple(sorted(core))
    core_set = set(core)
    boundary = tuple(n for n in ranked if n not in core_set)[:BOUNDARY_K]
    delta = {n: 0.0 if prev_p is None else p[n] - prev_p[n] for n in NUMBERS}
    return {
        "round": round_no,
        "p": p,
        "core": core,
        "boundary": boundary,
        "delta": delta,
        "ranked": ranked,
    }


def pick_a_to_b(state):
    baseline = tuple(sorted(state["ranked"][:7]))
    bset = set(baseline)
    aset = set(state["core"])
    boundary = set(state["boundary"])

    rising_b = [n for n in boundary - bset if sign(state["delta"][n]) > 0]
    falling_a = [n for n in (bset & aset) if sign(state["delta"][n]) < 0]

    falling_b_in_baseline = [n for n in (bset - aset) if sign(state["delta"][n]) < 0]
    omitted_a_nonfalling = [n for n in (aset - bset) if sign(state["delta"][n]) >= 0]

    can_a_to_b = bool(rising_b and falling_a)
    can_b_to_a = bool(falling_b_in_baseline and omitted_a_nonfalling)

    # Preserve v0 auditor behavior: ambiguous means no intervention.
    if not can_a_to_b or can_b_to_a:
        return None

    in_num = max(rising_b, key=lambda n: (state["delta"][n], state["p"][n], -n))
    out_num = min(falling_a, key=lambda n: (state["delta"][n], state["p"][n], n))
    audited = tuple(sorted((bset - {out_num}) | {in_num}))

    boundary_rank = list(state["boundary"]).index(in_num) + 1
    core_baseline_overlap = len(aset & bset)
    baseline_noncore_count = 7 - core_baseline_overlap

    return {
        "baseline": baseline,
        "audited": audited,
        "out": out_num,
        "in": in_num,
        "out_delta": state["delta"][out_num],
        "in_delta": state["delta"][in_num],
        "motion_gap": state["delta"][in_num] - state["delta"][out_num],
        "out_p": state["p"][out_num],
        "in_p": state["p"][in_num],
        "mass_gap_in_minus_out": state["p"][in_num] - state["p"][out_num],
        "in_boundary_rank": boundary_rank,
        "core_baseline_overlap": core_baseline_overlap,
        "baseline_noncore_count": baseline_noncore_count,
    }


def stat_block(df, label):
    print(f"--- {label} n={len(df)} ---")
    if df.empty:
        print("none")
        print()
        return

    cols = [
        "out_delta", "in_delta", "motion_gap",
        "mass_gap_in_minus_out", "in_boundary_rank",
        "core_baseline_overlap", "baseline_noncore_count",
    ]
    for c in cols:
        print(
            f"{c}: mean={df[c].mean():+.6f} "
            f"median={df[c].median():+.6f} "
            f"min={df[c].min():+.6f} max={df[c].max():+.6f}"
        )
    print()


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = draws_from_df(df)

    states = []
    prev_p = None
    for i in range(LONG_WINDOW, len(draws)):
        lp = frequency_distribution(draws[i - LONG_WINDOW:i])
        rp = frequency_distribution(draws[i - RECENT_WINDOW:i])
        p = mix_distribution(lp, rp)
        states.append(make_state(int(df.iloc[i]["round"]), p, prev_p))
        prev_p = p

    rows = []
    for sidx, state in enumerate(states):
        if sidx == 0:
            continue

        probe = pick_a_to_b(state)
        if probe is None:
            continue

        i = LONG_WINDOW + sidx
        actual = truth(df.iloc[i])
        bh = hits(actual, probe["baseline"])
        ah = hits(actual, probe["audited"])
        delta_hit = ah - bh
        outcome = "IMPROVE" if delta_hit > 0 else ("WORSEN" if delta_hit < 0 else "SAME")

        rows.append({
            "round": state["round"],
            "outcome": outcome,
            "baseline_hit": bh,
            "audited_hit": ah,
            "delta_hit": delta_hit,
            **probe,
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 C AUDITOR A->B SEPARATOR OBSERVATION v0 ===")
    print("Purpose: observe whether pre-intervention state differs between successful and harmful A->B swaps.")
    print("No new rule, threshold, weight, or selector is fitted.")
    print()

    print("COUNTS")
    print(res["outcome"].value_counts().to_string())
    print()

    stat_block(res[res["outcome"] == "IMPROVE"], "IMPROVE")
    stat_block(res[res["outcome"] == "WORSEN"], "WORSEN")
    stat_block(res[res["outcome"] == "SAME"], "SAME")

    imp = res[res["outcome"] == "IMPROVE"]
    bad = res[res["outcome"] == "WORSEN"]
    if len(imp) and len(bad):
        print("IMPROVE minus WORSEN | mean difference")
        for c in [
            "out_delta", "in_delta", "motion_gap",
            "mass_gap_in_minus_out", "in_boundary_rank",
            "core_baseline_overlap", "baseline_noncore_count",
        ]:
            d = imp[c].mean() - bad[c].mean()
            print(f"{c}: {d:+.6f}")
        print()

    print("LATE40 ONLY")
    cutoff = int(res["round"].min() + 0.60 * (res["round"].max() - res["round"].min()))
    late = res[res["round"] >= cutoff]
    print(late["outcome"].value_counts().to_string())
    print()
    stat_block(late[late["outcome"] == "IMPROVE"], "LATE40 IMPROVE")
    stat_block(late[late["outcome"] == "WORSEN"], "LATE40 WORSEN")

    print("RECENT 30 A->B INTERVENTIONS")
    cols = [
        "round", "outcome", "out", "in", "delta_hit",
        "out_delta", "in_delta", "motion_gap",
        "mass_gap_in_minus_out", "in_boundary_rank",
        "core_baseline_overlap",
    ]
    print(res[cols].tail(30).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
