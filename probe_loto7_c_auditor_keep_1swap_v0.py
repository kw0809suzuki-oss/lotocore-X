from __future__ import annotations

from pathlib import Path
import statistics

import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, NUMBERS,
    draws_from_df, frequency_distribution, mix_distribution,
    distribution_state, rank_numbers, choose_shape_aware,
)

OUT = Path("results/loto7_c_auditor_keep_1swap_v0.csv")
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

    delta = {
        n: 0.0 if prev_p is None else p[n] - prev_p[n]
        for n in NUMBERS
    }

    return {
        "round": round_no,
        "p": p,
        "core": core,
        "boundary": boundary,
        "delta": delta,
    }


def audit_one_swap(state):
    """
    C is an auditor, not a predictor.

    Baseline stays primary. C may only:
      KEEP
      A_TO_B: replace one falling A-side baseline number with one rising B-boundary number.
      B_TO_A: replace one falling B-side baseline number with one non-falling omitted A-core number.

    If both directions appear valid at once, ambiguity remains unresolved -> KEEP.
    There is no balance score, fitted weight, or threshold search.
    """
    baseline = tuple(sorted(rank_numbers(state["p"])[:7]))
    bset = set(baseline)
    aset = set(state["core"])
    boundary = set(state["boundary"])

    rising_b = [
        n for n in boundary - bset
        if sign(state["delta"][n]) > 0
    ]
    falling_a = [
        n for n in (bset & aset)
        if sign(state["delta"][n]) < 0
    ]

    falling_b_in_baseline = [
        n for n in (bset - aset)
        if sign(state["delta"][n]) < 0
    ]
    omitted_a_nonfalling = [
        n for n in (aset - bset)
        if sign(state["delta"][n]) >= 0
    ]

    can_a_to_b = bool(rising_b and falling_a)
    can_b_to_a = bool(falling_b_in_baseline and omitted_a_nonfalling)

    if can_a_to_b and can_b_to_a:
        return baseline, "KEEP_AMBIGUOUS", None, None

    if can_a_to_b:
        in_num = max(
            rising_b,
            key=lambda n: (state["delta"][n], state["p"][n], -n),
        )
        out_num = min(
            falling_a,
            key=lambda n: (state["delta"][n], state["p"][n], n),
        )
        audited = tuple(sorted((bset - {out_num}) | {in_num}))
        return audited, "A_TO_B", out_num, in_num

    if can_b_to_a:
        out_num = min(
            falling_b_in_baseline,
            key=lambda n: (state["delta"][n], state["p"][n], n),
        )
        in_num = max(
            omitted_a_nonfalling,
            key=lambda n: (state["delta"][n], state["p"][n], -n),
        )
        audited = tuple(sorted((bset - {out_num}) | {in_num}))
        return audited, "B_TO_A", out_num, in_num

    return baseline, "KEEP", None, None


def print_intervention_summary(rows, label):
    subset = [r for r in rows if r["intervened"]]
    print(f"--- {label} ---")
    print(f"all_rounds={len(rows)} interventions={len(subset)} rate={len(subset)/len(rows):.6%}" if rows else "all_rounds=0")

    if not subset:
        print("no interventions")
        print()
        return

    deltas = [r["delta_hit"] for r in subset]
    improved = sum(d > 0 for d in deltas)
    same = sum(d == 0 for d in deltas)
    worsened = sum(d < 0 for d in deltas)

    print(f"baseline_hit_mean={statistics.mean(r['baseline_hit'] for r in subset):.6f}")
    print(f"audited_hit_mean={statistics.mean(r['audited_hit'] for r in subset):.6f}")
    print(f"mean_delta_hit={statistics.mean(deltas):+.6f}")
    print(f"improved={improved}/{len(subset)} = {improved/len(subset):.6%}")
    print(f"same={same}/{len(subset)} = {same/len(subset):.6%}")
    print(f"worsened={worsened}/{len(subset)} = {worsened/len(subset):.6%}")
    print(f"net_improve_minus_worsen={improved-worsened:+d}")

    for direction in ("A_TO_B", "B_TO_A"):
        part = [r for r in subset if r["decision"] == direction]
        if not part:
            print(f"{direction}: n=0")
            continue
        ds = [r["delta_hit"] for r in part]
        ip = sum(d > 0 for d in ds)
        wp = sum(d < 0 for d in ds)
        print(
            f"{direction}: n={len(part)} "
            f"mean_delta={statistics.mean(ds):+.6f} "
            f"improve={ip} worsen={wp} net={ip-wp:+d}"
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

        i = LONG_WINDOW + sidx
        actual = truth(df.iloc[i])
        baseline = tuple(sorted(rank_numbers(state["p"])[:7]))
        audited, decision, out_num, in_num = audit_one_swap(state)

        baseline_hit = hits(actual, baseline)
        audited_hit = hits(actual, audited)
        intervened = decision in {"A_TO_B", "B_TO_A"}

        rows.append({
            "round": state["round"],
            "baseline": "-".join(f"{n:02d}" for n in baseline),
            "a_core": "-".join(f"{n:02d}" for n in state["core"]),
            "boundary": "-".join(f"{n:02d}" for n in state["boundary"]),
            "decision": decision,
            "intervened": intervened,
            "out": out_num,
            "in": in_num,
            "audited": "-".join(f"{n:02d}" for n in audited),
            "baseline_hit": baseline_hit,
            "audited_hit": audited_hit,
            "delta_hit": audited_hit - baseline_hit,
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 C AUDITOR | KEEP vs 1-SWAP v0 ===")
    print("Parent objective: improve actual next-draw hit count.")
    print("C is only an audit layer. It never generates a fresh 7-number set.")
    print("B is held as open boundary freedom, not predicted future.")
    print("No balance score, fitted weight, or threshold search.")
    print("Evaluation focuses on rounds where C actually intervenes.")
    print()

    print_intervention_summary(rows, "ALL")
    print_intervention_summary(rows[-200:], "LAST200")
    print_intervention_summary(rows[-100:], "LAST100")

    cut = int(len(rows) * 0.60)
    print_intervention_summary(rows[:cut], "EARLY60")
    print_intervention_summary(rows[cut:], "LATE40")

    print("RECENT INTERVENTIONS")
    recent = res[res["intervened"]].tail(20)
    if len(recent):
        cols = [
            "round", "decision", "out", "in",
            "baseline_hit", "audited_hit", "delta_hit",
            "baseline", "audited",
        ]
        print(recent[cols].to_string(index=False))
    else:
        print("none")

    print()
    print("KEEP COUNTS")
    print(res["decision"].value_counts().to_string())
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
