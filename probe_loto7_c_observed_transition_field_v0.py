from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, NUMBERS,
    draws_from_df, frequency_distribution, mix_distribution, rank_numbers,
)

OUT = Path("results/loto7_c_observed_transition_field_v0.csv")
RECENT_TRANSITIONS = 20


def truth(row):
    return {int(row[f"n{i}"]) for i in range(1, 8)}


def transition_features(history, n, recent_k=RECENT_TRANSITIONS):
    # history contains only disclosed draws up to the prediction boundary.
    bits = [1 if n in d else 0 for d in history]
    pairs = list(zip(bits[:-1], bits[1:]))

    def block_features(ps):
        prev_present = sum(a == 1 for a, _ in ps)
        prev_absent = sum(a == 0 for a, _ in ps)
        enters = sum(a == 0 and b == 1 for a, b in ps)
        exits = sum(a == 1 and b == 0 for a, b in ps)
        stays = sum(a == 1 and b == 1 for a, b in ps)
        absent_stays = sum(a == 0 and b == 0 for a, b in ps)
        return {
            "enter_rate": enters / prev_absent if prev_absent else 0.0,
            "exit_rate": exits / prev_present if prev_present else 0.0,
            "stay_rate": stays / prev_present if prev_present else 0.0,
            "enter_count": enters,
            "exit_count": exits,
            "stay_count": stays,
            "absent_stay_count": absent_stays,
        }

    life = block_features(pairs)
    recent = block_features(pairs[-recent_k:])

    current_present = bits[-1]
    last_transition = 0
    if len(bits) >= 2:
        last_transition = bits[-1] - bits[-2]  # -1 exit, 0 no change, +1 enter

    run = 1
    for j in range(len(bits) - 2, -1, -1):
        if bits[j] == bits[-1]:
            run += 1
        else:
            break
    signed_run = run if current_present else -run

    return {
        "current_present": current_present,
        "last_transition": last_transition,
        "signed_run": signed_run,
        "life_enter_rate": life["enter_rate"],
        "life_exit_rate": life["exit_rate"],
        "life_stay_rate": life["stay_rate"],
        "recent_enter_rate": recent["enter_rate"],
        "recent_exit_rate": recent["exit_rate"],
        "recent_stay_rate": recent["stay_rate"],
    }


def next_motion(current_draw, next_draw, n):
    a = 1 if n in current_draw else 0
    b = 1 if n in next_draw else 0
    if a == 0 and b == 1:
        return "ENTER"
    if a == 1 and b == 1:
        return "STAY"
    if a == 1 and b == 0:
        return "EXIT"
    return "ABSENT"


FEATURES = [
    "current_present",
    "last_transition",
    "signed_run",
    "life_enter_rate",
    "life_exit_rate",
    "life_stay_rate",
    "recent_enter_rate",
    "recent_exit_rate",
    "recent_stay_rate",
]


def print_group_means(df, label_col, labels, title):
    print(f"=== {title} ===")
    for lab in labels:
        g = df[df[label_col] == lab]
        print(f"--- {lab} n={len(g)} ---")
        if g.empty:
            print("none")
            continue
        for c in FEATURES:
            print(f"{c}: mean={g[c].mean():+.6f} median={g[c].median():+.6f}")
        print()


def print_pair_diffs(df, mask_a, mask_b, name_a, name_b, title):
    a = df[mask_a]
    b = df[mask_b]
    print(f"=== {title} | {name_a} minus {name_b} ===")
    print(f"{name_a} n={len(a)} | {name_b} n={len(b)}")
    for c in FEATURES:
        print(f"{c}: {a[c].mean() - b[c].mean():+.6f}")
    print()


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = [set(d) for d in draws_from_df(df)]
    rows = []

    # Prediction target is draw i. Everything used by C is frozen at history[:i].
    for i in range(LONG_WINDOW, len(draws)):
        history = draws[:i]
        current_draw = history[-1]
        target_draw = draws[i]

        lp = frequency_distribution([list(d) for d in draws[i - LONG_WINDOW:i]])
        rp = frequency_distribution([list(d) for d in draws[i - RECENT_WINDOW:i]])
        p = mix_distribution(lp, rp)
        baseline = set(rank_numbers(p)[:7])

        for n in NUMBERS:
            feat = transition_features(history, n)
            target_hit = n in target_draw
            in_baseline = n in baseline
            rows.append({
                "target_round": int(df.iloc[i]["round"]),
                "number": n,
                "in_baseline": in_baseline,
                "target_hit": target_hit,
                "baseline_role": (
                    "BASELINE_HIT" if in_baseline and target_hit else
                    "BASELINE_MISS" if in_baseline else
                    "OUTSIDE_HIT" if target_hit else
                    "OUTSIDE_MISS"
                ),
                "next_motion": next_motion(current_draw, target_draw, n),
                **feat,
            })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 C | OBSERVED TRANSITION FIELD v0 ===")
    print("C only observes transitions already disclosed before each target round.")
    print("No score, threshold, selector, KEEP/SWAP rule, or future result enters feature construction.")
    print(f"target_rounds={res['target_round'].nunique()} rows={len(res)}")
    print()

    print_group_means(
        res, "next_motion", ["ENTER", "STAY", "EXIT", "ABSENT"],
        "WORLD NEXT-MOTION LABELS (outcome used only after freeze)"
    )

    print_pair_diffs(
        res,
        res["next_motion"] == "ENTER",
        res["next_motion"] == "ABSENT",
        "ENTER", "ABSENT",
        "Can prior observed transitions distinguish future inflow from continued absence?"
    )
    print_pair_diffs(
        res,
        res["next_motion"] == "STAY",
        res["next_motion"] == "EXIT",
        "STAY", "EXIT",
        "Can prior observed transitions distinguish persistence from exit?"
    )

    print_group_means(
        res, "baseline_role",
        ["BASELINE_HIT", "BASELINE_MISS", "OUTSIDE_HIT", "OUTSIDE_MISS"],
        "BASELINE OVERLAY"
    )
    print_pair_diffs(
        res,
        res["baseline_role"] == "BASELINE_HIT",
        res["baseline_role"] == "BASELINE_MISS",
        "BASELINE_HIT", "BASELINE_MISS",
        "Inside Baseline: retained-value observation"
    )
    print_pair_diffs(
        res,
        res["baseline_role"] == "OUTSIDE_HIT",
        res["baseline_role"] == "OUTSIDE_MISS",
        "OUTSIDE_HIT", "OUTSIDE_MISS",
        "Outside Baseline: pickup-value observation"
    )

    # Stability view only: late 40% vs early 60%, still no fitting.
    rounds = sorted(res["target_round"].unique())
    cut_round = rounds[int(len(rounds) * 0.60)]
    for tag, part in [
        ("EARLY60", res[res["target_round"] < cut_round]),
        ("LATE40", res[res["target_round"] >= cut_round]),
    ]:
        print(f"=== {tag} STABILITY VIEW ===")
        for a_name, b_name in [("BASELINE_HIT", "BASELINE_MISS"), ("OUTSIDE_HIT", "OUTSIDE_MISS")]:
            a = part[part["baseline_role"] == a_name]
            b = part[part["baseline_role"] == b_name]
            print(f"{a_name} minus {b_name} | n={len(a)} vs {len(b)}")
            for c in FEATURES:
                print(f"{c}: {a[c].mean() - b[c].mean():+.6f}")
            print()

    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
