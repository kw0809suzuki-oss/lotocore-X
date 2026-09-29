from __future__ import annotations

from pathlib import Path
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_transition_recurrence_v0.csv")


def draw(row):
    return frozenset(int(row[f"n{i}"]) for i in range(1, 8))


def transition(a, b):
    return {
        "stay": frozenset(a & b),
        "exit": frozenset(a - b),
        "enter": frozenset(b - a),
    }


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def transition_similarity(x, y):
    # Whole-transition comparison. No fitted weights or threshold.
    return (
        jaccard(x["stay"], y["stay"])
        + jaccard(x["exit"], y["exit"])
        + jaccard(x["enter"], y["enter"])
    ) / 3.0


def fmt(s):
    return "-".join(f"{n:02d}" for n in sorted(s))


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = [draw(r) for _, r in df.iterrows()]
    transitions = [transition(draws[i], draws[i + 1]) for i in range(len(draws) - 1)]

    rows = []

    # Current transition Ti must have a known successor Ti+1.
    # Match only against strictly earlier transitions Tj whose successor Tj+1
    # was also already disclosed before Ti.
    for i in range(2, len(transitions) - 1):
        eligible = list(range(0, i - 1))
        if not eligible:
            continue

        sims = [(transition_similarity(transitions[i], transitions[j]), j) for j in eligible]
        best_sim, j = max(sims, key=lambda x: (x[0], -x[1]))

        observed_next_sim = transition_similarity(transitions[i + 1], transitions[j + 1])

        control_next_sims = [
            transition_similarity(transitions[i + 1], transitions[k + 1])
            for k in eligible
        ]
        control_mean = statistics.mean(control_next_sims)
        control_median = statistics.median(control_next_sims)

        rows.append({
            "current_to_round": int(df.iloc[i + 1]["round"]),
            "matched_to_round": int(df.iloc[j + 1]["round"]),
            "current_transition_similarity": best_sim,
            "next_transition_similarity": observed_next_sim,
            "control_next_similarity_mean": control_mean,
            "control_next_similarity_median": control_median,
            "next_minus_control_mean": observed_next_sim - control_mean,
            "next_minus_control_median": observed_next_sim - control_median,
            "current_stay": fmt(transitions[i]["stay"]),
            "current_exit": fmt(transitions[i]["exit"]),
            "current_enter": fmt(transitions[i]["enter"]),
            "matched_stay": fmt(transitions[j]["stay"]),
            "matched_exit": fmt(transitions[j]["exit"]),
            "matched_enter": fmt(transitions[j]["enter"]),
            "next_stay": fmt(transitions[i + 1]["stay"]),
            "next_exit": fmt(transitions[i + 1]["exit"]),
            "next_enter": fmt(transitions[i + 1]["enter"]),
            "matched_next_stay": fmt(transitions[j + 1]["stay"]),
            "matched_next_exit": fmt(transitions[j + 1]["exit"]),
            "matched_next_enter": fmt(transitions[j + 1]["enter"]),
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    d = res["next_minus_control_mean"]
    positive = int((d > 0).sum())
    same = int((d == 0).sum())
    negative = int((d < 0).sum())

    print("=== LOTO7 TRANSITION RECURRENCE v0 ===")
    print("Question: when a disclosed whole-transition resembles an earlier whole-transition, does the following transition also resemble the earlier successor?")
    print("Transition is preserved as STAY / EXIT / ENTER sets.")
    print("Similarity = unweighted mean Jaccard(STAY, EXIT, ENTER).")
    print("No threshold search, fitted weights, selector, or prediction rule.")
    print()
    print(f"draws={len(draws)} transitions={len(transitions)} evaluated={len(res)}")
    print(f"mean matched current similarity={res['current_transition_similarity'].mean():.6f}")
    print(f"mean observed successor similarity={res['next_transition_similarity'].mean():.6f}")
    print(f"mean control successor similarity={res['control_next_similarity_mean'].mean():.6f}")
    print(f"mean successor lift={d.mean():+.6f}")
    print(f"median successor lift={d.median():+.6f}")
    print(f"positive/same/negative={positive}/{same}/{negative}")
    print()

    # Simple stability split, descriptive only.
    cut = int(len(res) * 0.60)
    for label, part in [("EARLY60", res.iloc[:cut]), ("LATE40", res.iloc[cut:])]:
        pdiff = part["next_minus_control_mean"]
        print(f"--- {label} n={len(part)} ---")
        print(f"mean matched current similarity={part['current_transition_similarity'].mean():.6f}")
        print(f"mean observed successor similarity={part['next_transition_similarity'].mean():.6f}")
        print(f"mean control successor similarity={part['control_next_similarity_mean'].mean():.6f}")
        print(f"mean successor lift={pdiff.mean():+.6f}")
        print(f"positive/same/negative={(pdiff>0).sum()}/{(pdiff==0).sum()}/{(pdiff<0).sum()}")
        print()

    print("RECENT 20")
    cols = [
        "current_to_round", "matched_to_round",
        "current_transition_similarity",
        "next_transition_similarity",
        "control_next_similarity_mean",
        "next_minus_control_mean",
    ]
    print(res[cols].tail(20).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
