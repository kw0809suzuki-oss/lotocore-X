from __future__ import annotations

from pathlib import Path
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_transition_successor_number_transfer_v0.csv")


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
    return (
        jaccard(x["stay"], y["stay"])
        + jaccard(x["exit"], y["exit"])
        + jaccard(x["enter"], y["enter"])
    ) / 3.0


def hit(a, b):
    return len(set(a) & set(b))


def fmt(s):
    return "-".join(f"{n:02d}" for n in sorted(s))


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = [draw(r) for _, r in df.iterrows()]
    transitions = [transition(draws[i], draws[i + 1]) for i in range(len(draws) - 1)]

    rows = []

    # Ti = draw[i] -> draw[i+1]
    # We find an earlier similar Tj. Its already-disclosed successor Tj+1
    # ends at draw[j+2], which becomes the transferred 7-number candidate.
    # Target is draw[i+2], revealed only after the candidate is frozen.
    for i in range(2, len(transitions) - 1):
        eligible = list(range(0, i - 1))
        if not eligible:
            continue

        sims = [(transition_similarity(transitions[i], transitions[j]), j) for j in eligible]
        best_sim, j = max(sims, key=lambda x: (x[0], -x[1]))

        transferred = draws[j + 2]
        target = draws[i + 2]
        current_endpoint = draws[i + 1]

        transferred_hit = hit(transferred, target)
        carry_hit = hit(current_endpoint, target)

        control_hits = [hit(draws[k + 2], target) for k in eligible]
        control_mean = statistics.mean(control_hits)
        control_median = statistics.median(control_hits)

        # Decompose the transferred candidate into the matched successor's
        # observed STAY and ENTER components, preserving the whole transition.
        matched_successor = transitions[j + 1]
        stay_hits = hit(matched_successor["stay"], target)
        enter_hits = hit(matched_successor["enter"], target)

        rows.append({
            "target_round": int(df.iloc[i + 2]["round"]),
            "current_to_round": int(df.iloc[i + 1]["round"]),
            "matched_to_round": int(df.iloc[j + 1]["round"]),
            "current_transition_similarity": best_sim,
            "transferred_candidate": fmt(transferred),
            "actual_target": fmt(target),
            "transferred_hit": transferred_hit,
            "carry_forward_hit": carry_hit,
            "control_hit_mean": control_mean,
            "control_hit_median": control_median,
            "transfer_minus_control_mean": transferred_hit - control_mean,
            "transfer_minus_carry": transferred_hit - carry_hit,
            "matched_successor_stay": fmt(matched_successor["stay"]),
            "matched_successor_enter": fmt(matched_successor["enter"]),
            "stay_component_size": len(matched_successor["stay"]),
            "enter_component_size": len(matched_successor["enter"]),
            "stay_component_hits": stay_hits,
            "enter_component_hits": enter_hits,
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    d = res["transfer_minus_control_mean"]
    dc = res["transfer_minus_carry"]

    print("=== LOTO7 TRANSITION SUCCESSOR -> NUMBER TRANSFER v0 ===")
    print("Question: if Ti resembles earlier Tj, do the 7 numbers at the end of Tj+1 transfer to the actual next draw after Ti?")
    print("Candidate = STAY + ENTER of the matched earlier successor transition.")
    print("No score fitting, threshold search, selector, or combination with Baseline.")
    print()
    print(f"evaluated={len(res)}")
    print(f"mean matched-transition similarity={res['current_transition_similarity'].mean():.6f}")
    print(f"mean transferred hit={res['transferred_hit'].mean():.6f}")
    print(f"mean control hit={res['control_hit_mean'].mean():.6f}")
    print(f"mean transfer lift vs control={d.mean():+.6f}")
    print(f"median transfer lift vs control={d.median():+.6f}")
    print(f"positive/same/negative vs control={(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}")
    print()
    print(f"mean carry-forward hit={res['carry_forward_hit'].mean():.6f}")
    print(f"mean transfer lift vs carry={dc.mean():+.6f}")
    print(f"positive/same/negative vs carry={(dc>0).sum()}/{(dc==0).sum()}/{(dc<0).sum()}")
    print()
    print("HIT DISTRIBUTION | transferred")
    print(res["transferred_hit"].value_counts().sort_index().to_string())
    print()
    print("COMPONENT CONTRIBUTION")
    print(f"mean matched successor STAY size={res['stay_component_size'].mean():.6f}")
    print(f"mean matched successor ENTER size={res['enter_component_size'].mean():.6f}")
    print(f"mean target hits from STAY component={res['stay_component_hits'].mean():.6f}")
    print(f"mean target hits from ENTER component={res['enter_component_hits'].mean():.6f}")
    print()

    cut = int(len(res) * 0.60)
    for label, part in [("EARLY60", res.iloc[:cut]), ("LATE40", res.iloc[cut:])]:
        pdiff = part["transfer_minus_control_mean"]
        pcarry = part["transfer_minus_carry"]
        print(f"--- {label} n={len(part)} ---")
        print(f"mean transferred hit={part['transferred_hit'].mean():.6f}")
        print(f"mean control hit={part['control_hit_mean'].mean():.6f}")
        print(f"mean lift vs control={pdiff.mean():+.6f}")
        print(f"mean carry hit={part['carry_forward_hit'].mean():.6f}")
        print(f"mean lift vs carry={pcarry.mean():+.6f}")
        print(f"3+ rate transferred={(part['transferred_hit']>=3).mean():.6%}")
        print()

    print("RECENT 20")
    cols = [
        "target_round","matched_to_round","current_transition_similarity",
        "transferred_hit","control_hit_mean","transfer_minus_control_mean",
        "carry_forward_hit","transfer_minus_carry",
        "matched_successor_stay","matched_successor_enter","transferred_candidate"
    ]
    print(res[cols].tail(20).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
