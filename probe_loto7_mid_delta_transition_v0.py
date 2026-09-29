from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_mid_delta_transition_v0.csv")


def draw(row):
    return [int(row[f"n{i}"]) for i in range(1, 8)]


def mid_count(xs):
    return sum(13 <= n <= 24 for n in xs)


def sign_label(x):
    if x > 0:
        return "UP"
    if x < 0:
        return "DOWN"
    return "FLAT"


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = [draw(r) for _, r in df.iterrows()]
    mids = [mid_count(d) for d in draws]
    deltas = [mids[i+1] - mids[i] for i in range(len(mids)-1)]

    rows=[]
    for i in range(len(deltas)-1):
        rows.append({
            "from_round": int(df.iloc[i]["round"]),
            "to_round": int(df.iloc[i+1]["round"]),
            "next_round": int(df.iloc[i+2]["round"]),
            "mid_count_from": mids[i],
            "mid_count_to": mids[i+1],
            "delta_mid": deltas[i],
            "delta_mid_label": sign_label(deltas[i]),
            "next_delta_mid": deltas[i+1],
            "next_delta_mid_label": sign_label(deltas[i+1]),
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 MID DELTA TRANSITION v0 ===")
    print("Purpose: observe how MID-band occupancy change at t transitions to MID-band occupancy change at t+1.")
    print("No prediction rule, no thresholds, no number identities.")
    print()

    ctab = pd.crosstab(
        res["delta_mid_label"],
        res["next_delta_mid_label"],
        normalize="index"
    )
    print("SIGN TRANSITION PROBABILITIES")
    print(ctab.to_string())
    print()

    print("RAW COUNTS")
    print(pd.crosstab(res["delta_mid_label"], res["next_delta_mid_label"]).to_string())
    print()

    print("NEXT DELTA MEAN BY CURRENT SIGN")
    for label in ["UP","FLAT","DOWN"]:
        part = res[res["delta_mid_label"] == label]
        print(
            f"{label}: n={len(part)} "
            f"mean_next_delta={part['next_delta_mid'].mean():+.6f} "
            f"median={part['next_delta_mid'].median():+.6f}"
        )
    print()

    print("EXACT DELTA -> NEXT DELTA")
    exact = (
        res.groupby("delta_mid")["next_delta_mid"]
        .agg(["count","mean","median"])
        .sort_index()
    )
    print(exact.to_string())
    print()

    cut = int(len(res) * 0.60)
    for tag, part in [("EARLY60", res.iloc[:cut]), ("LATE40", res.iloc[cut:])]:
        print(f"--- {tag} ---")
        print(pd.crosstab(
            part["delta_mid_label"],
            part["next_delta_mid_label"],
            normalize="index"
        ).to_string())
        print()

    print("RECENT 30")
    print(res[[
        "to_round","mid_count_to","delta_mid_label","delta_mid",
        "next_delta_mid_label","next_delta_mid"
    ]].tail(30).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
