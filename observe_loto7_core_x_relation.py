from __future__ import annotations

from pathlib import Path

import pandas as pd

import lotocore
import x_agent

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_core_x_relation.csv")
WINDOW = 100


def actual_numbers(row) -> set[int]:
    return {int(row[f"n{i}"]) for i in range(1, 8)}


def rate(hits: int, slots: int) -> float:
    return hits / slots if slots else float("nan")


def run() -> pd.DataFrame:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(WINDOW, len(df)):
        history = df.iloc[i - WINDOW : i]
        row = df.iloc[i]
        actual = actual_numbers(row)

        core = set(lotocore.predict(history).numbers)
        x = set(x_agent.predict(history, competition_gate=True).numbers)

        both = core & x
        core_only = core - x
        x_only = x - core
        union = core | x

        groups = {
            "both": both,
            "core_only": core_only,
            "x_only": x_only,
            "union": union,
        }

        rec = {
            "round": int(row["round"]),
            "date": row["date"],
            "actual_in_union": len(actual & union),
            "overlap_size": len(both),
        }
        for name, nums in groups.items():
            rec[f"{name}_size"] = len(nums)
            rec[f"{name}_hits"] = len(nums & actual)
        rows.append(rec)

    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> None:
    print("=== LOTO7 CORE-X RELATION OBSERVATION ===")
    print(f"n={len(res)} window={WINDOW}")

    total_hits = {}
    total_slots = {}
    for name in ("both", "core_only", "x_only"):
        hits = int(res[f"{name}_hits"].sum())
        slots = int(res[f"{name}_size"].sum())
        total_hits[name] = hits
        total_slots[name] = slots
        print(
            f"{name}: slots={slots} hits={hits} "
            f"per_candidate_hit_rate={rate(hits, slots):.6f}"
        )

    print(
        "actual_in_union_mean="
        f"{res.actual_in_union.mean():.4f} "
        f"dist={res.actual_in_union.value_counts().sort_index().to_dict()}"
    )
    print(
        "overlap_size_mean="
        f"{res.overlap_size.mean():.4f} "
        f"dist={res.overlap_size.value_counts().sort_index().to_dict()}"
    )

    # Pure Observe/Diff: no causal or predictive interpretation here.
    best = max(
        ("both", "core_only", "x_only"),
        key=lambda n: rate(total_hits[n], total_slots[n]),
    )
    print(f"highest_observed_per_candidate_rate={best}")
    print("WHY_NOT_ANALYZED=1")
    print("LOTO7_CORE_X_RELATION_OBSERVATION_COMPLETE")


def main() -> None:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    summarize(res)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
