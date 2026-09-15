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
        groups = {
            "both": core & x,
            "core_only": core - x,
            "x_only": x - core,
            "union": core | x,
        }
        rec = {
            "round": int(row["round"]),
            "date": row["date"],
            "actual_in_union": len(actual & groups["union"]),
            "overlap_size": len(groups["both"]),
        }
        for name, nums in groups.items():
            rec[f"{name}_size"] = len(nums)
            rec[f"{name}_hits"] = len(nums & actual)
        rows.append(rec)
    return pd.DataFrame(rows)


def group_rates(part: pd.DataFrame) -> dict[str, float]:
    return {
        name: rate(int(part[f"{name}_hits"].sum()), int(part[f"{name}_size"].sum()))
        for name in ("both", "core_only", "x_only")
    }


def print_segment(label: str, part: pd.DataFrame) -> None:
    rates = group_rates(part)
    residue_vs_core = rates["both"] - rates["core_only"]
    residue_vs_x = rates["both"] - rates["x_only"]
    print(
        f"SEGMENT {label}: n={len(part)} rounds={int(part.iloc[0].round)}..{int(part.iloc[-1].round)} "
        f"both={rates['both']:.6f} core_only={rates['core_only']:.6f} x_only={rates['x_only']:.6f} "
        f"residue_vs_core={residue_vs_core:+.6f} residue_vs_x={residue_vs_x:+.6f}"
    )


def summarize(res: pd.DataFrame) -> None:
    print("=== LOTO7 CORE-X RELATION OBSERVATION ===")
    print(f"n={len(res)} window={WINDOW}")
    rates = group_rates(res)
    for name in ("both", "core_only", "x_only"):
        hits = int(res[f"{name}_hits"].sum())
        slots = int(res[f"{name}_size"].sum())
        print(f"{name}: slots={slots} hits={hits} per_candidate_hit_rate={rates[name]:.6f}")

    print("=== TEMPORAL RESIDUE CHECK ===")
    mid = len(res) // 2
    print_segment("first_half", res.iloc[:mid])
    print_segment("second_half", res.iloc[mid:])

    # Six chronological blocks: enough to see whether the sign persists without
    # inventing a tuned threshold or selecting favorable periods afterward.
    for idx, part in enumerate(pd.np.array_split(res, 6) if hasattr(pd, 'np') else [], start=1):
        print_segment(f"block_{idx}", part)

    # pandas 3 removed pd.np; use positional blocks explicitly.
    import numpy as np
    for idx, indices in enumerate(np.array_split(range(len(res)), 6), start=1):
        part = res.iloc[list(indices)]
        print_segment(f"sixth_{idx}", part)

    print(
        "actual_in_union_mean="
        f"{res.actual_in_union.mean():.4f} dist={res.actual_in_union.value_counts().sort_index().to_dict()}"
    )
    print(
        "overlap_size_mean="
        f"{res.overlap_size.mean():.4f} dist={res.overlap_size.value_counts().sort_index().to_dict()}"
    )
    print("WHY_NOT_ANALYZED=1")
    print("LOTO7_CORE_X_RELATION_TEMPORAL_OBSERVATION_COMPLETE")


def main() -> None:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    summarize(res)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
