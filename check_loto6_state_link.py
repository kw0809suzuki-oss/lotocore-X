from __future__ import annotations

import argparse
import random
from math import sqrt
from pathlib import Path

import pandas as pd

import loto6_core

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_state_link.csv")
DRAW_SIZE = 6
STATE_WINDOW = 20


def actual_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)))


def overlap(a, b) -> int:
    return len(set(a) & set(b))


def run(window: int = 100, analogs: int = 12) -> pd.DataFrame:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    if len(df) <= window:
        raise ValueError(f"need more than {window} draws, got {len(df)}")

    rows = []
    for i in range(window, len(df)):
        history = df.iloc[i - window : i]
        draws = loto6_core._draws(history)
        actual = actual_numbers(df.iloc[i])
        round_no = int(df.iloc[i]["round"])

        current_center, current_amplitude = loto6_core._macro_state(draws[-STATE_WINDOW:])

        examples = []
        for end in range(STATE_WINDOW, len(draws)):
            block = draws[end - STATE_WINDOW : end]
            center, amplitude = loto6_core._macro_state(block)
            distance = sqrt(
                ((center - current_center) / 5.0) ** 2
                + ((amplitude - current_amplitude) / 3.0) ** 2
            )
            examples.append((distance, draws[end]))

        examples.sort(key=lambda x: x[0])
        k = min(analogs, len(examples))
        nearest = examples[:k]

        rng = random.Random(round_no)
        random_examples = rng.sample(examples, k)

        nearest_overlap = sum(overlap(next_draw, actual) for _, next_draw in nearest) / k
        random_overlap = sum(overlap(next_draw, actual) for _, next_draw in random_examples) / k

        rows.append(
            {
                "round": round_no,
                "date": df.iloc[i]["date"],
                "state_center": round(current_center, 4),
                "state_amplitude": round(current_amplitude, 4),
                "nearest_mean_overlap": nearest_overlap,
                "random_mean_overlap": random_overlap,
                "lift": nearest_overlap - random_overlap,
            }
        )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    n = len(results)
    near = results.nearest_mean_overlap.mean()
    rnd = results.random_mean_overlap.mean()
    print("=== LOTO6 STATE-LINK CHECK ===")
    print(f"rounds={n}")
    print(f"nearest12_mean_overlap={near:.4f}")
    print(f"random12_mean_overlap={rnd:.4f}")
    print(f"lift={near - rnd:+.4f}")
    print(
        "PAIRED: "
        f"NEAR>R={(results.lift > 0).sum()} "
        f"NEAR=R={(results.lift == 0).sum()} "
        f"NEAR<R={(results.lift < 0).sum()}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--analogs", type=int, default=12)
    args = p.parse_args()

    results = run(args.window, args.analogs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    summarize(results)
    print(f"saved -> {OUT}")
    print("LOTO6_STATE_LINK_COMPLETE")


if __name__ == "__main__":
    main()
