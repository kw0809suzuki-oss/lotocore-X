from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

import loto6_core
import loto6_core_field

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_field_core_compare.csv")
NUMBERS = list(range(1, 44))
DRAW_SIZE = 6


def actual_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)))


def random_prediction(round_no: int) -> tuple[int, ...]:
    return tuple(sorted(random.Random(round_no).sample(NUMBERS, DRAW_SIZE)))


def run(window: int = 100) -> pd.DataFrame:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    if len(df) <= window:
        raise ValueError(f"need more than {window} draws, got {len(df)}")

    rows = []
    for i in range(window, len(df)):
        history = df.iloc[i - window : i]
        actual = actual_numbers(df.iloc[i])
        rnd = int(df.iloc[i]["round"])

        predictions = {
            "analog_core": loto6_core.predict(history).numbers,
            "field_core": loto6_core_field.predict(history).numbers,
            "random": random_prediction(rnd),
        }

        for model, pred in predictions.items():
            hits = len(set(pred) & set(actual))
            rows.append(
                {
                    "round": rnd,
                    "date": df.iloc[i]["date"],
                    "model": model,
                    "prediction": "-".join(map(str, pred)),
                    "actual": "-".join(map(str, actual)),
                    "hits": hits,
                }
            )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    print("=== LOTO6 ANALOG CORE / FIELD CORE / RANDOM ===")
    for model in ("analog_core", "field_core", "random"):
        g = results[results.model == model]
        print(
            f"{model:11s} n={len(g)} mean_hits={g.hits.mean():.4f} "
            f"hit2+={(g.hits >= 2).mean():.4f} "
            f"hit3+={(g.hits >= 3).mean():.4f} "
            f"best={int(g.hits.max())} "
            f"dist={g.hits.value_counts().sort_index().to_dict()}"
        )

    pivot = results.pivot(index="round", columns="model", values="hits")
    print(
        "PAIRED FIELD vs ANALOG: "
        f"FIELD>ANALOG={(pivot.field_core > pivot.analog_core).sum()} "
        f"FIELD=ANALOG={(pivot.field_core == pivot.analog_core).sum()} "
        f"FIELD<ANALOG={(pivot.field_core < pivot.analog_core).sum()}"
    )
    print(
        "PAIRED FIELD vs RANDOM: "
        f"FIELD>R={(pivot.field_core > pivot.random).sum()} "
        f"FIELD=R={(pivot.field_core == pivot.random).sum()} "
        f"FIELD<R={(pivot.field_core < pivot.random).sum()}"
    )


def main() -> None:
    results = run(100)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    summarize(results)
    print(f"saved -> {OUT}")
    print("LOTO6_FIELD_CORE_COMPARE_COMPLETE")


if __name__ == "__main__":
    main()
