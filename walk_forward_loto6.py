from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

import loto6_core
import loto6_x_agent

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_compare.csv")
NUMBERS = list(range(1, 44))
DRAW_SIZE = 6


def actual_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)))


def metrics(pred: tuple[int, ...], actual: tuple[int, ...]) -> dict:
    hits = len(set(pred) & set(actual))
    pc = sum(pred) / DRAW_SIZE
    ac = sum(actual) / DRAW_SIZE
    pv = sum((x - pc) ** 2 for x in pred) / DRAW_SIZE
    av = sum((x - ac) ** 2 for x in actual) / DRAW_SIZE
    return {
        "hits": hits,
        "center_error": abs(pc - ac),
        "variance_error": abs(pv - av),
    }


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
            "core": loto6_core.predict(history),
            "x": loto6_x_agent.predict(history),
        }

        for model, pred_obj in predictions.items():
            pred = pred_obj.numbers
            state = pred_obj.state
            row = {
                "round": rnd,
                "date": df.iloc[i]["date"],
                "model": model,
                "prediction": "-".join(map(str, pred)),
                "actual": "-".join(map(str, actual)),
                **metrics(pred, actual),
            }
            row.update({k: v for k, v in state.items() if k != "model"})
            rows.append(row)

        pred = random_prediction(rnd)
        rows.append({
            "round": rnd,
            "date": df.iloc[i]["date"],
            "model": "random",
            "prediction": "-".join(map(str, pred)),
            "actual": "-".join(map(str, actual)),
            **metrics(pred, actual),
        })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    print("=== LOTO6 CORE / X / RANDOM WALK-FORWARD ===")
    for model in ("core", "x", "random"):
        g = results[results.model == model]
        print(
            f"{model:6s} n={len(g)} mean_hits={g.hits.mean():.4f} "
            f"hit3+={(g.hits >= 3).mean():.4f} best={int(g.hits.max())} "
            f"center_error={g.center_error.mean():.4f} "
            f"variance_error={g.variance_error.mean():.4f} "
            f"dist={g.hits.value_counts().sort_index().to_dict()}"
        )

    pivot = results.pivot(index="round", columns="model", values="hits")
    print(
        "PAIRED CORE vs X: "
        f"X>CORE={(pivot.x > pivot.core).sum()} "
        f"X=CORE={(pivot.x == pivot.core).sum()} "
        f"X<CORE={(pivot.x < pivot.core).sum()}"
    )
    print(
        "PAIRED X vs RANDOM: "
        f"X>R={(pivot.x > pivot.random).sum()} "
        f"X=R={(pivot.x == pivot.random).sum()} "
        f"X<R={(pivot.x < pivot.random).sum()}"
    )
    print(
        "PAIRED CORE vs RANDOM: "
        f"CORE>R={(pivot.core > pivot.random).sum()} "
        f"CORE=R={(pivot.core == pivot.random).sum()} "
        f"CORE<R={(pivot.core < pivot.random).sum()}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()
    results = run(args.window)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    summarize(results)
    print(f"saved -> {OUT}")
    print("LOTO6_COMPARE_COMPLETE")


if __name__ == "__main__":
    main()
