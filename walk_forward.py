from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import pandas as pd

import lotocore
import x_agent

DATA = Path("data/loto7.csv")
OUT = Path("results/latest.csv")
NUMBERS = list(range(1, 38))


def actual_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def metrics(pred: tuple[int, ...], actual: tuple[int, ...]) -> dict:
    hits = len(set(pred) & set(actual))
    pc = sum(pred) / 7.0
    ac = sum(actual) / 7.0
    pv = sum((x - pc) ** 2 for x in pred) / 7.0
    av = sum((x - ac) ** 2 for x in actual) / 7.0
    return {
        "hits": hits,
        "center_error": abs(pc - ac),
        "variance_error": abs(pv - av),
    }


def random_prediction(round_no: int) -> tuple[int, ...]:
    rng = random.Random(round_no)
    return tuple(sorted(rng.sample(NUMBERS, 7)))


def run(window: int = 100) -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit("data/loto7.csv not found; run fetch_loto7.py first")

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    if len(df) <= window:
        raise SystemExit(f"need more than {window} draws; found {len(df)}")

    rows = []
    for i in range(window, len(df)):
        history = df.iloc[i-window:i]
        actual = actual_numbers(df.iloc[i])
        rnd = int(df.iloc[i]["round"])

        predictions = {
            "lotocore": lotocore.predict(history),
            "x": x_agent.predict(history),
            "random": None,
        }

        for model, pred_obj in predictions.items():
            if model == "random":
                pred = random_prediction(rnd)
                state = {"model": "random"}
            else:
                pred = pred_obj.numbers
                state = pred_obj.state

            m = metrics(pred, actual)
            row = {
                "round": rnd,
                "date": df.iloc[i]["date"],
                "model": model,
                "prediction": "-".join(map(str, pred)),
                "actual": "-".join(map(str, actual)),
                **m,
            }
            for k, v in state.items():
                if k != "model":
                    row[k] = v
            rows.append(row)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    print("\n=== WALK FORWARD SUMMARY ===")
    for model in ("lotocore", "x", "random"):
        g = results[results.model == model]
        dist = g["hits"].value_counts().sort_index().to_dict()
        print(
            f"{model:8s} n={len(g)} mean_hits={g.hits.mean():.4f} "
            f"hit3+={(g.hits >= 3).mean():.4f} best={g.hits.max()} "
            f"mean_center_error={g.center_error.mean():.4f} "
            f"mean_variance_error={g.variance_error.mean():.4f} "
            f"hit_distribution={dist}"
        )

    # Paired round-by-round comparison; more meaningful than raw totals alone.
    pivot = results.pivot(index="round", columns="model", values="hits")
    if {"x", "lotocore", "random"}.issubset(pivot.columns):
        print(
            "PAIRED "
            f"X>LotoCore={(pivot.x > pivot.lotocore).sum()} "
            f"X=LotoCore={(pivot.x == pivot.lotocore).sum()} "
            f"X<LotoCore={(pivot.x < pivot.lotocore).sum()} | "
            f"X>Random={(pivot.x > pivot.random).sum()} "
            f"X=Random={(pivot.x == pivot.random).sum()} "
            f"X<Random={(pivot.x < pivot.random).sum()}"
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
    print("LOTOCORE_X_WALK_FORWARD_COMPLETE")


if __name__ == "__main__":
    main()
