from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd

NUMBERS = list(range(1, 44))
PICK = 6
SELF_WINDOW = 16
CANDIDATES = 24
NUDGE = 0.05  # intentionally weak; direction is never prescribed


def draw_uniform(rng: random.Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(NUMBERS, PICK)))


def spread(draw: tuple[int, ...]) -> float:
    c = sum(draw) / PICK
    return math.sqrt(sum((x - c) ** 2 for x in draw) / PICK)


def atmosphere(history: list[tuple[int, ...]]) -> dict:
    recent = history[-SELF_WINDOW:]
    if not recent:
        return {"mode": "free", "target_spread": None}
    spreads = [spread(d) for d in recent]
    recent_mean = sum(spreads) / len(spreads)
    if len(recent) < 6:
        mode = "free"
    else:
        mid = sorted(spreads)[len(spreads) // 2]
        mode = "active" if recent_mean >= mid else "quiet"
    return {"mode": mode, "target_spread": recent_mean}


def draw_random_kun(rng: random.Random, history: list[tuple[int, ...]]) -> tuple[tuple[int, ...], dict]:
    """Uniform Random is the body; self-observation only weakly changes wobble.

    We never encode up/down, hot/cold numbers, prediction, or mean reversion.
    A uniform draw is retained with 95% probability. In the remaining 5%,
    several uniform candidates are drawn and one with a spread compatible with
    the current self-observed atmosphere is selected. Direction stays free.
    """
    state = atmosphere(history)
    base = draw_uniform(rng)
    if not history or rng.random() >= NUDGE:
        return base, {**state, "nudged": 0}

    pool = [base] + [draw_uniform(rng) for _ in range(CANDIDATES - 1)]
    target = state["target_spread"]
    chosen = min(pool, key=lambda d: abs(spread(d) - target))
    return chosen, {**state, "nudged": 1}


def hits(pred: tuple[int, ...], actual: tuple[int, ...]) -> int:
    return len(set(pred) & set(actual))


def run(seed: int = 20260913, warmup: int = 20) -> pd.DataFrame:
    df = pd.read_csv("data/loto6.csv").sort_values("round").reset_index(drop=True)
    rng = random.Random(seed)
    self_history: list[tuple[int, ...]] = []
    rows = []
    for _, row in df.iterrows():
        actual = tuple(sorted(int(row[f"n{i}"]) for i in range(1, 7)))
        pure = draw_uniform(rng)
        rk, state = draw_random_kun(rng, self_history)
        self_history.append(rk)
        if len(self_history) <= warmup:
            continue
        rows.append({
            "round": int(row["round"]), "date": row.get("date", ""),
            "random_kun": "-".join(map(str, rk)), "pure_random": "-".join(map(str, pure)),
            "actual": "-".join(map(str, actual)),
            "random_kun_hits": hits(rk, actual), "pure_random_hits": hits(pure, actual),
            "mode": state["mode"], "nudged": state["nudged"],
        })
    return pd.DataFrame(rows)


def summarize(out: pd.DataFrame) -> None:
    print("=== RANDOM-KUN LOTO6 ===")
    print("body=uniform random | self-window=16 | nudge=5% | direction=free")
    for col in ("random_kun_hits", "pure_random_hits"):
        x = out[col]
        print(f"{col}: n={len(x)} mean={x.mean():.4f} hit2+={(x>=2).mean():.4f} hit3+={(x>=3).mean():.4f} best={x.max()} dist={dict(sorted(Counter(x).items()))}")
    a, b = out.random_kun_hits, out.pure_random_hits
    print(f"PAIRED RK>PURE={(a>b).sum()} RK=PURE={(a==b).sum()} RK<PURE={(a<b).sum()}")
    print(f"NUDGE fires={int(out.nudged.sum())}/{len(out)} rate={out.nudged.mean():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=20260913); args = p.parse_args()
    out = run(args.seed)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/random_kun_loto6.csv", index=False)
    summarize(out)
    print("saved -> results/random_kun_loto6.csv")
    print("RANDOM_KUN_COMPLETE")


if __name__ == "__main__":
    main()
