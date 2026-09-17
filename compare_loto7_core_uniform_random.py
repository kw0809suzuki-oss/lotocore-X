"""Direct walk-forward comparison: LOTO Core vs ordinary uniform-random tickets.

Question: does the existing LOTO7 model's final 7-number ticket behave differently
from simply choosing 7 unique numbers uniformly from 1..37?

At every target round, the model sees only the preceding window. Random sees no
future information. To reduce Monte Carlo noise, RANDOM_REPS independent uniform
random tickets are generated per target round and their hit distribution is pooled.
No thresholds or model parameters are tuned from target results.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_core_vs_uniform_random.csv")
RANDOM_REPS = 1000


def actual_numbers(row):
    return set(int(row[f"n{i}"]) for i in range(1, 8))


def hits(ticket, actual):
    return len(set(ticket) & actual)


def run(window=100, reps=RANDOM_REPS):
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []
    random_hist = Counter()
    core_hist = Counter()
    for i in range(window, len(df)):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        core_ticket = tuple(lotocore.predict(history).numbers)
        core_hit = hits(core_ticket, actual)
        core_hist[core_hit] += 1

        rng = random.Random(rnd * 1000003 + 20260917)
        rhits = []
        for _ in range(reps):
            t = rng.sample(range(1, 38), 7)
            h = hits(t, actual)
            random_hist[h] += 1
            rhits.append(h)

        rows.append({
            "round": rnd,
            "date": row.get("date", ""),
            "core_hits": core_hit,
            "random_mean_hits": sum(rhits) / len(rhits),
            "random_p_3plus": sum(h >= 3 for h in rhits) / len(rhits),
            "random_p_4plus": sum(h >= 4 for h in rhits) / len(rhits),
            "random_p_5plus": sum(h >= 5 for h in rhits) / len(rhits),
        })
    return pd.DataFrame(rows), core_hist, random_hist


def rate(hist, threshold):
    n = sum(hist.values())
    return sum(v for k, v in hist.items() if k >= threshold) / n if n else 0.0


def mean_hits(hist):
    n = sum(hist.values())
    return sum(k*v for k,v in hist.items()) / n if n else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--random-reps", type=int, default=RANDOM_REPS)
    args = p.parse_args()
    res, core, rnd = run(args.window, args.random_reps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("LOTO7 CORE vs UNIFORM RANDOM / WALK-FORWARD")
    print(f"target_rounds={len(res)} history_window={args.window} random_reps_per_round={args.random_reps}")
    print(f"CORE   mean_hits={mean_hits(core):.4f} 3+={rate(core,3):.4f} 4+={rate(core,4):.4f} 5+={rate(core,5):.4f} 6+={rate(core,6):.4f} 7={rate(core,7):.4f} dist={dict(sorted(core.items()))}")
    print(f"RANDOM mean_hits={mean_hits(rnd):.4f} 3+={rate(rnd,3):.4f} 4+={rate(rnd,4):.4f} 5+={rate(rnd,5):.4f} 6+={rate(rnd,6):.4f} 7={rate(rnd,7):.4f} dist={dict(sorted(rnd.items()))}")
    print("NOTE: backtest observation only; not evidence of future lottery advantage by itself.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
