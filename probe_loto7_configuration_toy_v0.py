from __future__ import annotations

from pathlib import Path
import random
import statistics

import pandas as pd

import lotocore

DATA = Path("data/loto7.csv")
WINDOW = 100
NULLS = 300
SEED = 20260927
NUMBERS = list(range(1, 38))


def actual(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def shape(nums):
    xs = tuple(sorted(nums))
    lo, hi = xs[0], xs[-1]
    span = hi - lo
    if span <= 0:
        return tuple(0.0 for _ in xs)
    return tuple((x - lo) / span for x in xs)


def gaps(nums):
    xs = tuple(sorted(nums))
    span = xs[-1] - xs[0]
    if span <= 0:
        return (0.0,) * 6
    return tuple((xs[i+1] - xs[i]) / span for i in range(6))


def mae(a, b):
    return sum(abs(x-y) for x, y in zip(a, b)) / len(a)


def fmt(vals):
    return "/".join(f"{v:.2f}" for v in vals)


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(WINDOW, len(df)):
        history = df.iloc[i-WINDOW:i]
        core = tuple(sorted(lotocore.predict(history).numbers))
        truth = actual(df.iloc[i])

        cs, ts = shape(core), shape(truth)
        cg, tg = gaps(core), gaps(truth)
        shape_err = mae(cs, ts)
        gap_err = mae(cg, tg)

        null_shape = []
        null_gap = []
        for _ in range(NULLS):
            r = tuple(sorted(rng.sample(NUMBERS, 7)))
            null_shape.append(mae(shape(r), ts))
            null_gap.append(mae(gaps(r), tg))

        rows.append({
            "round": int(df.iloc[i]["round"]),
            "core_shape_err": shape_err,
            "random_shape_mean": statistics.mean(null_shape),
            "shape_advantage": statistics.mean(null_shape) - shape_err,
            "core_gap_err": gap_err,
            "random_gap_mean": statistics.mean(null_gap),
            "gap_advantage": statistics.mean(null_gap) - gap_err,
            "core": "-".join(f"{x:02d}" for x in core),
            "actual": "-".join(f"{x:02d}" for x in truth),
            "core_shape": fmt(cs),
            "actual_shape": fmt(ts),
            "core_gaps": fmt(cg),
            "actual_gaps": fmt(tg),
        })

    res = pd.DataFrame(rows)

    print("=== LOTO7 CONFIGURATION TOY v0 ===")
    print(f"targets={len(res)} rounds={int(res['round'].min())}..{int(res['round'].max())}")
    print("Remove translation and scale: map each sorted 7-number set to [0,1] using its own min/max.")
    print(f"Random null: {NULLS} random 7-number sets per target.")
    print()
    print(f"shape MAE: CORE={res.core_shape_err.mean():.6f} random={res.random_shape_mean.mean():.6f} advantage={res.shape_advantage.mean():+.6f}")
    print(f"CORE beats random mean (shape)={(res.shape_advantage > 0).mean():.6f}")
    print(f"gap-rhythm MAE: CORE={res.core_gap_err.mean():.6f} random={res.random_gap_mean.mean():.6f} advantage={res.gap_advantage.mean():+.6f}")
    print(f"CORE beats random mean (gap)={(res.gap_advantage > 0).mean():.6f}")
    print()
    print("RECENT 20")
    cols=["round","core","actual","core_shape_err","random_shape_mean","shape_advantage","core_gap_err","random_gap_mean","gap_advantage","core_shape","actual_shape"]
    print(res.tail(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
