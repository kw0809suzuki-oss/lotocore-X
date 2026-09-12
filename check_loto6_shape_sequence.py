from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

DATA = Path("data/loto6.csv")
OUT = Path("results/loto6_shape_sequence_vs_shuffle.csv")
SEED = 20260912
SIMS = 1000
DRAW_SIZE = 6


def draw_shape(row: pd.Series) -> tuple[float, ...]:
    nums = sorted(float(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1))
    c = sum(nums) / DRAW_SIZE
    return tuple(x - c for x in nums)


def rms_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def step_distances(shapes: list[tuple[float, ...]], order: list[int] | None = None) -> list[float]:
    seq = shapes if order is None else [shapes[i] for i in order]
    return [rms_distance(a, b) for a, b in zip(seq, seq[1:])]


def summary(ds: list[float]) -> dict[str, float]:
    s = pd.Series(ds)
    return {
        "mean_step_distance": float(s.mean()),
        "median_step_distance": float(s.median()),
        "p10_step_distance": float(s.quantile(0.10)),
        "p90_step_distance": float(s.quantile(0.90)),
    }


def two_sided_p(sim: pd.Series, real: float) -> float:
    lo = float((sim <= real).mean())
    hi = float((sim >= real).mean())
    return min(1.0, 2.0 * min(lo, hi))


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").tail(400).reset_index(drop=True)
    if len(df) < 400:
        raise ValueError(f"need 400 draws, got {len(df)}")

    shapes = [draw_shape(row) for _, row in df.iterrows()]
    real = summary(step_distances(shapes))

    rng = random.Random(SEED)
    sim_rows = []
    base_order = list(range(len(shapes)))
    for _ in range(SIMS):
        order = base_order.copy()
        rng.shuffle(order)
        sim_rows.append(summary(step_distances(shapes, order)))

    sim = pd.DataFrame(sim_rows)
    rows = []
    for name, real_value in real.items():
        rows.append({
            "metric": name,
            "real": round(real_value, 6),
            "shuffle_mean": round(float(sim[name].mean()), 6),
            "shuffle_p05": round(float(sim[name].quantile(0.05)), 6),
            "shuffle_p95": round(float(sim[name].quantile(0.95)), 6),
            "two_sided_p": round(two_sided_p(sim[name], real_value), 4),
            "real_percentile": round(float((sim[name] <= real_value).mean()), 4),
        })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=== LOTO6 SIX-POINT SHAPE SEQUENCE vs SHUFFLED ORDER ===")
    print(f"rounds={int(df.iloc[0]['round'])}..{int(df.iloc[-1]['round'])} sims={SIMS} seed={SEED}")
    print("shape = sorted six numbers with each draw centroid removed (6D, sum=0)")
    print("null = same 400 observed shapes, temporal order randomly shuffled")
    for r in rows:
        print(
            f"  {r['metric']}: real={r['real']} shuffle_mean={r['shuffle_mean']} "
            f"p05={r['shuffle_p05']} p95={r['shuffle_p95']} "
            f"two_sided_p={r['two_sided_p']} percentile={r['real_percentile']}"
        )
    print(f"saved -> {OUT}")
    print("LOTO6_SHAPE_SEQUENCE_COMPLETE")


if __name__ == "__main__":
    main()
