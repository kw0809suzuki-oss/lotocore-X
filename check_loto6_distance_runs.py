from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path("data/loto6.csv")
OUT_SUMMARY = Path("results/loto6_distance_runs_vs_random.csv")
OUT_RUNS = Path("results/loto6_distance_runs_real.csv")
DRAW_SIZE = 6
BASELINE_WINDOW = 100
SIMS = 1000
SEED = 20260912


@dataclass
class Run:
    start_round: int
    end_round: int
    side: int
    length: int
    peak_abs_distance: float
    area_abs_distance: float
    entry_abs_distance: float


def centers_from_df(df: pd.DataFrame) -> list[float]:
    return [
        sum(float(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)) / DRAW_SIZE
        for _, row in df.iterrows()
    ]


def sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def trailing_distances(centers: list[float], rounds: list[int]) -> list[tuple[int, float, float, float]]:
    out: list[tuple[int, float, float, float]] = []
    for i in range(BASELINE_WINDOW, len(centers)):
        baseline = sum(centers[i - BASELINE_WINDOW:i]) / BASELINE_WINDOW
        d = centers[i] - baseline
        out.append((rounds[i], centers[i], baseline, d))
    return out


def extract_runs(distance_rows: list[tuple[int, float, float, float]]) -> list[Run]:
    runs: list[Run] = []
    current_side = 0
    start_round = None
    ds: list[float] = []

    def close(end_round: int) -> None:
        nonlocal current_side, start_round, ds
        if current_side == 0 or start_round is None or not ds:
            return
        runs.append(
            Run(
                start_round=start_round,
                end_round=end_round,
                side=current_side,
                length=len(ds),
                peak_abs_distance=max(abs(x) for x in ds),
                area_abs_distance=sum(abs(x) for x in ds),
                entry_abs_distance=abs(ds[0]),
            )
        )

    prev_round = None
    for rnd, _, _, d in distance_rows:
        s = sign(d)
        if s == 0:
            if current_side != 0 and prev_round is not None:
                close(prev_round)
            current_side = 0
            start_round = None
            ds = []
            prev_round = rnd
            continue

        if current_side == 0:
            current_side = s
            start_round = rnd
            ds = [d]
        elif s == current_side:
            ds.append(d)
        else:
            if prev_round is not None:
                close(prev_round)
            current_side = s
            start_round = rnd
            ds = [d]
        prev_round = rnd

    if current_side != 0 and prev_round is not None:
        close(prev_round)
    return runs


def summarize(distance_rows: list[tuple[int, float, float, float]], runs: list[Run]) -> dict[str, float]:
    if not distance_rows or not runs:
        return {
            "crossing_rate": 0.0,
            "mean_run_length": 0.0,
            "max_run_length": 0.0,
            "mean_peak_abs_distance": 0.0,
            "max_peak_abs_distance": 0.0,
            "mean_area_abs_distance": 0.0,
            "mean_entry_abs_distance": 0.0,
        }

    signs = [sign(x[3]) for x in distance_rows]
    valid_pairs = [(a, b) for a, b in zip(signs, signs[1:]) if a != 0 and b != 0]
    crossing_rate = sum(a != b for a, b in valid_pairs) / len(valid_pairs) if valid_pairs else 0.0

    return {
        "crossing_rate": crossing_rate,
        "mean_run_length": sum(r.length for r in runs) / len(runs),
        "max_run_length": float(max(r.length for r in runs)),
        "mean_peak_abs_distance": sum(r.peak_abs_distance for r in runs) / len(runs),
        "max_peak_abs_distance": max(r.peak_abs_distance for r in runs),
        "mean_area_abs_distance": sum(r.area_abs_distance for r in runs) / len(runs),
        "mean_entry_abs_distance": sum(r.entry_abs_distance for r in runs) / len(runs),
    }


def random_centers(rng: random.Random, n: int) -> list[float]:
    nums = range(1, 44)
    out = []
    for _ in range(n):
        draw = rng.sample(nums, DRAW_SIZE)
        out.append(sum(draw) / DRAW_SIZE)
    return out


def analyze(centers: list[float], rounds: list[int]) -> tuple[dict[str, float], list[Run], list[tuple[int, float, float, float]]]:
    rows = trailing_distances(centers, rounds)
    runs = extract_runs(rows)
    return summarize(rows, runs), runs, rows


def main() -> None:
    df = pd.read_csv(DATA).sort_values("round").tail(400).reset_index(drop=True)
    if len(df) < 400:
        raise ValueError(f"need 400 draws, got {len(df)}")

    rounds = [int(x) for x in df["round"]]
    real_centers = centers_from_df(df)
    real_summary, real_runs, distance_rows = analyze(real_centers, rounds)

    rng = random.Random(SEED)
    sim_rows = []
    synthetic_rounds = list(range(len(df)))
    for _ in range(SIMS):
        sim_centers = random_centers(rng, len(df))
        summary, _, _ = analyze(sim_centers, synthetic_rounds)
        sim_rows.append(summary)
    sim = pd.DataFrame(sim_rows)

    result_rows = []
    for metric, real_value in real_summary.items():
        vals = sim[metric]
        upper_tail = float((vals >= real_value).mean())
        lower_tail = float((vals <= real_value).mean())
        two_sided = min(1.0, 2.0 * min(upper_tail, lower_tail))
        result_rows.append({
            "metric": metric,
            "real": round(real_value, 6),
            "random_mean": round(float(vals.mean()), 6),
            "random_p05": round(float(vals.quantile(0.05)), 6),
            "random_p95": round(float(vals.quantile(0.95)), 6),
            "upper_tail_p": round(upper_tail, 4),
            "lower_tail_p": round(lower_tail, 4),
            "two_sided_tail_p": round(two_sided, 4),
        })

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(OUT_SUMMARY, index=False)

    pd.DataFrame([
        {
            "start_round": r.start_round,
            "end_round": r.end_round,
            "side": "UP" if r.side > 0 else "DOWN",
            "length": r.length,
            "peak_abs_distance": round(r.peak_abs_distance, 6),
            "area_abs_distance": round(r.area_abs_distance, 6),
            "entry_abs_distance": round(r.entry_abs_distance, 6),
        }
        for r in real_runs
    ]).to_csv(OUT_RUNS, index=False)

    print("=== LOTO6 DISTANCE-RUN DIAGNOSTIC ===")
    print(
        f"rounds={rounds[0]}..{rounds[-1]} baseline_window={BASELINE_WINDOW} "
        f"observed_points={len(distance_rows)} runs={len(real_runs)} sims={SIMS} seed={SEED}"
    )
    print("distance = draw centroid - mean centroid of previous 100 draws")
    print("run = consecutive same-sign distance until the moving baseline is crossed")
    print("REAL vs RANDOM:")
    for r in result_rows:
        print(
            f"  {r['metric']}: real={r['real']} random_mean={r['random_mean']} "
            f"p05={r['random_p05']} p95={r['random_p95']} two_sided_p={r['two_sided_tail_p']}"
        )

    print("RECENT REAL RUNS:")
    for r in real_runs[-10:]:
        side = "UP" if r.side > 0 else "DOWN"
        print(
            f"  {r.start_round}..{r.end_round} {side} len={r.length} "
            f"peak={r.peak_abs_distance:.3f} area={r.area_abs_distance:.3f} entry={r.entry_abs_distance:.3f}"
        )

    print(f"saved -> {OUT_SUMMARY}")
    print(f"saved -> {OUT_RUNS}")
    print("LOTO6_DISTANCE_RUN_COMPLETE")


if __name__ == "__main__":
    main()
