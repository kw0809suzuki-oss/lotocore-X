from __future__ import annotations

import csv
import io
import json
import math
import random
from pathlib import Path

import requests

SOURCE = "https://loto6.thekyo.jp/data/loto6.csv"
OUT_JSON = Path("results/loto6_transition_null.json")
OUT_CSV = Path("results/loto6_transition_null_samples.csv")

TRAIN_END = 1500
NULL_REPS = 1000
NULL_SEED = 20260924
N_STATES = 7


def fetch_centers() -> list[tuple[int, float]]:
    response = requests.get(SOURCE, timeout=30, headers={"User-Agent": "lotocore-X transition-null probe"})
    response.raise_for_status()
    text = response.content.decode("shift_jis")
    rows: list[tuple[int, float]] = []
    for cols in csv.reader(io.StringIO(text)):
        try:
            round_no = int(cols[0])
            nums = [int(x) for x in cols[2:8]]
        except (ValueError, IndexError):
            continue
        if len(nums) != 6:
            continue
        rows.append((round_no, sum(nums) / 6.0))
    rows.sort(key=lambda x: x[0])
    return rows


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    return xs[math.floor((len(xs) - 1) * p)]


def amp_band(value: float, cuts: tuple[float, float]) -> int:
    return 0 if value <= cuts[0] else 1 if value <= cuts[1] else 2


def transition_state(delta: float, cuts: tuple[float, float]) -> int:
    if delta == 0:
        return 3
    a = amp_band(abs(delta), cuts)
    return a if delta < 0 else 4 + a


def brier(probs: list[float], target: int) -> float:
    return sum((p - (1.0 if i == target else 0.0)) ** 2 for i, p in enumerate(probs))


def evaluate(centers: list[float]) -> dict:
    deltas: list[float | None] = [None]
    for i in range(1, len(centers)):
        deltas.append(centers[i] - centers[i - 1])

    train_nonzero = [
        abs(float(deltas[i]))
        for i in range(1, min(TRAIN_END, len(centers)))
        if deltas[i] is not None and float(deltas[i]) != 0
    ]
    cuts = (quantile(train_nonzero, 1 / 3), quantile(train_nonzero, 2 / 3))
    states = [None if d is None else transition_state(float(d), cuts) for d in deltas]

    base = [0] * N_STATES
    cond = [[0] * N_STATES for _ in range(N_STATES)]
    cond_n = [0] * N_STATES
    train_n = 0

    # Python indices: round r -> index r-1. Train current rounds 2..1499, targets 3..1500.
    for current_idx in range(1, TRAIN_END - 1):
        a, y = states[current_idx], states[current_idx + 1]
        if a is None or y is None:
            continue
        base[y] += 1
        cond[a][y] += 1
        cond_n[a] += 1
        train_n += 1

    bp = [c / train_n for c in base]
    cp = [
        ([c / cond_n[s] for c in cond[s]] if cond_n[s] else list(bp))
        for s in range(N_STATES)
    ]

    base_scores: list[float] = []
    candidate_scores: list[float] = []
    # Evaluate transitions 1500->1501 through 2138->2139.
    for current_round in range(TRAIN_END, len(centers)):
        current_idx = current_round - 1
        target_idx = current_idx + 1
        if target_idx >= len(states):
            break
        a, y = states[current_idx], states[target_idx]
        if a is None or y is None:
            continue
        base_scores.append(brier(bp, y))
        candidate_scores.append(brier(cp[a], y))

    b0 = sum(base_scores) / len(base_scores)
    b1 = sum(candidate_scores) / len(candidate_scores)
    return {
        "n": len(base_scores),
        "brier_baseline": b0,
        "brier_transition": b1,
        "brier_delta": b1 - b0,
        "cuts": cuts,
    }


def main() -> None:
    rows = fetch_centers()
    centers = [c for _, c in rows]
    real = evaluate(centers)

    rng = random.Random(NULL_SEED)
    null_rows: list[dict] = []
    train_part = centers[:TRAIN_END]
    eval_part = centers[TRAIN_END:]

    for rep in range(NULL_REPS):
        shuffled_train = list(train_part)
        shuffled_eval = list(eval_part)
        rng.shuffle(shuffled_train)
        rng.shuffle(shuffled_eval)
        result = evaluate(shuffled_train + shuffled_eval)
        null_rows.append({"rep": rep + 1, "brier_delta": result["brier_delta"]})

    null_deltas = sorted(float(x["brier_delta"]) for x in null_rows)
    null_mean = sum(null_deltas) / len(null_deltas)
    q025 = null_deltas[math.floor(0.025 * len(null_deltas))]
    q50 = null_deltas[math.floor(0.5 * len(null_deltas))]
    q975 = null_deltas[math.floor(0.975 * len(null_deltas))]

    # More negative is better. One-sided empirical p: how often shuffled null is at least as good as real.
    null_as_good_or_better = sum(d <= real["brier_delta"] for d in null_deltas)
    p_one_sided = (null_as_good_or_better + 1) / (NULL_REPS + 1)
    real_percentile = 100.0 * sum(d <= real["brier_delta"] for d in null_deltas) / NULL_REPS

    independent_null_exceeded = real["brier_delta"] < q025 and p_one_sided <= 0.025

    payload = {
        "probe": "LOTO6 Transition Independent-Null v0",
        "source": SOURCE,
        "history": {"draws": len(rows), "latest_round": rows[-1][0]},
        "design": {
            "real_probe": "same fixed Transition definition: centroid delta direction x amplitude",
            "null": "shuffle draw centers independently within train rounds 1-1500 and evaluation rounds 1501-latest; preserves each period's marginal center distribution while destroying chronological order",
            "null_reps": NULL_REPS,
            "purpose": "separate lottery-sequence information from overlap/differencing mechanics shared by delta_t and delta_t+1",
            "primary_statistic": "candidate Brier minus unconditional-baseline Brier; lower is better",
            "gate": "real delta must be below the null 2.5th percentile and one-sided empirical p <= 0.025",
        },
        "real": {
            "n": real["n"],
            "brier_baseline": real["brier_baseline"],
            "brier_transition": real["brier_transition"],
            "brier_delta": real["brier_delta"],
            "amplitude_cuts": list(real["cuts"]),
        },
        "null": {
            "mean_brier_delta": null_mean,
            "q025": q025,
            "median": q50,
            "q975": q975,
            "min": min(null_deltas),
            "max": max(null_deltas),
            "real_percentile_lower_is_better": real_percentile,
            "one_sided_empirical_p": p_one_sided,
        },
        "gate": {
            "independent_null_exceeded": independent_null_exceeded,
            "status": "PASS" if independent_null_exceeded else "NOT_ESTABLISHED",
        },
        "boundary": (
            "A NOT_ESTABLISHED result means the observed Transition predictability is reproducible under an order-destroyed null and therefore cannot be attributed to lottery-sequence information. "
            "A PASS would only establish excess information beyond this specific null, not direct number prediction or improved lottery returns."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rep", "brier_delta"])
        writer.writeheader()
        writer.writerows(null_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("LOTO6_TRANSITION_NULL_COMPLETE")


if __name__ == "__main__":
    main()
