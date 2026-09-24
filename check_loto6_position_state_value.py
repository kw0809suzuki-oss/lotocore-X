from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

import requests

SOURCE = "https://loto6.thekyo.jp/data/loto6.csv"
OUT_JSON = Path("results/loto6_position_state_value.json")
OUT_CSV = Path("results/loto6_position_state_value_blocks.csv")

TRAIN_END = 1500
ROLLING_WINDOW = 20
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260924

POSITION_LABELS = ("LOW", "MID", "HIGH")
DISTANCE_LABELS = ("NEAR", "MID", "FAR")
N_STATES = 9


def fetch_all_draws() -> list[dict]:
    response = requests.get(SOURCE, timeout=30, headers={"User-Agent": "lotocore-X position-state probe"})
    response.raise_for_status()
    text = response.content.decode("shift_jis")
    rows: list[dict] = []
    for cols in csv.reader(io.StringIO(text)):
        try:
            round_no = int(cols[0])
            nums = sorted(int(x) for x in cols[2:8])
        except (ValueError, IndexError):
            continue
        if len(nums) != 6:
            continue
        rows.append({"round": round_no, "date": cols[1], "nums": nums, "center": sum(nums) / 6.0})
    rows.sort(key=lambda x: x["round"])
    return rows


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        raise ValueError("empty quantile input")
    return xs[math.floor((len(xs) - 1) * p)]


def band3(value: float, cuts: tuple[float, float]) -> int:
    return 0 if value <= cuts[0] else 1 if value <= cuts[1] else 2


def brier(probs: list[float], target: int) -> float:
    return sum((p - (1.0 if i == target else 0.0)) ** 2 for i, p in enumerate(probs))


def top_k(probs: list[float], k: int) -> set[int]:
    return {i for i, _ in sorted(enumerate(probs), key=lambda x: (-x[1], x[0]))[:k]}


def state_label(state: int) -> str:
    return f"{POSITION_LABELS[state // 3]} x {DISTANCE_LABELS[state % 3]}"


def main() -> None:
    draws = fetch_all_draws()
    if not draws or draws[-1]["round"] < TRAIN_END + 1:
        raise RuntimeError("insufficient LOTO6 history")

    centers = [float(d["center"]) for d in draws]

    for i, draw in enumerate(draws):
        if i < ROLLING_WINDOW:
            draw["baseline20"] = None
            draw["distance20"] = None
        else:
            baseline = sum(centers[i - ROLLING_WINDOW : i]) / ROLLING_WINDOW
            draw["baseline20"] = baseline
            draw["distance20"] = abs(float(draw["center"]) - baseline)

    train_rows = [d for d in draws if d["round"] <= TRAIN_END and d["distance20"] is not None]
    position_cuts = (
        quantile([float(d["center"]) for d in train_rows], 1 / 3),
        quantile([float(d["center"]) for d in train_rows], 2 / 3),
    )
    distance_cuts = (
        quantile([float(d["distance20"]) for d in train_rows], 1 / 3),
        quantile([float(d["distance20"]) for d in train_rows], 2 / 3),
    )

    for draw in draws:
        if draw["distance20"] is None:
            draw["state"] = None
            continue
        p = band3(float(draw["center"]), position_cuts)
        d = band3(float(draw["distance20"]), distance_cuts)
        draw["state"] = p * 3 + d

    by_round = {int(d["round"]): d for d in draws}

    baseline_counts = [0] * N_STATES
    conditional_counts = [[0] * N_STATES for _ in range(N_STATES)]
    conditional_n = [0] * N_STATES
    train_transitions = 0

    for r in range(ROLLING_WINDOW + 1, TRAIN_END):
        current = by_round.get(r)
        nxt = by_round.get(r + 1)
        if not current or not nxt or current["state"] is None or nxt["state"] is None:
            continue
        a, y = int(current["state"]), int(nxt["state"])
        baseline_counts[y] += 1
        conditional_counts[a][y] += 1
        conditional_n[a] += 1
        train_transitions += 1

    baseline_probs = [c / train_transitions for c in baseline_counts]
    conditional_probs: list[list[float]] = []
    for s in range(N_STATES):
        if conditional_n[s]:
            conditional_probs.append([c / conditional_n[s] for c in conditional_counts[s]])
        else:
            conditional_probs.append(list(baseline_probs))

    baseline_top1 = next(iter(top_k(baseline_probs, 1)))
    baseline_top3 = top_k(baseline_probs, 3)

    eval_rows: list[dict] = []
    latest_round = int(draws[-1]["round"])
    for r in range(TRAIN_END, latest_round):
        current = by_round.get(r)
        nxt = by_round.get(r + 1)
        if not current or not nxt or current["state"] is None or nxt["state"] is None:
            continue
        a, y = int(current["state"]), int(nxt["state"])
        state_probs = conditional_probs[a]
        s_top1 = next(iter(top_k(state_probs, 1)))
        s_top3 = top_k(state_probs, 3)
        eval_rows.append(
            {
                "from_round": r,
                "to_round": r + 1,
                "current_state": a,
                "target_state": y,
                "baseline_brier": brier(baseline_probs, y),
                "position_brier": brier(state_probs, y),
                "baseline_top1": int(baseline_top1 == y),
                "position_top1": int(s_top1 == y),
                "baseline_top3": int(y in baseline_top3),
                "position_top3": int(y in s_top3),
                "train_examples_for_state": conditional_n[a],
            }
        )

    if not eval_rows:
        raise RuntimeError("no evaluation transitions")

    def avg(key: str, rows: list[dict]) -> float:
        return sum(float(x[key]) for x in rows) / len(rows)

    def summarize(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "brier_baseline": avg("baseline_brier", rows),
            "brier_position": avg("position_brier", rows),
            "brier_delta": avg("position_brier", rows) - avg("baseline_brier", rows),
            "top1_baseline": avg("baseline_top1", rows),
            "top1_position": avg("position_top1", rows),
            "top3_baseline": avg("baseline_top3", rows),
            "top3_position": avg("position_top3", rows),
            "mean_train_examples_for_current_state": avg("train_examples_for_state", rows),
            "min_train_examples_for_current_state": min(int(x["train_examples_for_state"]) for x in rows),
        }

    overall = summarize(eval_rows)
    block_specs = [
        ("1500->1700", 1500, 1700),
        ("1700->1900", 1700, 1900),
        ("1900->latest", 1900, latest_round),
    ]
    blocks: list[dict] = []
    for name, start, end in block_specs:
        rows = [x for x in eval_rows if start <= int(x["from_round"]) < end]
        s = summarize(rows)
        s["name"] = name
        blocks.append(s)

    rng = random.Random(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sample = [eval_rows[rng.randrange(len(eval_rows))] for _ in range(len(eval_rows))]
        deltas.append(avg("position_brier", sample) - avg("baseline_brier", sample))
    deltas.sort()
    ci_low = deltas[math.floor(0.025 * len(deltas))]
    ci_high = deltas[math.floor(0.975 * len(deltas))]

    gate = (
        overall["brier_delta"] < 0
        and ci_high < 0
        and all(float(block["brier_delta"]) <= 0 for block in blocks)
    )

    current = draws[-1]
    current_state = int(current["state"]) if current["state"] is not None else None

    baseline_rank = sorted(
        (
            {"state": i, "label": state_label(i), "pct": baseline_probs[i] * 100.0}
            for i in range(N_STATES)
        ),
        key=lambda x: -float(x["pct"]),
    )

    payload = {
        "probe": "LOTO6 Position State Value v0",
        "source": SOURCE,
        "history": {
            "draws": len(draws),
            "latest_round": latest_round,
        },
        "design": {
            "train": f"rounds {ROLLING_WINDOW + 1}-{TRAIN_END}",
            "eval": f"transitions {TRAIN_END}->{TRAIN_END + 1} through {latest_round - 1}->{latest_round}",
            "position_axis": "draw centroid; LOW/MID/HIGH by training-only 1/3 and 2/3 quantiles",
            "distance_axis": f"absolute distance from trailing-{ROLLING_WINDOW} centroid mean; NEAR/MID/FAR by training-only 1/3 and 2/3 quantiles",
            "state_space": "3 x 3 = 9 Position states",
            "baseline": "training unconditional next-State distribution",
            "candidate": "training next-State distribution conditional on current Position State",
            "primary_metric": "9-class Brier score; lower is better",
            "adoption_gate": "Brier delta < 0, bootstrap 95% upper bound < 0, and all three evaluation blocks have delta <= 0",
            "no_future_leakage": True,
        },
        "thresholds": {
            "position_centroid": list(position_cuts),
            "distance_from_trailing20": list(distance_cuts),
        },
        "training": {
            "transitions": train_transitions,
            "examples_per_current_state": {
                state_label(i): conditional_n[i] for i in range(N_STATES)
            },
            "baseline_top_states": baseline_rank[:5],
        },
        "evaluation": {
            "overall": overall,
            "bootstrap95_brier_delta": [ci_low, ci_high],
            "blocks": blocks,
        },
        "current": {
            "round": latest_round,
            "center": current["center"],
            "baseline20": current["baseline20"],
            "distance20": current["distance20"],
            "state": current_state,
            "state_label": state_label(current_state) if current_state is not None else None,
        },
        "gate": {
            "position_information_established": gate,
            "status": "PASS" if gate else "NOT_ESTABLISHED",
        },
        "boundary": (
            "PASS means only that this fixed Position coordinate added holdout next-State information under the predeclared gate. "
            "It does not establish direct number prediction or improved lottery returns."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "n",
                "brier_baseline",
                "brier_position",
                "brier_delta",
                "top1_baseline",
                "top1_position",
                "top3_baseline",
                "top3_position",
            ],
        )
        writer.writeheader()
        for block in blocks:
            writer.writerow({k: block[k] for k in writer.fieldnames})

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("LOTO6_POSITION_STATE_VALUE_COMPLETE")


if __name__ == "__main__":
    main()
