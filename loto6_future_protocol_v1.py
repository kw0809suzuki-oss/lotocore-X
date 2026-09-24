from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import requests

SOURCE = "https://loto6.thekyo.jp/data/loto6.csv"
OUT = Path("results/loto6_future_protocol_v1.json")

START_ROUND = 2140
HORIZON = 200
MODEL_VERSION = "frequency-joint-v1"
ALPHA = 1.0
TOTAL_COMBINATIONS = 6_096_454
UNIFORM_PROB = 1.0 / TOTAL_COMBINATIONS
UNIFORM_LOG = math.log(UNIFORM_PROB)
PASS_LOG_LR = math.log(100.0)


def fetch_draws() -> list[dict]:
    r = requests.get(
        SOURCE,
        timeout=30,
        headers={"User-Agent": "lotocore-X future-protocol-v1"},
    )
    r.raise_for_status()
    text = r.content.decode("shift_jis")
    draws: list[dict] = []
    for cols in csv.reader(io.StringIO(text)):
        try:
            round_no = int(cols[0])
            nums = sorted(int(x) for x in cols[2:8])
        except (ValueError, IndexError):
            continue
        if len(nums) != 6 or len(set(nums)) != 6 or not all(1 <= n <= 43 for n in nums):
            continue
        draws.append({"round": round_no, "date": cols[1], "nums": nums})
    draws.sort(key=lambda x: x["round"])
    return draws


def e6(weights: list[float]) -> float:
    dp = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for w in weights:
        for k in range(6, 0, -1):
            dp[k] += dp[k - 1] * w
    return dp[6]


def model_from_counts(counts: list[int], n_draws: int):
    denom = n_draws * 6 + 43 * ALPHA
    weights = [(counts[n] + ALPHA) / denom for n in range(1, 44)]
    z = e6(weights)

    def probability(nums: list[int]) -> float:
        p = 1.0
        for n in nums:
            p *= weights[n - 1]
        return p / z

    return probability


def state_hash(counts: list[int], target_round: int, last_history_round: int) -> str:
    payload = {
        "protocol": "LOTO6_FUTURE_PROTOCOL_V1",
        "model_version": MODEL_VERSION,
        "alpha": ALPHA,
        "target_round": target_round,
        "last_history_round": last_history_round,
        "counts_1_to_43": counts[1:44],
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    draws = fetch_draws()
    if not draws or draws[-1]["round"] < START_ROUND - 1:
        raise RuntimeError(f"need history through round {START_ROUND - 1}")

    counts = [0] * 44
    n_draws = 0
    evidence: list[dict] = []
    cumulative = 0.0

    for draw in draws:
        round_no = int(draw["round"])

        if round_no >= START_ROUND:
            probability = model_from_counts(counts, n_draws)
            p_model = probability(draw["nums"])
            model_log = math.log(p_model)
            log_lr = model_log - UNIFORM_LOG
            cumulative += log_lr
            evidence.append(
                {
                    "round": round_no,
                    "date": draw["date"],
                    "nums": draw["nums"],
                    "history_through": round_no - 1,
                    "model_probability": p_model,
                    "uniform_probability": UNIFORM_PROB,
                    "model_log_score": model_log,
                    "uniform_log_score": UNIFORM_LOG,
                    "log_lr": log_lr,
                    "cumulative_log_lr": cumulative,
                    "pre_draw_state_hash": state_hash(counts, round_no, round_no - 1),
                }
            )

        for n in draw["nums"]:
            counts[n] += 1
        n_draws += 1

    latest = draws[-1]
    next_round = int(latest["round"]) + 1
    next_hash = state_hash(counts, next_round, int(latest["round"]))

    passed = cumulative >= PASS_LOG_LR
    complete = len(evidence) >= HORIZON
    status = "PASS" if passed else "NOT_ESTABLISHED" if complete else "OBSERVING"

    payload = {
        "protocol": "LOTO6 Future Protocol v1",
        "model_version": MODEL_VERSION,
        "source": SOURCE,
        "frozen": {
            "development_rounds": "1-2139",
            "future_start_round": START_ROUND,
            "horizon": HORIZON,
            "alpha": ALPHA,
            "primary_null": f"Uniform over C(43,6)={TOTAL_COMBINATIONS}",
            "primary_score": "cumulative log likelihood ratio versus Uniform",
            "pass_log_lr": PASS_LOG_LR,
            "pass_likelihood_ratio": 100,
        },
        "history": {
            "draws_available": len(draws),
            "latest_round": int(latest["round"]),
            "latest_date": latest["date"],
        },
        "evaluation": {
            "future_draws_scored": len(evidence),
            "cumulative_log_lr": cumulative,
            "mean_log_lr": cumulative / len(evidence) if evidence else 0.0,
            "status": status,
        },
        "next_forecast_state": {
            "target_round": next_round,
            "history_through": int(latest["round"]),
            "model_version": MODEL_VERSION,
            "alpha": ALPHA,
            "state_sha256": next_hash,
        },
        "evidence": evidence,
        "boundary": (
            "PASS means this frozen adaptive probability machine accumulated a likelihood ratio "
            "of at least 100 versus the official uniform joint null during the predeclared future test. "
            "It does not establish profitability, causality, or permanent predictability."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("LOTO6_FUTURE_PROTOCOL_V1_COMPLETE")


if __name__ == "__main__":
    main()
