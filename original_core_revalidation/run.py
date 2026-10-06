from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_loto7 import fetch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "original_core_v0.json"
RESULT_DIR = ROOT / "results" / "original_core_revalidation"


def state(draw: np.ndarray) -> np.ndarray:
    x = draw.astype(float)
    center = float(x.mean())
    variance = float(((x - center) ** 2).mean())
    return np.array([center, variance], dtype=float)


def load_draws() -> pd.DataFrame:
    # Fetch enough history so the latest 100 targets still have a long past.
    df = fetch(limit=800).sort_values("round").reset_index(drop=True)
    return df


def predict_region(states: np.ndarray, target_idx: int, k: int, qlo: float, qhi: float):
    # Current observable state is target_idx-1. Candidate reference j must have
    # j+1 < target_idx, so every successor used by the predictor is strictly past.
    current_idx = target_idx - 1
    ref_idx = np.arange(0, current_idx - 1)
    if len(ref_idx) < k:
        raise ValueError("not enough past references")

    past_for_scale = states[:current_idx]
    mu = past_for_scale.mean(axis=0)
    sd = past_for_scale.std(axis=0)
    sd[sd == 0] = 1.0

    current_z = (states[current_idx] - mu) / sd
    ref_z = (states[ref_idx] - mu) / sd
    dist = np.sqrt(((ref_z - current_z) ** 2).sum(axis=1))
    order = np.lexsort((ref_idx, dist))
    neighbors = ref_idx[order[:k]]
    successors = neighbors + 1

    # Leakage guard.
    assert np.all(successors < target_idx)

    nxt = states[successors]
    lo = np.quantile(nxt, qlo, axis=0)
    hi = np.quantile(nxt, qhi, axis=0)
    return lo, hi, neighbors, successors


def inside(s: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.all((s >= lo) & (s <= hi)))


def run(config: dict) -> dict:
    df = load_draws()
    cols = [f"n{i}" for i in range(1, 8)]
    draws = df[cols].to_numpy(dtype=int)
    states = np.vstack([state(d) for d in draws])

    target_count = int(config["target_count"])
    k = int(config["neighbor_k"])
    qlo, qhi = map(float, config["region_quantiles"])
    repeats = int(config["random_repeats"])
    seed = int(config["random_seed"])

    start = len(df) - target_count
    if start <= k + 2:
        raise ValueError("history too short for requested target_count")

    core_hits = 0
    random_hits = np.zeros(repeats, dtype=int)
    rows = []

    for target_idx in range(start, len(df)):
        lo, hi, neighbors, successors = predict_region(states, target_idx, k, qlo, qhi)
        actual = states[target_idx]
        core_hit = inside(actual, lo, hi)
        core_hits += int(core_hit)

        width = hi - lo

        # Same-width Random null:
        # choose a random past successor state as the region center.
        # This keeps the null on the empirical state distribution and uses no
        # target/future information.
        past_centers = states[1:target_idx]
        rng = np.random.default_rng(seed + int(df.iloc[target_idx]["round"]))
        picks = rng.integers(0, len(past_centers), size=repeats)
        centers = past_centers[picks]
        rnd_lo = centers - width / 2.0
        rnd_hi = centers + width / 2.0
        hit_mask = np.all((actual >= rnd_lo) & (actual <= rnd_hi), axis=1)
        random_hits += hit_mask.astype(int)

        rows.append(
            {
                "target_round": int(df.iloc[target_idx]["round"]),
                "history_end_round": int(df.iloc[target_idx - 1]["round"]),
                "core_hit": int(core_hit),
                "actual_center": float(actual[0]),
                "actual_variance": float(actual[1]),
                "region_center_lo": float(lo[0]),
                "region_center_hi": float(hi[0]),
                "region_variance_lo": float(lo[1]),
                "region_variance_hi": float(hi[1]),
                "neighbor_rounds": [int(df.iloc[j]["round"]) for j in neighbors],
                "successor_rounds": [int(df.iloc[j]["round"]) for j in successors],
            }
        )

    random_mean = float(random_hits.mean())
    random_median = float(np.median(random_hits))
    p_one_sided = float((1 + np.sum(random_hits >= core_hits)) / (repeats + 1))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT_DIR / "targets.csv", index=False)

    csv_bytes = df.to_csv(index=False).encode()
    cfg_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    summary = {
        "experiment": "original_core_revalidation_v0",
        "evaluation": "latest_100_walk_forward",
        "target_round_start": int(df.iloc[start]["round"]),
        "target_round_end": int(df.iloc[-1]["round"]),
        "targets": target_count,
        "core_hits": int(core_hits),
        "core_hit_rate": core_hits / target_count,
        "random_repeats": repeats,
        "random_mean_hits": random_mean,
        "random_median_hits": random_median,
        "random_5pct_hits": float(np.quantile(random_hits, 0.05)),
        "random_95pct_hits": float(np.quantile(random_hits, 0.95)),
        "difference_vs_random_mean": float(core_hits - random_mean),
        "one_sided_empirical_p": p_one_sided,
        "config": config,
        "input_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "boundary": [
            "past-only",
            "center+variance only",
            "K fixed",
            "IQR region fixed",
            "same-width empirical-random regions",
            "no ticket generation",
            "no feature tuning",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULT_DIR / "random_hit_counts.json").write_text(
        json.dumps(random_hits.tolist()) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run(config)


if __name__ == "__main__":
    main()
