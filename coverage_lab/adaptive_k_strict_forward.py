#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PRESET = {"lotocore": "top7_mass", "x": "field_center_abs"}
RULES = ("last1", "last2_both")
COMPRESSED_KS = (10, 12)
GATE_KS = (10, 12)
WIDE_K = 37


def gate_diff(m: pd.DataFrame, hist_rounds: list[int]) -> float:
    z = m[m["round"].isin(hist_rounds)]
    diffs = []
    for k in GATE_KS:
        q = z[z.k == k]
        hi = q[q.state_bin == "high"].hit_lift.mean()
        lo = q[q.state_bin == "low"].hit_lift.mean()
        if pd.isna(hi) or pd.isna(lo):
            return float("nan")
        diffs.append(float(hi - lo))
    return sum(diffs) / len(diffs)


def rule_is_fresh(signs: list[bool], rule: str) -> bool:
    if rule == "last1":
        return bool(signs and signs[-1])
    if rule == "last2_both":
        return len(signs) >= 2 and signs[-1] and signs[-2]
    raise ValueError(f"unknown rule: {rule}")


def strict_forward_blocks(rounds: list[int], history_window: int, step: int):
    # Evaluation blocks are strictly after the decision window and do not overlap
    # when step is also the evaluation-block width. The final block may be shorter.
    for start in range(0, len(rounds) - history_window, step):
        hist = rounds[start : start + history_window]
        eva = rounds[start + history_window : start + history_window + step]
        if not eva:
            continue
        yield hist, eva


def random_same_k_control(k_values: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(k_values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # A random K-subset against 7 winning numbers has a Hypergeometric(37,7,K)
    # hit count. K=37 deterministically gives all 7.
    draws = rng.hypergeometric(ngood=7, nbad=30, nsample=k_values, size=(reps, n))
    expected = 7.0 * k_values / 37.0
    lifts = draws / expected
    recalls = draws / 7.0
    return float(lifts.mean()), float(recalls.mean()), float(draws.mean())


def evaluate_model(
    m: pd.DataFrame,
    model: str,
    metric: str,
    history_window: int,
    step: int,
    reps: int,
    seed: int,
):
    rounds = sorted(int(x) for x in m["round"].unique())
    signs: list[bool] = []
    out = []

    for block_idx, (hist_rounds, eval_rounds) in enumerate(
        strict_forward_blocks(rounds, history_window, step)
    ):
        diff = gate_diff(m, hist_rounds)
        if pd.isna(diff):
            continue
        signs.append(diff > 0)
        eval_z = m[m["round"].isin(eval_rounds)]

        for rule_idx, rule in enumerate(RULES):
            fresh = rule_is_fresh(signs, rule)
            for k_idx, compressed_k in enumerate(COMPRESSED_KS):
                fixed = eval_z[eval_z.k == compressed_k][
                    ["round", "state_bin", "topk_hits", "hit_lift", "recall"]
                ].rename(
                    columns={
                        "topk_hits": "fixed_hits",
                        "hit_lift": "fixed_lift",
                        "recall": "fixed_recall",
                    }
                )
                wide = eval_z[eval_z.k == WIDE_K][
                    ["round", "topk_hits", "hit_lift", "recall"]
                ].rename(
                    columns={
                        "topk_hits": "wide_hits",
                        "hit_lift": "wide_lift",
                        "recall": "wide_recall",
                    }
                )
                q = fixed.merge(wide, on="round", how="inner")
                trigger = fresh & (q.state_bin == "high")
                q["selected_k"] = np.where(trigger, compressed_k, WIDE_K)
                q["adaptive_hits"] = np.where(trigger, q.fixed_hits, q.wide_hits)
                q["adaptive_lift"] = np.where(trigger, q.fixed_lift, q.wide_lift)
                q["adaptive_recall"] = np.where(trigger, q.fixed_recall, q.wide_recall)

                control_seed = (
                    seed
                    + (block_idx * 1000)
                    + (rule_idx * 100)
                    + (k_idx * 10)
                    + (0 if model == "lotocore" else 1)
                )
                rand_lift, rand_recall, rand_hits = random_same_k_control(
                    q.selected_k.to_numpy(dtype=int), reps=reps, seed=control_seed
                )

                out.append(
                    {
                        "model": model,
                        "metric": metric,
                        "rule": rule,
                        "compressed_k": compressed_k,
                        "block_idx": block_idx,
                        "history_start": hist_rounds[0],
                        "decision_end": hist_rounds[-1],
                        "eval_start": eval_rounds[0],
                        "eval_end": eval_rounds[-1],
                        "eval_n": len(eval_rounds),
                        "gate_mean_diff": diff,
                        "gate_positive": diff > 0,
                        "fresh": fresh,
                        "compressed_rounds": int(trigger.sum()),
                        "compression_rate": float(trigger.mean()),
                        "adaptive_mean_k": float(q.selected_k.mean()),
                        "adaptive_hits": float(q.adaptive_hits.mean()),
                        "adaptive_recall": float(q.adaptive_recall.mean()),
                        "adaptive_hit_lift": float(q.adaptive_lift.mean()),
                        "fixed_hits": float(q.fixed_hits.mean()),
                        "fixed_recall": float(q.fixed_recall.mean()),
                        "fixed_hit_lift": float(q.fixed_lift.mean()),
                        "wide_hits": float(q.wide_hits.mean()),
                        "wide_recall": float(q.wide_recall.mean()),
                        "wide_hit_lift": float(q.wide_lift.mean()),
                        "same_k_random_hits": rand_hits,
                        "same_k_random_recall": rand_recall,
                        "same_k_random_hit_lift": rand_lift,
                    }
                )
    return out


def weighted_mean(g: pd.DataFrame, col: str) -> float:
    w = g.eval_n.to_numpy(dtype=float)
    x = g[col].to_numpy(dtype=float)
    return float(np.average(x, weights=w))


def summarize(res: pd.DataFrame):
    print("=== LOTO7 ADAPTIVE-K STRICT FORWARD TEST ===")
    print(
        "Decision: trailing 60 observed rounds; evaluation: next 20 unseen rounds "
        "(final block may be shorter)."
    )
    print("Freshness rules fixed from Phase 8 candidates: last1 / last2_both.")
    print("Policy: fresh AND model-state=high -> K10 or K12; otherwise K37.")
    print(
        "Controls: model fixed-K, K37 wide, deterministic-seed same-K random Monte Carlo."
    )
    for (model, rule, k), g in res.groupby(
        ["model", "rule", "compressed_k"], sort=True
    ):
        n = int(g.eval_n.sum())
        compressed = int(g.compressed_rounds.sum())
        mean_k = weighted_mean(g, "adaptive_mean_k")
        adapt = weighted_mean(g, "adaptive_hit_lift")
        fixed = weighted_mean(g, "fixed_hit_lift")
        wide = weighted_mean(g, "wide_hit_lift")
        rnd = weighted_mean(g, "same_k_random_hit_lift")
        arec = weighted_mean(g, "adaptive_recall")
        rrec = weighted_mean(g, "same_k_random_recall")
        print(
            f"MODEL {model:8s} RULE {rule:10s} K{k:02d} "
            f"eval_rounds={n:3d} compressed={compressed:3d} "
            f"({compressed/n:.3f}) meanK={mean_k:.2f} "
            f"lift adaptive={adapt:.4f} fixed={fixed:.4f} "
            f"wide={wide:.4f} sameK_random={rnd:.4f} "
            f"delta_random={adapt-rnd:+.4f} delta_fixed={adapt-fixed:+.4f} "
            f"recall adaptive={arec:.4f} random={rrec:.4f}"
        )
    print(
        "BOUNDARY: candidate-layer test only. No 10-ticket allocator, payout, "
        "or profitability claim is included."
    )
    print(
        "BOUNDARY: rules and K values are compared as preregistered candidates; "
        "this run does not select a winner for future use."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("results/loto7_candidate_compression_state_observables.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/loto7_candidate_compression_adaptive_k_strict_forward.csv"
        ),
    )
    p.add_argument("--history-window", type=int, default=60)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--random-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args()

    df = pd.read_csv(a.input).dropna(subset=["state_bin"])
    rows = []
    for model, metric in PRESET.items():
        m = df[(df.model == model) & (df.metric == metric)].copy()
        rows.extend(
            evaluate_model(
                m,
                model=model,
                metric=metric,
                history_window=a.history_window,
                step=a.step,
                reps=a.random_reps,
                seed=a.seed,
            )
        )
    res = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    summarize(res)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
