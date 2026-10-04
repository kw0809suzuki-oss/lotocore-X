#!/usr/bin/env python3
"""LOTO7 model championship v0 — GitHub-side ring only.

Compares existing GitHub models on the exact same 10-ticket / target-round surface:
  uniform_random | core10 | x10 | phase10_adaptive

Current Library Frozen is intentionally NOT implemented here. It will be joined
later from its authoritative Library implementation; old GitHub "frozen" probes
must not be substituted for it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import lotocore
import x_agent
from coverage_lab.practical_10ticket_backtest import (
    ranked_bundle,
    uniform_random_bundle,
    bundle_metrics,
    strict_forward_blocks,
    gate_diff,
    main_numbers,
    bonus_numbers,
)

def ranked_core(history: pd.DataFrame) -> list[int]:
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return sorted(range(1, 38), key=lambda n: (ranks[n], n))

def ranked_x(history: pd.DataFrame) -> list[int]:
    snap = x_agent.score_snapshot(history, competition_gate=True)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return sorted(range(1, 38), key=lambda n: (ranks[n], n))

def run(data_path: Path, state_path: Path, history_window=60, step=20, model_window=100):
    df = pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    by_round = {int(r["round"]): i for i, r in df.iterrows()}

    obs = pd.read_csv(state_path).dropna(subset=["state_bin"])
    m = obs[(obs.model == "x") & (obs.metric == "field_center_abs")].copy()
    rounds = sorted(int(x) for x in m["round"].unique())

    rows, signs = [], []
    for block_idx, (hist_rounds, eval_rounds) in enumerate(
        strict_forward_blocks(rounds, history_window, step)
    ):
        diff = gate_diff(m, hist_rounds)
        if pd.isna(diff):
            continue
        signs.append(diff > 0)
        fresh = bool(signs[-1])

        for rnd in eval_rounds:
            z = m[(m["round"] == rnd) & (m.k == 10)]
            if z.empty:
                continue
            state_bin = str(z.iloc[0].state_bin)
            adaptive_k = 10 if (fresh and state_bin == "high") else 37

            idx = by_round[rnd]
            if idx < model_window:
                continue
            history = df.iloc[idx-model_window:idx]
            row = df.iloc[idx]
            main, bonus = main_numbers(row), bonus_numbers(row)
            core_order = ranked_core(history)
            x_order = ranked_x(history)
            seed = rnd * 100_003 + 20261004

            # Same existing 10-ticket allocator for ranked candidate models.
            # Phase10 keeps its original adaptive K rule.
            strategies = {
                "uniform_random": uniform_random_bundle(seed + 1),
                "core10": ranked_bundle(core_order, seed + 2),
                "x10": ranked_bundle(x_order, seed + 3),
                "phase10_adaptive": ranked_bundle(x_order[:adaptive_k], seed + 4),
            }

            for strategy, tickets in strategies.items():
                rec = {
                    "round": rnd,
                    "date": row.get("date", ""),
                    "block_idx": block_idx,
                    "strategy": strategy,
                    "adaptive_k": adaptive_k,
                }
                rec.update(bundle_metrics(tickets, main, bonus, rnd))
                rows.append(rec)
    return pd.DataFrame(rows)

def summarize(res: pd.DataFrame):
    print("=== LOTO7 MODEL CHAMPIONSHIP v0 | GITHUB SIDE ===")
    print(f"rounds={res['round'].nunique()} strategies={res.strategy.nunique()} tickets_per_round=10")
    order = ["uniform_random", "core10", "x10", "phase10_adaptive"]
    for name in order:
        g = res[res.strategy == name]
        print(
            f"{name:18s} n={len(g)} "
            f"mean_max={g.max_main_hits.mean():.4f} "
            f"P3+={(g.max_main_hits>=3).mean():.4f} "
            f"P4+={(g.max_main_hits>=4).mean():.4f} "
            f"P5+={(g.max_main_hits>=5).mean():.4f} "
            f"best={int(g.max_main_hits.max())}"
        )
    p = res.pivot(index="round", columns="strategy", values="max_main_hits")
    base = p["uniform_random"]
    for name in order[1:]:
        x = p[name]
        print(f"PAIRED {name} vs random: win={(x>base).sum()} tie={(x==base).sum()} loss={(x<base).sum()}")
    print("BOUNDARY: GitHub-side ring only. Current Library Frozen is not present and is not substituted.")
    print("BOUNDARY: no model logic is tuned from these results.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/loto7.csv"))
    p.add_argument("--state", type=Path, default=Path("results/loto7_candidate_compression_state_observables.csv"))
    p.add_argument("--out", type=Path, default=Path("results/loto7_model_championship_v0_github.csv"))
    p.add_argument("--history-window", type=int, default=60)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--model-window", type=int, default=100)
    a = p.parse_args()
    res = run(a.data, a.state, a.history_window, a.step, a.model_window)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    summarize(res)
    print(f"saved -> {a.out}")

if __name__ == "__main__":
    main()

