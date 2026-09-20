#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import x_agent
from practical_10ticket_backtest import ranked_bundle, gate_diff

METRIC = "field_center_abs"


def current_state_bin(df: pd.DataFrame, state_obs: pd.DataFrame, model_window: int = 100):
    hist = df.tail(model_window)
    snap = x_agent.score_snapshot(hist, competition_gate=True)
    value = abs(float(snap["state"]["field_center_delta"]))

    m = (
        state_obs[(state_obs.model == "x") & (state_obs.metric == METRIC)]
        [["round", "metric_value"]]
        .drop_duplicates("round")
        .sort_values("round")
        .tail(100)
    )
    q1 = float(m.metric_value.quantile(1 / 3))
    q2 = float(m.metric_value.quantile(2 / 3))
    if value <= q1:
        state_bin = "low"
    elif value >= q2:
        state_bin = "high"
    else:
        state_bin = "mid"
    return snap, value, q1, q2, state_bin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/loto7.csv"))
    p.add_argument(
        "--state",
        type=Path,
        default=Path("results/loto7_candidate_compression_state_observables.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/loto7_phase10_next_bundle.json"),
    )
    p.add_argument("--model-window", type=int, default=100)
    a = p.parse_args()

    df = pd.read_csv(a.data).sort_values("round").reset_index(drop=True)
    obs = pd.read_csv(a.state).dropna(subset=["state_bin"])
    m = obs[(obs.model == "x") & (obs.metric == METRIC)].copy()

    completed_rounds = sorted(int(x) for x in m["round"].unique())
    gate_rounds = completed_rounds[-60:]
    diff = gate_diff(m, gate_rounds)
    fresh = bool(diff > 0)

    snap, metric_value, q1, q2, state_bin = current_state_bin(
        df, obs, model_window=a.model_window
    )
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))

    next_round = int(df["round"].max()) + 1
    selected_k = 10 if (fresh and state_bin == "high") else 37
    pool = ranked[:selected_k]
    seed = next_round * 100_003 + 20260919 + 1
    tickets = ranked_bundle(pool, seed)

    ticket_membership = {n: [] for n in range(1, 38)}
    for i, ticket in enumerate(tickets, 1):
        for n in ticket:
            ticket_membership[n].append(i)

    candidate_trace = []
    for n in ranked:
        comp = snap.get("components", {}).get(str(n), {})
        candidate_trace.append({
            "number": n,
            "rank": ranks[n],
            "score": round(float(snap["scores"][str(n)]), 8),
            "in_selected_pool": n in pool,
            "ticket_count": len(ticket_membership[n]),
            "ticket_ids": ticket_membership[n],
            "score_components": {
                k: round(float(v), 8) for k, v in comp.items()
            },
        })

    payload = {
        "snapshot_type": "pre_draw_fixed",
        "history_end_round": int(df["round"].max()),
        "next_round": next_round,
        "model": "x",
        "freshness_rule": "last1",
        "gate_round_start": gate_rounds[0],
        "gate_round_end": gate_rounds[-1],
        "gate_mean_diff": round(float(diff), 6),
        "fresh": fresh,
        "metric": METRIC,
        "metric_value": round(metric_value, 8),
        "prior100_q1": round(q1, 8),
        "prior100_q2": round(q2, 8),
        "state_bin": state_bin,
        "selected_k": selected_k,
        "candidate_pool": pool,
        "decision_trace": {
            "freshness_gate": {
                "rule": "last1",
                "gate_mean_diff": round(float(diff), 6),
                "fresh": fresh,
            },
            "state_gate": {
                "metric": METRIC,
                "value": round(metric_value, 8),
                "prior100_q1": round(q1, 8),
                "prior100_q2": round(q2, 8),
                "state_bin": state_bin,
            },
            "compression_gate": {
                "rule": "K10 only when fresh AND state=high; otherwise K37",
                "selected_k": selected_k,
            },
        },
        "candidate_trace": candidate_trace,
        "ticket_count": 10,
        "price_per_ticket_yen": 300,
        "total_cost_yen": 3000,
        "tickets": [list(t) for t in tickets],
        "boundary": [
            "Generated only from history through round 695 (or current history_end_round).",
            "No future draw result is used.",
            "This is a fixed research snapshot, not a profit guarantee.",
            "candidate_trace is a contemporaneous computation trace, not a post-result explanation.",
        ],
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== LOTO7 PHASE 10 NEXT PRE-DRAW BUNDLE ===")
    print(
        f"history_end={payload['history_end_round']} next_round={next_round} "
        f"gate_diff={diff:+.4f} fresh={fresh} "
        f"state={state_bin} metric={metric_value:.6f} "
        f"q1={q1:.6f} q2={q2:.6f} selected_k={selected_k}"
    )
    print("candidate_pool=" + "-".join(f"{n:02d}" for n in pool))
    for i, t in enumerate(tickets, 1):
        print(f"T{i:02d} " + "-".join(f"{n:02d}" for n in t))
    print("cost_yen=3000")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
