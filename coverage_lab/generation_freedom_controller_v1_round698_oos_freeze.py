#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K, portfolio_bundle,
)
from coverage_lab.generation_freedom_controller_v1_inner_validation import (
    prep, select_a_centroid, select_b_core_static,
    select_c_dynamic, select_d_controller_v1,
)

BASE = Path("coverage_lab/fixtures/loto7_1_696.csv")
R697 = Path("coverage_lab/fixtures/loto7_round697_public_20261002.csv")
OUT = Path("results/generation_freedom_controller_v1_round698_oos_freeze.json")
TARGET_ROUND = 698


def digest(tickets):
    payload = ";".join("-".join(f"{n:02d}" for n in t) for t in tickets)
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    base = pd.read_csv(BASE)
    add = pd.read_csv(R697)
    df = pd.concat([base, add], ignore_index=True).sort_values("round").reset_index(drop=True)

    if int(df.iloc[-1]["round"]) != 697:
        raise RuntimeError("forward freeze requires history through round 697 exactly")
    if len(df[df["round"] == 697]) != 1:
        raise RuntimeError("round 697 must exist exactly once")

    history = df.iloc[-MODEL_WINDOW:]
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    scores = {int(k): float(v) for k, v in snap["scores"].items()}
    ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))
    pool = ranked[:POOL_K]

    seed = TARGET_ROUND * 100_003 + 20261002
    core, combos, z, m, median, high, d, core_min = prep(pool, scores, seed + 2)

    A = select_a_centroid(high, d)
    B = select_b_core_static(high, core_min)
    C = select_c_dynamic(high, core, z)
    D = select_d_controller_v1(high, core, z, m)

    portfolio20, trace = portfolio_bundle(pool, scores, seed + 2)
    d_current = [tuple(x) for x in trace["diversify"]]
    if D != d_current:
        raise RuntimeError("Controller v1 fidelity mismatch at forward freeze")

    variants = {
        "A_centroid_static_16": core + A,
        "B_core_static_16": core + B,
        "C_dynamic_frontier_16": core + C,
        "D_controller_v1_16": core + D,
        "D_current_portfolio20_reference": portfolio20,
    }

    payload = {
        "experiment": "generation_freedom_controller_v1_round698_oos_freeze",
        "target_round": TARGET_ROUND,
        "target_draw_date": "2026-10-09",
        "freeze_history_last_round": 697,
        "history_window": MODEL_WINDOW,
        "candidate_pool_core18": pool,
        "seed": seed,
        "round697_input": {
            "date": "2026-10-02",
            "main_numbers": [1, 6, 12, 23, 28, 32, 36],
            "bonus_numbers": [31, 34],
            "purpose": "history only for target round 698 generation",
        },
        "variants": {
            name: {
                "ticket_count": len(tickets),
                "tickets": [list(t) for t in tickets],
                "sha256": digest(tickets),
            }
            for name, tickets in variants.items()
        },
        "controller_fidelity": {
            "D_matches_current_portfolio_diversify": True,
        },
        "evaluation_status": "FROZEN_BEFORE_ROUND_698_RESULT",
        "outer_validation_protocol": [
            "Round 698 result must not be used to regenerate or alter these frozen tickets.",
            "Primary external comparison is A/B/C/D on equal 16-ticket Core10+Diversify6 bundles.",
            "Current Portfolio20 is stored only as an end-to-end reference, not as an equal-budget comparator to the 16-ticket mechanism variants.",
            "No threshold, ratio, tie-break, distance definition, or candidate sampling rule may be tuned after seeing round 698.",
            "One future draw is one observation only. Continue the same freeze-before-result process on later rounds before drawing any external conclusion.",
        ],
        "source_note": [
            "Round 697 was already public on 2026-10-02 and is used only as prior history for round 698.",
            "Round 698 is scheduled for 2026-10-09; the freeze is created before that draw.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== GENERATION FREEDOM CONTROLLER v1 ROUND 698 OOS FREEZE ===")
    print("TARGET", TARGET_ROUND)
    print("HISTORY_LAST", 697)
    print("CORE18", "-".join(f"{n:02d}" for n in pool))
    print("HASHES", json.dumps({k:v["sha256"] for k,v in payload["variants"].items()}, sort_keys=True))
    print("STATUS", payload["evaluation_status"])
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
