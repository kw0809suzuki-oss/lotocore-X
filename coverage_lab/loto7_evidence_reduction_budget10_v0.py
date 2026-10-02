#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW,
    POOL_K,
    A_TICKETS,
    main_numbers,
    unique_structured_bundle,
    portfolio_bundle,
    bundle_metrics,
)
from coverage_lab.loto7_hybrid_ratio_sweep_v0 import select


def merge_unique_budget(a_pick, b_pick, b20, total):
    chosen = list(a_pick)
    used = set(chosen)
    preferred = list(b_pick)
    preferred_set = set(preferred)
    leftovers = [t for t in b20 if t not in preferred_set]
    repairs = 0

    for t in preferred:
        if t not in used:
            pick = t
        else:
            repairs += 1
            pick = next((x for x in leftovers if x not in used), None)
            if pick is None:
                raise RuntimeError("no unused Portfolio replacement")
            leftovers.remove(pick)
        chosen.append(pick)
        used.add(pick)

    if len(chosen) != total or len(set(chosen)) != total:
        raise RuntimeError(f"hybrid is not {total} unique tickets")
    return chosen, repairs


def success(bundle, actual):
    return bundle_metrics(bundle, actual)["tickets_ge5"] > 0


def main():
    df = pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)

    obs = {
        "20": {"structured": 0, "portfolio": 0, "hybrid_10_10": 0, "hybrid_gt_both": 0},
        "10": {
            "structured": 0,
            "portfolio": 0,
            "hybrid_5_5": 0,
            "hybrid_gt_both": 0,
            "exclusive_total": 0,
            "exclusive_recovered": 0,
            "collision_repairs": 0,
        },
    }

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx - MODEL_WINDOW : idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        pool = sorted(range(1, 38), key=lambda n: (ranks[n], n))[:POOL_K]

        seed = rnd * 100_003 + 20261002
        a20, _ = unique_structured_bundle(pool, A_TICKETS, seed + 1)
        b20, _ = portfolio_bundle(pool, scores, seed + 2)

        # Reproduce the existing 20-ticket 10+10 hybrid.
        a20_ok = success(a20, actual)
        b20_ok = success(b20, actual)
        h20, _ = merge_unique_budget(select(a20, 10), select(b20, 10), b20, 20)
        h20_ok = success(h20, actual)
        obs["20"]["structured"] += int(a20_ok)
        obs["20"]["portfolio"] += int(b20_ok)
        obs["20"]["hybrid_10_10"] += int(h20_ok)
        obs["20"]["hybrid_gt_both"] += int(h20_ok and not a20_ok and not b20_ok)

        # Evidence reduction: same source arms, half the ticket budget.
        a10 = select(a20, 10)
        b10 = select(b20, 10)
        h10, repairs = merge_unique_budget(select(a20, 5), select(b20, 5), b20, 10)

        a10_ok = success(a10, actual)
        b10_ok = success(b10, actual)
        h10_ok = success(h10, actual)

        obs["10"]["structured"] += int(a10_ok)
        obs["10"]["portfolio"] += int(b10_ok)
        obs["10"]["hybrid_5_5"] += int(h10_ok)
        obs["10"]["hybrid_gt_both"] += int(h10_ok and not a10_ok and not b10_ok)
        obs["10"]["collision_repairs"] += repairs

        exclusive = a10_ok != b10_ok
        obs["10"]["exclusive_total"] += int(exclusive)
        obs["10"]["exclusive_recovered"] += int(exclusive and h10_ok)

    payload = {
        "experiment": "loto7_evidence_reduction_budget10_v0",
        "evaluation_rounds": int(len(df) - MODEL_WINDOW),
        "question": "Does success-set expansion survive when total ticket budget is halved from 20 to 10?",
        "frozen": [
            "Same 596 target rounds.",
            "Same CORE18 construction.",
            "Same Structured20 and Portfolio20 source generators.",
            "Same deterministic spread-index selector.",
            "No result-informed tuning.",
        ],
        "reduction": {
            "from": "20 tickets: Structured20 / Portfolio20 / Hybrid 10+10",
            "to": "10 tickets: Structured10 / Portfolio10 / Hybrid 5+5",
        },
        "observed": obs,
        "criterion": "At budget 10, Hybrid 5+5 retains the phenomenon only if its 5+ success-round set extends beyond both same-budget single-arm sets.",
        "boundary": [
            "This is one reduction probe on the original 596-round world, not a World-B replication.",
            "Failure at 10 tickets would not prove that every 10-ticket construction fails.",
            "Survival at 10 tickets would show that 20 tickets are not necessary for this specific construction to exhibit success-set expansion.",
        ],
    }

    out = Path("results/loto7_evidence_reduction_budget10_v0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== EVIDENCE REDUCTION BUDGET10 v0 ===")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    print("saved ->", out)


if __name__ == "__main__":
    main()
