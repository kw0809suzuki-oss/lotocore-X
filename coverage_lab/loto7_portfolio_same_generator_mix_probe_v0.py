#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW, POOL_K,
    main_numbers, portfolio_bundle, bundle_metrics,
)

EVEN = list(range(0, 20, 2))

def merge_unique(a_pick, b_pick, b20):
    chosen = list(a_pick)
    used = set(chosen)
    preferred_set = set(b_pick)
    leftovers = [t for t in b20 if t not in preferred_set]
    repairs = 0
    for t in b_pick:
        if t not in used:
            pick = t
        else:
            repairs += 1
            pick = next((x for x in leftovers if x not in used), None)
            if pick is None:
                raise RuntimeError("no unused B20 replacement")
            leftovers.remove(pick)
        chosen.append(pick)
        used.add(pick)
    if len(chosen) != 20 or len(set(chosen)) != 20:
        raise RuntimeError("mixed bundle is not 20 unique tickets")
    return chosen, repairs

def main():
    df = pd.read_csv("coverage_lab/fixtures/loto7_1_696.csv").sort_values("round").reset_index(drop=True)

    a_success = set()
    b_success = set()
    mix_success = set()
    repairs = 0

    for idx in range(MODEL_WINDOW, len(df)):
        history = df.iloc[idx-MODEL_WINDOW:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        pool = sorted(range(1, 38), key=lambda n: (ranks[n], n))[:POOL_K]

        base_seed = rnd * 100_003 + 20261002
        a20, _ = portfolio_bundle(pool, scores, base_seed + 2)
        b20, _ = portfolio_bundle(pool, scores, base_seed + 102)

        mix20, rep = merge_unique(
            [a20[i] for i in EVEN],
            [b20[i] for i in EVEN],
            b20,
        )
        repairs += rep

        if bundle_metrics(a20, actual)["tickets_ge5"] > 0:
            a_success.add(rnd)
        if bundle_metrics(b20, actual)["tickets_ge5"] > 0:
            b_success.add(rnd)
        if bundle_metrics(mix20, actual)["tickets_ge5"] > 0:
            mix_success.add(rnd)

    payload = {
        "experiment": "loto7_portfolio_same_generator_mix_probe_v0",
        "question": "Can two Portfolio worlds with different fixed RNG seeds produce different success sets, and can a fixed 10+10 mix broaden 5+ successful rounds?",
        "source_a": {
            "seed_rule": "base_seed + 2",
            "fiveplus_rounds": len(a_success),
            "success_rounds": sorted(a_success),
        },
        "source_b": {
            "seed_rule": "base_seed + 102",
            "fiveplus_rounds": len(b_success),
            "success_rounds": sorted(b_success),
        },
        "nonalignment": {
            "a_only": sorted(a_success - b_success),
            "b_only": sorted(b_success - a_success),
            "both": sorted(a_success & b_success),
            "exclusive_total": len(a_success ^ b_success),
            "union_total": len(a_success | b_success),
        },
        "mix_10_plus_10": {
            "indices_from_each": EVEN,
            "fiveplus_rounds": len(mix_success),
            "success_rounds": sorted(mix_success),
            "collision_repairs": repairs,
            "vs_a_added": sorted(mix_success - a_success),
            "vs_a_lost": sorted(a_success - mix_success),
            "vs_b_added": sorted(mix_success - b_success),
            "vs_b_lost": sorted(b_success - mix_success),
        },
        "boundary": [
            "Both source bundles use the same Portfolio generator and same CORE18 pool.",
            "Only RNG seed differs: +2 versus +102, fixed before reading outcomes.",
            "Mix is fixed 10+10 using even positions from each source.",
            "Total ticket budget is 20 and all mixed tickets are unique.",
            "No realized draw information enters generation, selection, or collision repair.",
            "This is one exploratory reproduction probe, not a tuned seed search."
        ],
    }

    out = Path("results/loto7_portfolio_same_generator_mix_probe_v0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("A", len(a_success), sorted(a_success))
    print("B", len(b_success), sorted(b_success))
    print("EXCLUSIVE", len(a_success ^ b_success), sorted(a_success ^ b_success))
    print("MIX", len(mix_success), sorted(mix_success))
    print("saved ->", out)

if __name__ == "__main__":
    main()
