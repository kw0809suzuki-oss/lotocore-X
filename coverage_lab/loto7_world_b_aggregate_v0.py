#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

RATIOS = ("5+15", "8+12", "10+10", "12+8", "15+5")
WORLD_A = {
    "structured20_5plus_rounds": 11,
    "portfolio20_5plus_rounds": 12,
    "ratios": {"5+15": 14, "8+12": 13, "10+10": 14, "12+8": 12, "15+5": 13},
    "exclusive_recovered": {"5+15": 9, "8+12": 8, "10+10": 9, "12+8": 7, "15+5": 8},
}


def q(vals, p):
    xs = sorted(vals)
    if not xs:
        return None
    pos = (len(xs) - 1) * p
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(vals):
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "q05": q(vals, 0.05),
        "q25": q(vals, 0.25),
        "q75": q(vals, 0.75),
        "q95": q(vals, 0.95),
        "min": min(vals),
        "max": max(vals),
    }


def main():
    files = sorted(Path("artifacts").glob("**/world_b_shard_*.json"))
    if not files:
        files = sorted(Path("results/world_b").glob("world_b_shard_*.json"))
    worlds = []
    meta = None
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = meta or payload["world_definition"]
        worlds.extend(payload["worlds"])
    worlds.sort(key=lambda x: x["world_id"])

    if len(worlds) != 500:
        raise RuntimeError(f"expected 500 worlds, got {len(worlds)}")
    if [w["world_id"] for w in worlds] != list(range(1, 501)):
        raise RuntimeError("world ids are not exactly 1..500")

    summary = {
        "experiment": "loto7_world_b_500_v0",
        "world_definition": meta,
        "world_a_reference": WORLD_A,
        "world_count": len(worlds),
        "baseline_distribution": {
            "structured20_5plus_rounds": stats([w["structured20_5plus_rounds"] for w in worlds]),
            "portfolio20_5plus_rounds": stats([w["portfolio20_5plus_rounds"] for w in worlds]),
            "best_single_5plus_rounds": stats([w["best_single_5plus_rounds"] for w in worlds]),
            "exclusive_total": stats([w["exclusive_total"] for w in worlds]),
        },
        "ratios": {},
        "any_ratio_expansion": {
            "worlds": sum(any(w["ratios"][r]["expanded_success_set"] for r in RATIOS) for w in worlds),
        },
        "boundary": [
            "World B contains 500 independent Uniform 7-of-37 worlds, each 100 warmup + 596 evaluation draws.",
            "World A ticket generators, role split, hybrid ratios, selection and collision repair are imported unchanged.",
            "No World B outcome is used to tune a generator, ratio, threshold, or seed.",
            "Expanded success set means hybrid 5+ successful-round count is strictly greater than the better of Structured20 and Portfolio20 in the same world.",
            "This measures reproduction under a null/random world; it does not establish causality or predictive skill."
        ],
    }
    summary["any_ratio_expansion"]["rate"] = summary["any_ratio_expansion"]["worlds"] / len(worlds)

    for r in RATIOS:
        hits = [w["ratios"][r]["fiveplus_rounds"] for w in worlds]
        deltas = [w["ratios"][r]["delta_vs_best_single"] for w in worlds]
        recovered = [w["ratios"][r]["exclusive_recovered"] for w in worlds]
        pos = sum(d > 0 for d in deltas)
        zero = sum(d == 0 for d in deltas)
        neg = sum(d < 0 for d in deltas)
        summary["ratios"][r] = {
            "fiveplus_rounds": stats(hits),
            "delta_vs_best_single": stats(deltas),
            "exclusive_recovered": stats(recovered),
            "expansion_worlds": pos,
            "equal_worlds": zero,
            "contraction_worlds": neg,
            "expansion_rate": pos / len(worlds),
        }

    outdir = Path("results/world_b")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "loto7_world_b_500_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (outdir / "loto7_world_b_500_worlds.json").write_text(
        json.dumps(worlds, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("=== LOTO7 WORLD B 500 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
