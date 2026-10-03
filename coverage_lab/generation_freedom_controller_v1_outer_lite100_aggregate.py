#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path

WORLD_COUNT = 100
NAMES = ["A_centroid", "B_core_static", "C_dynamic", "D_controller_v1"]


def main():
    files = sorted(Path("artifacts").glob("**/shard_*.json"))
    worlds = []
    meta = None
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        meta = meta or payload["world_source"]
        worlds.extend(payload["worlds"])

    worlds.sort(key=lambda x: x["world_id"])
    if len(worlds) != WORLD_COUNT:
        raise RuntimeError(f"expected {WORLD_COUNT} worlds, got {len(worlds)}")
    if [w["world_id"] for w in worlds] != list(range(1, WORLD_COUNT + 1)):
        raise RuntimeError("world ids are not exactly 1..100")
    if not all(w["fidelity_ok"] for w in worlds):
        raise RuntimeError("fidelity failure in outer worlds")

    summary = {
        "experiment": "generation_freedom_controller_v1_outer_lite100",
        "world_source": meta,
        "world_count": WORLD_COUNT,
        "total_external_evaluation_rounds": WORLD_COUNT * 596,
        "success16": {},
        "success6": {},
        "pairwise_vs_D": {},
        "boundary": [
            "Worlds are the pre-existing deterministic World C Lite100 family.",
            "100 worlds x 596 evaluation rounds = 59,600 external/null-world evaluation points.",
            "This tests structural behavior outside the real-history 596-round sample; it does not establish predictive skill.",
            "No tuning is permitted from this result.",
        ],
    }

    for n in NAMES:
        vals16 = [w["success16_counts"][n] for w in worlds]
        vals6 = [w["success6_counts"][n] for w in worlds]
        summary["success16"][n] = {
            "mean_per_world": statistics.fmean(vals16),
            "median_per_world": statistics.median(vals16),
            "min": min(vals16),
            "max": max(vals16),
        }
        summary["success6"][n] = {
            "mean_per_world": statistics.fmean(vals6),
            "median_per_world": statistics.median(vals6),
            "min": min(vals6),
            "max": max(vals6),
        }

    for other in ("A_centroid", "B_core_static", "C_dynamic"):
        key = f"D_vs_{other}"
        d_only = [w["pairwise"][key]["D_only"] for w in worlds]
        o_only = [w["pairwise"][key]["other_only"] for w in worlds]
        union = [w["pairwise"][key]["union"] for w in worlds]
        d_counts = [w["success16_counts"]["D_controller_v1"] for w in worlds]
        o_counts = [w["success16_counts"][other] for w in worlds]
        summary["pairwise_vs_D"][other] = {
            "worlds_D_has_unique_success": sum(v > 0 for v in d_only),
            "worlds_other_has_unique_success": sum(v > 0 for v in o_only),
            "mean_D_only_rounds": statistics.fmean(d_only),
            "mean_other_only_rounds": statistics.fmean(o_only),
            "mean_union_rounds": statistics.fmean(union),
            "worlds_D_more_success_rounds": sum(d > o for d, o in zip(d_counts, o_counts)),
            "worlds_equal_success_rounds": sum(d == o for d, o in zip(d_counts, o_counts)),
            "worlds_D_fewer_success_rounds": sum(d < o for d, o in zip(d_counts, o_counts)),
        }

    outdir = Path("results/controller_v1_outer_lite100")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
