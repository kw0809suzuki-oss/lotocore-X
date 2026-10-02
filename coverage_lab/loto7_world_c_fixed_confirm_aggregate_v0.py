#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

def main():
    files = sorted(Path("artifacts").glob("**/world_c_shard_*.json"))
    worlds = []
    meta = None
    for f in files:
        p = json.loads(f.read_text(encoding="utf-8"))
        meta = meta or p
        worlds.extend(p["worlds"])
    worlds.sort(key=lambda x: x["world_id"])
    if len(worlds) != 500 or [w["world_id"] for w in worlds] != list(range(1,501)):
        raise RuntimeError(f"expected exact worlds 1..500, got {len(worlds)}")

    hits = [w for w in worlds if w["fixed_condition"]]
    out = {
        "experiment": "loto7-world-c-fixed-confirm-v0",
        "pre_fixed_condition": "N_plus >= 4 AND delta_min >= 0",
        "world_count": 500,
        "condition_hits": len(hits),
        "condition_rate": len(hits)/500,
        "hit_world_ids": [w["world_id"] for w in hits],
        "boundary": [
            "This is an independent confirmation sample, not the 500-world discovery sample.",
            "The condition was fixed before this sample: N_plus >= 4 AND delta_min >= 0.",
            "No additional statistic or threshold is used for the confirmation decision."
        ]
    }
    Path("results/world_c_confirm").mkdir(parents=True, exist_ok=True)
    Path("results/world_c_confirm/loto7_world_c_fixed_confirm_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
