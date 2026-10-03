from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRUCTURES = ROOT / "models" / "loto7_astra_observation" / "structures" / "astra_structures_v0.json"

SUPPORTED_NOW = {
    "exact_gaps",
    "point",
    "blank_interval",
}

def classify_structure(s: dict) -> dict:
    rep = s["representation"]

    if s["id"] == "S1_partial_gap_lifetime":
        return {
            "status": "PARTIAL",
            "reason": "exact gaps are supported, but duration/break lifecycle is only partly represented by Finite Preservation State and not wired to the generator"
        }

    if s["id"] == "S2_temporal_path_coexistence":
        return {
            "status": "SUPPORTED_OBSERVATION",
            "reason": "explicit multi-draw presence paths are now represented and testable in temporal_paths.py; this is not yet a generation rule"
        }

    if s["id"] == "S3_point_vs_context":
        return {
            "status": "PARTIAL",
            "reason": "point preservation exists, but mutable neighbor-context tracking is not a first-class structure"
        }

    if s["id"] == "S4_blank_reentry_parallel_persistence":
        return {
            "status": "PARTIAL",
            "reason": "blank and point constraints exist separately; coordinated lifecycle across draws is not yet modeled"
        }

    if s["id"] == "S5_partial_shape_scope":
        return {
            "status": "SUPPORTED_OBSERVATION",
            "reason": "partial gap subsequences and uniform translated subsets are now observable in partial_gap_scope.py; this is not a whole-layout preserve rule"
        }

    return {"status": "UNKNOWN", "reason": "unclassified"}


def main() -> None:
    data = json.loads(STRUCTURES.read_text(encoding="utf-8"))
    rows = []
    for s in data["structures"]:
        rows.append({
            "id": s["id"],
            **classify_structure(s),
        })
    print(json.dumps({
        "model": "LOTO7 Astra Observation Model",
        "structure_count": len(rows),
        "classification": rows,
        "boundary": "This is an implementation-gap audit, not a decision to extend the generator."
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
