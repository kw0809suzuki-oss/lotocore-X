#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

WORLD_IDS = list(range(1, 11))
THRESHOLDS = (3, 4, 5)


def rate(successes: int, tickets: int) -> float:
    return successes / tickets if tickets else 0.0


def main() -> None:
    files = sorted(Path("artifacts").glob("**/world_*.json"))
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    worlds = [p["world"] for p in payloads]
    worlds.sort(key=lambda w: w["world_id"])

    got = [w["world_id"] for w in worlds]
    if got != WORLD_IDS:
        raise RuntimeError(f"expected worlds {WORLD_IDS}, got {got}")

    out = {
        "experiment": "diversify_face_outer10_v0",
        "world_ids": WORLD_IDS,
        "world_count": 10,
        "total_evaluation_rounds": sum(w["evaluation_rounds"] for w in worlds),
        "total_diversify_tickets": sum(w["diversify_tickets"] for w in worlds),
        "hypothesis_frozen": payloads[0]["hypothesis_frozen"],
        "thresholds": {},
        "per_world_5plus": [],
        "boundary": payloads[0]["boundary"],
    }

    for th in THRESHOLDS:
        k = str(th)
        ft = sum(w["thresholds"][k]["face_tickets"] for w in worlds)
        fs = sum(w["thresholds"][k]["face_successes"] for w in worlds)
        nt = sum(w["thresholds"][k]["nonface_tickets"] for w in worlds)
        ns = sum(w["thresholds"][k]["nonface_successes"] for w in worlds)
        fr = rate(fs, ft)
        nr = rate(ns, nt)
        out["thresholds"][f"{th}plus"] = {
            "face_tickets": ft,
            "face_successes": fs,
            "face_rate": fr,
            "nonface_tickets": nt,
            "nonface_successes": ns,
            "nonface_rate": nr,
            "rate_ratio": (fr / nr if nr > 0 else None),
            "rate_difference": fr - nr,
        }

    for w in worlds:
        x = w["thresholds"]["5"]
        fr = rate(x["face_successes"], x["face_tickets"])
        nr = rate(x["nonface_successes"], x["nonface_tickets"])
        out["per_world_5plus"].append({
            "world_id": w["world_id"],
            "face_tickets": x["face_tickets"],
            "face_successes": x["face_successes"],
            "nonface_tickets": x["nonface_tickets"],
            "nonface_successes": x["nonface_successes"],
            "face_rate": fr,
            "nonface_rate": nr,
            "face_rate_gt_nonface": fr > nr,
            "face_rate_eq_nonface": fr == nr,
        })

    out["worlds_5plus_face_rate_gt_nonface"] = sum(x["face_rate_gt_nonface"] for x in out["per_world_5plus"])
    out["worlds_5plus_equal"] = sum(x["face_rate_eq_nonface"] for x in out["per_world_5plus"])
    out["worlds_5plus_face_rate_lt_nonface"] = 10 - out["worlds_5plus_face_rate_gt_nonface"] - out["worlds_5plus_equal"]

    dest = Path("results/diversify_face_outer10_v0/summary.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("saved ->", dest)


if __name__ == "__main__":
    main()
