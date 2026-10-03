#!/usr/bin/env python3
# Full-history observation trigger: logic unchanged.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    MODEL_WINDOW,
    POOL_K,
    avg_candidate_score,
    euclid,
    portfolio_bundle,
    sampled_combinations,
    standardized_vectors,
)


DEFAULT_ROUNDS = (134, 249, 374)


def parse_rounds(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("at least one round is required")
    return vals


def observe_round(df: pd.DataFrame, rnd: int, model_window: int) -> tuple[list[dict], dict]:
    matches = df.index[df["round"].astype(int) == rnd].tolist()
    if len(matches) != 1:
        raise ValueError(f"round {rnd} not found exactly once")
    idx = matches[0]
    if idx < model_window:
        raise ValueError(f"round {rnd} does not have {model_window} prior rows")

    history = df.iloc[idx - model_window:idx]
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    scores = {int(k): float(v) for k, v in snap["scores"].items()}
    ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))
    pool = ranked[:POOL_K]

    seed = rnd * 100_003 + 20261002
    selected, trace = portfolio_bundle(pool, scores, seed + 2)

    role_by_ticket = {}
    for role in ("core", "diversify", "falsify"):
        for t in trace[role]:
            role_by_ticket[tuple(t)] = role

    combos = sampled_combinations(pool, seed + 2 + 5000, [tuple(t) for t in trace["core"]])
    z = standardized_vectors(combos)
    combo_score = {c: avg_candidate_score(c, scores) for c in combos}
    median_score = sorted(combo_score.values())[len(combo_score) // 2]

    core = [tuple(t) for t in trace["core"]]
    core_centroid = tuple(
        sum(z[t][i] for t in core) / len(core)
        for i in range(len(z[core[0]]))
    )

    ds = {t: euclid(z[t], core_centroid) for t in selected}
    sorted_d = sorted(selected, key=lambda t: (ds[t], t))
    d_rank = {t: i + 1 for i, t in enumerate(sorted_d)}

    sorted_m = sorted(selected, key=lambda t: (combo_score[t], t))
    m_rank = {t: i + 1 for i, t in enumerate(sorted_m)}

    rows = []
    for ticket_id, t in enumerate(selected, start=1):
        role = role_by_ticket.get(t, "unknown")
        rows.append(
            {
                "round_id": rnd,
                "ticket_id": ticket_id,
                "role_id": role,
                "ticket": "-".join(f"{n:02d}" for n in t),
                "distance_from_center_d": ds[t],
                "distance_rank_near_to_far": d_rank[t],
                "material_retention_m": combo_score[t],
                "material_retention_rank_low_to_high": m_rank[t],
                "material_vs_round_median": (
                    "above_or_equal" if combo_score[t] >= median_score else "below"
                ),
                "hypothesis_exit_vector_q": "unknown",
                "freedom_type": "unknown",
            }
        )

    role_summary = {}
    for role in ("core", "diversify", "falsify"):
        rr = [r for r in rows if r["role_id"] == role]
        role_summary[role] = {
            "ticket_count": len(rr),
            "mean_d": sum(r["distance_from_center_d"] for r in rr) / len(rr),
            "min_d": min(r["distance_from_center_d"] for r in rr),
            "max_d": max(r["distance_from_center_d"] for r in rr),
            "mean_m": sum(r["material_retention_m"] for r in rr) / len(rr),
            "m_above_or_equal_median": sum(
                r["material_vs_round_median"] == "above_or_equal" for r in rr
            ),
        }

    manifest = {
        "round_id": rnd,
        "history_end_round": int(history.iloc[-1]["round"]),
        "candidate_pool": pool,
        "center_definition": "centroid of existing Core tickets in existing standardized structural space",
        "d_source": "existing centroid distance using standardized_vectors + euclid",
        "m_source": "existing avg_candidate_score / combo_score",
        "q_source": "unknown: no direct existing value maps to hypothesis-exit vector",
        "generation_freedom": {
            "neighborhood": {
                "present": "unknown",
                "reason": "v0 observes d but does not invent an absolute/relative threshold",
            },
            "structural_displacement": {
                "present": "unknown",
                "reason": "v0 observes d and m but does not invent a freedom threshold",
            },
            "hypothesis_exit": {
                "present": "unknown",
                "reason": "q has no direct existing implementation value",
                "persistence": "unknown",
            },
        },
        "role_summary_observation_only": role_summary,
        "boundary": [
            "No winning numbers are used to generate the tickets or d/m observations.",
            "Existing portfolio generation logic is imported unchanged.",
            "role_id is descriptive only and is not used to assign freedom_type.",
            "q and freedom_type remain unknown rather than being inferred from role labels.",
        ],
    }
    return rows, manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("coverage_lab/fixtures/loto7_1_696.csv"))
    p.add_argument("--rounds", default=",".join(map(str, DEFAULT_ROUNDS)))
    p.add_argument("--model-window", type=int, default=MODEL_WINDOW)
    p.add_argument("--out-csv", type=Path, default=Path("results/generation_freedom_observer_v0.csv"))
    p.add_argument("--out-json", type=Path, default=Path("results/generation_freedom_observer_v0.json"))
    a = p.parse_args()

    df = pd.read_csv(a.data).sort_values("round").reset_index(drop=True)
    all_rows = []
    manifests = []
    for rnd in parse_rounds(a.rounds):
        rows, manifest = observe_round(df, rnd, a.model_window)
        all_rows.extend(rows)
        manifests.append(manifest)

    out = pd.DataFrame(all_rows)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv, index=False)

    payload = {
        "experiment": "generation_freedom_observer_v0",
        "mode": "observation_only",
        "rounds": manifests,
        "boundary": {
            "generation_logic_changed": False,
            "new_predictor_added": False,
            "new_score_added": False,
            "d_reuses_existing_value": True,
            "m_reuses_existing_value": True,
            "q_status": "unknown",
        },
    }
    a.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== GENERATION FREEDOM OBSERVER v0 ===")
    print(f"rounds={','.join(map(str, parse_rounds(a.rounds)))}")
    for m in manifests:
        print("ROUND", m["round_id"], json.dumps(m["role_summary_observation_only"], ensure_ascii=False, sort_keys=True))
    print("BOUNDARY", json.dumps(payload["boundary"], ensure_ascii=False, sort_keys=True))
    print(f"saved -> {a.out_csv}")
    print(f"saved -> {a.out_json}")


if __name__ == "__main__":
    main()
