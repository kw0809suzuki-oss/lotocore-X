from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import lotocore

from probe_loto7_transition_branch10_v0 import build_branch10, max_match

DATA = Path("data/loto7.csv")
OUT_CSV = Path("results/loto7_branch10_core10_random10.csv")
OUT_JSON = Path("results/loto7_branch10_core10_random10_summary.json")
RANDOM_REPS = 1000

def seeded(seed: int):
    x = seed & 0xFFFFFFFF
    while True:
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        yield x / 4294967296.0

def make_structured10(candidates: list[int], seed: int) -> list[list[int]]:
    """Python port of the current Floot LOTO7 makeStructured10 allocator."""
    if len(candidates) != 18:
        raise ValueError("Structured10 requires exactly 18 candidates")
    rng = seeded(seed)
    target = {n: 4 if i < 16 else 3 for i, n in enumerate(candidates)}
    rem = dict(target)
    pair_count: dict[tuple[int, int], int] = {}
    out: list[list[int]] = []

    for slot in range(10):
        tickets_left = 10 - slot
        forced = [n for n in candidates if rem.get(n, 0) == tickets_left]
        chosen = list(forced)
        while len(chosen) < 7:
            pool = [n for n in candidates if rem.get(n, 0) > 0 and n not in chosen]
            rows = []
            for n in pool:
                pair_penalty = sum(pair_count.get(tuple(sorted((n, x))), 0) for x in chosen)
                rows.append((n, pair_penalty, rem.get(n, 0)))
            min_pair = min(r[1] for r in rows)
            max_remaining = max(r[2] for r in rows if r[1] == min_pair)
            tied = sorted(r[0] for r in rows if r[1] == min_pair and r[2] == max_remaining)
            pick = tied[int(next(rng) * len(tied))]
            chosen.append(pick)

        ticket = sorted(chosen)
        out.append(ticket)
        for n in ticket:
            rem[n] -= 1
        for i in range(len(ticket)):
            for j in range(i + 1, len(ticket)):
                k = tuple(sorted((ticket[i], ticket[j])))
                pair_count[k] = pair_count.get(k, 0) + 1

    return out

def core18(history: pd.DataFrame) -> list[int]:
    snap = lotocore.score_snapshot(history.tail(100))
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return sorted(range(1, 38), key=lambda n: (ranks[n], n))[:18]

def random_bundle(rng: random.Random) -> list[list[int]]:
    out, seen = [], set()
    while len(out) < 10:
        t = tuple(sorted(rng.sample(range(1, 38), 7)))
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out

def summarize(df: pd.DataFrame, key: str) -> dict:
    s = df[key]
    return {
        "mean_max_match": float(s.mean()),
        "rate_3plus": float((s >= 3).mean()),
        "rate_4plus": float((s >= 4).mean()),
        "rate_5plus": float((s >= 5).mean()),
        "count_3plus": int((s >= 3).sum()),
        "count_4plus": int((s >= 4).sum()),
        "count_5plus": int((s >= 5).sum()),
    }

def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    cols = [f"n{i}" for i in range(1, 8)]
    draws = [{"round": int(r["round"]), "nums": [int(r[c]) for c in cols]} for _, r in df.iterrows()]
    rows = []

    for target_i in range(100, len(draws)):
        branch = build_branch10(draws[:target_i])
        if branch is None:
            continue

        target = draws[target_i]
        actual = target["nums"]
        history_df = df.iloc[:target_i]
        c18 = core18(history_df)
        core_tickets = make_structured10(c18, target["round"] * 2654435761 + 11)

        branch_mm = max_match(actual, branch["tickets"])
        core_mm = max_match(actual, core_tickets)

        r3 = r4 = r5 = 0
        rmean = 0.0
        rng = random.Random(target["round"] * 1000003 + 20261001)
        for _ in range(RANDOM_REPS):
            mm = max_match(actual, random_bundle(rng))
            rmean += mm
            r3 += mm >= 3
            r4 += mm >= 4
            r5 += mm >= 5

        rows.append({
            "round": target["round"],
            "orbit": " -> ".join(branch["orbit"]),
            "historical_matches": branch["matches"],
            "branch10_max_match": branch_mm,
            "core10_max_match": core_mm,
            "random_mean_max_match": rmean / RANDOM_REPS,
            "random_rate_3plus": r3 / RANDOM_REPS,
            "random_rate_4plus": r4 / RANDOM_REPS,
            "random_rate_5plus": r5 / RANDOM_REPS,
            "core18": "-".join(f"{n:02d}" for n in c18),
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    branch_s = summarize(out, "branch10_max_match")
    core_s = summarize(out, "core10_max_match")
    random_s = {
        "mean_max_match": float(out["random_mean_max_match"].mean()),
        "rate_3plus": float(out["random_rate_3plus"].mean()),
        "rate_4plus": float(out["random_rate_4plus"].mean()),
        "rate_5plus": float(out["random_rate_5plus"].mean()),
    }

    summary = {
        "comparison": "Transition Branch 10 vs LOTO CORE 10 proxy vs Uniform Random 10",
        "evaluated_rounds": n,
        "history_round_min": int(df["round"].min()),
        "history_round_max": int(df["round"].max()),
        "random_reps_per_round": RANDOM_REPS,
        "definitions": {
            "branch10": "Current 2-transition orbit -> historical next-branch allocation -> nearest historical origin-state next ticket.",
            "core10": "Repository LOTO Core score_snapshot -> top18 -> current Floot makeStructured10 allocator port.",
            "random10": "10 unique uniform random 7-number tickets; 1000 bundles per target round.",
        },
        "branch10": branch_s,
        "core10": core_s,
        "random10": random_s,
        "paired_branch_vs_core": {
            "branch_gt_core": int((out["branch10_max_match"] > out["core10_max_match"]).sum()),
            "equal": int((out["branch10_max_match"] == out["core10_max_match"]).sum()),
            "branch_lt_core": int((out["branch10_max_match"] < out["core10_max_match"]).sum()),
        },
        "boundary": [
            "CORE10 here is a deterministic repository proxy, not the Floot AI-generated CORE18 endpoint.",
            "Branch10 v0 copies historical analog next-draw tickets and does not force copied ticket geometry relative to the target current state.",
            "Historical walk-forward comparison does not establish future lottery advantage.",
        ],
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
