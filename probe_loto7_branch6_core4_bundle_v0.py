from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from probe_loto7_transition_branch10_v0 import build_branch10, max_match
from compare_loto7_branch10_core10_random10 import make_structured10, core18, random_bundle

DATA = Path("data/loto7.csv")
OUT_CSV = Path("results/loto7_branch6_core4_bundle_v0.csv")
OUT_JSON = Path("results/loto7_branch6_core4_bundle_v0_summary.json")
RANDOM_REPS = 1000

# Fixed before observing outcomes. Spread selections across each independent 10-ticket bundle.
BRANCH_POSITIONS = (0, 2, 4, 5, 7, 9)
CORE_POSITIONS = (0, 3, 6, 9)

def pick(bundle, positions):
    return [bundle[i] for i in positions]

def summarize(out, key):
    s = out[key]
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
    cols = [f"n{i}" for i in range(1,8)]
    draws = [{"round": int(r["round"]), "nums": [int(r[c]) for c in cols]} for _,r in df.iterrows()]
    rows = []

    for target_i in range(100, len(draws)):
        prior = draws[:target_i]
        branch = build_branch10(prior)
        if branch is None:
            continue

        target = draws[target_i]
        actual = target["nums"]

        c18 = core18(df.iloc[:target_i])
        core10 = make_structured10(c18, target["round"] * 2654435761 + 11)

        branch6 = pick(branch["tickets"], BRANCH_POSITIONS)
        core4 = pick(core10, CORE_POSITIONS)
        bundle64 = branch6 + core4

        rng = random.Random(target["round"] * 1000003 + 20261001)
        rr = [max_match(actual, random_bundle(rng)) for _ in range(RANDOM_REPS)]

        rows.append({
            "round": target["round"],
            "bundle64_max_match": max_match(actual, bundle64),
            "branch10_max_match": max_match(actual, branch["tickets"]),
            "core10_max_match": max_match(actual, core10),
            "random_mean_max_match": sum(rr)/RANDOM_REPS,
            "random_rate_3plus": sum(x>=3 for x in rr)/RANDOM_REPS,
            "random_rate_4plus": sum(x>=4 for x in rr)/RANDOM_REPS,
            "random_rate_5plus": sum(x>=5 for x in rr)/RANDOM_REPS,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "comparison": "Branch6 + Core4 bundle v0 vs Branch10 vs Core10 vs Random10",
        "evaluated_rounds": len(out),
        "history_round_max": int(df["round"].max()),
        "random_reps_per_round": RANDOM_REPS,
        "selection": {
            "branch_positions_zero_based": list(BRANCH_POSITIONS),
            "core_positions_zero_based": list(CORE_POSITIONS),
            "rule": "fixed evenly-spread positions selected before observing outcomes"
        },
        "bundle64": summarize(out, "bundle64_max_match"),
        "branch10": summarize(out, "branch10_max_match"),
        "core10": summarize(out, "core10_max_match"),
        "random10": {
            "mean_max_match": float(out["random_mean_max_match"].mean()),
            "rate_3plus": float(out["random_rate_3plus"].mean()),
            "rate_4plus": float(out["random_rate_4plus"].mean()),
            "rate_5plus": float(out["random_rate_5plus"].mean()),
        },
        "paired_bundle64_vs_branch10": {
            "bundle64_gt": int((out.bundle64_max_match > out.branch10_max_match).sum()),
            "equal": int((out.bundle64_max_match == out.branch10_max_match).sum()),
            "bundle64_lt": int((out.bundle64_max_match < out.branch10_max_match).sum()),
        },
        "paired_bundle64_vs_core10": {
            "bundle64_gt": int((out.bundle64_max_match > out.core10_max_match).sum()),
            "equal": int((out.bundle64_max_match == out.core10_max_match).sum()),
            "bundle64_lt": int((out.bundle64_max_match < out.core10_max_match).sum()),
        },
        "boundary": [
            "Branch10 and Core10 are generated independently and are not altered internally.",
            "The 6/4 bundle uses fixed predeclared ticket positions; no outcome-based subset selection is used.",
            "This is one allocation probe only; no search over 5/5, 7/3, or ticket-position combinations was performed.",
            "Historical walk-forward comparison does not establish future lottery advantage."
        ]
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
