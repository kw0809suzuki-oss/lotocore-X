from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data/loto7.csv")
OUT_CSV = Path("results/loto7_transition_branch10_v0.csv")
OUT_JSON = Path("results/loto7_transition_branch10_v0_summary.json")
MIN_MATCHES = 10
RANDOM_REPS = 1000

def state(nums):
    xs = sorted(nums)
    return {"center": sum(xs) / 7.0, "range": xs[-1] - xs[0]}

def transition(a, b):
    dc = b["center"] - a["center"]
    dr = b["range"] - a["range"]
    vertical = "HIGH" if dc > 0 else "LOW" if dc < 0 else "FLAT"
    spread = "OPEN" if dr > 0 else "CLOSE" if dr < 0 else "SAME"
    return f"{vertical}+{spread}"

def allocate(counts, slots=10):
    total = sum(counts.values())
    rows = []
    for key, count in counts.items():
        raw = count * slots / total
        rows.append([key, count, raw, int(raw)])
    left = slots - sum(r[3] for r in rows)
    rows.sort(key=lambda r: (-(r[2]-r[3]), -r[1], r[0]))
    for i in range(left):
        rows[i][3] += 1
    return [(r[0], r[3]) for r in rows if r[3] > 0]

def distance(a, b):
    return abs(a["center"] - b["center"]) / 36.0 + abs(a["range"] - b["range"]) / 36.0

def build_branch10(prior):
    states = [state(d["nums"]) for d in prior]
    trs = [transition(states[i], states[i+1]) for i in range(len(states)-1)]
    if len(trs) < 2:
        return None
    orbit = (trs[-2], trs[-1])
    current_state = states[-1]
    examples = []
    for i in range(1, len(trs)-1):
        if trs[i-1] == orbit[0] and trs[i] == orbit[1]:
            examples.append({
                "branch": trs[i+1],
                "origin": states[i+1],
                "ticket": sorted(prior[i+2]["nums"]),
                "round": prior[i+2]["round"],
            })
    if len(examples) < MIN_MATCHES:
        return None
    counts = Counter(e["branch"] for e in examples)
    allocation = allocate(counts, 10)
    tickets = []
    for branch, slots in allocation:
        pool = [e for e in examples if e["branch"] == branch]
        pool.sort(key=lambda e: (distance(e["origin"], current_state), -e["round"]))
        tickets.extend(e["ticket"] for e in pool[:slots])
    if len(tickets) != 10:
        return None
    return {"orbit": orbit, "matches": len(examples), "allocation": allocation, "tickets": tickets}

def max_match(actual, tickets):
    actual = set(actual)
    return max(sum(n in actual for n in t) for t in tickets)

def random_bundle(seed):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < 10:
        t = tuple(sorted(rng.sample(range(1, 38), 7)))
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out

def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    cols = [f"n{i}" for i in range(1, 8)]
    draws = [{"round": int(r["round"]), "nums": [int(r[c]) for c in cols]} for _, r in df.iterrows()]
    rows = []
    for target_i in range(4, len(draws)):
        model = build_branch10(draws[:target_i])
        if model is None:
            continue
        target = draws[target_i]
        mm = max_match(target["nums"], model["tickets"])
        random_matches = []
        for rep in range(RANDOM_REPS):
            seed = target["round"] * 1000003 + rep * 7919 + 17
            random_matches.append(max_match(target["nums"], random_bundle(seed)))
        rows.append({
            "round": target["round"],
            "orbit": " -> ".join(model["orbit"]),
            "historical_matches": model["matches"],
            "allocation": json.dumps(model["allocation"], ensure_ascii=False),
            "branch10_max_match": mm,
            "random_mean_max_match": sum(random_matches)/RANDOM_REPS,
            "random_rate_3plus": sum(x >= 3 for x in random_matches)/RANDOM_REPS,
            "random_rate_4plus": sum(x >= 4 for x in random_matches)/RANDOM_REPS,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    summary = {
        "model": "Transition Branch 10 v0 (historical-analog)",
        "definition": "Current 2-transition orbit -> historical next-branch counts -> largest-remainder allocation to 10 slots -> nearest historical origin-state ticket within each branch.",
        "boundary": "Tickets are historical analog next-draw tickets; v0 does not force each copied ticket to satisfy the branch geometry relative to the target current state.",
        "history_round_min": int(df["round"].min()),
        "history_round_max": int(df["round"].max()),
        "evaluated_rounds": n,
        "min_historical_same_orbit_matches": MIN_MATCHES,
        "random_reps_per_round": RANDOM_REPS,
        "branch10": {
            "mean_max_match": float(out["branch10_max_match"].mean()),
            "rate_3plus": float((out["branch10_max_match"] >= 3).mean()),
            "rate_4plus": float((out["branch10_max_match"] >= 4).mean()),
            "rate_5plus": float((out["branch10_max_match"] >= 5).mean()),
            "count_3plus": int((out["branch10_max_match"] >= 3).sum()),
            "count_4plus": int((out["branch10_max_match"] >= 4).sum()),
            "count_5plus": int((out["branch10_max_match"] >= 5).sum()),
        },
        "uniform_random10": {
            "mean_max_match": float(out["random_mean_max_match"].mean()),
            "rate_3plus": float(out["random_rate_3plus"].mean()),
            "rate_4plus": float(out["random_rate_4plus"].mean()),
        },
    }
    summary["difference_pt"] = {
        "rate_3plus": (summary["branch10"]["rate_3plus"] - summary["uniform_random10"]["rate_3plus"]) * 100,
        "rate_4plus": (summary["branch10"]["rate_4plus"] - summary["uniform_random10"]["rate_4plus"]) * 100,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
