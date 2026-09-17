"""T+1 vNext G2 — received Flow + arrival State -> Claim.

Independent observation branch. No CORE/X/Random changes.

Purpose
-------
Test the next structural step after the naive "repeat received width direction"
claim failed.  The claim is generated from the *arrival state*, not direction
alone.

To avoid using the current edge (round 694, width=33) as a fitted predictive
threshold, this walk uses an expanding-history boundary at every T:

    wide_boundary_T = empirical 75th percentile of widths observed through T-1

The boundary therefore exists before T+1 is opened and changes only from past
information.  When State_T arrives at/above that boundary, Claim_T+1 is CLOSE.
Otherwise no claim is made (UNKNOWN).

This is deliberately a one-component claim. Mean and exact next width remain
Possible/unclaimed. UNKNOWN is therefore "no claim issued", not a retroactive
escape from a wrong claim.
"""
from pathlib import Path
import csv
import math
from collections import Counter

DATA = Path("data/loto7.csv")
OUT = Path("playground/tplus1-time-imagination/output/vnext_arrival_state_walk.csv")
MIN_HISTORY = 20
QUANTILE = 0.75


def state(nums):
    return {"mean": sum(nums) / 7.0, "width": max(nums) - min(nums)}


def movement(prev, cur):
    mean = "HIGH" if cur["mean"] > prev["mean"] else "LOW" if cur["mean"] < prev["mean"] else "SAME"
    width = "OPEN" if cur["width"] > prev["width"] else "CLOSE" if cur["width"] < prev["width"] else "SAME"
    return mean, width


def nearest_rank(values, q):
    xs = sorted(values)
    if not xs:
        raise ValueError("empty values")
    rank = max(1, math.ceil(q * len(xs)))
    return xs[rank - 1]


def load_rows():
    rows = []
    with DATA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nums = [int(r[f"n{i}"]) for i in range(1, 8)]
            rows.append((int(r["round"]), state(nums)))
    return rows


def run():
    rows = load_rows()
    out = []
    # i is T. Boundary uses only rows[:i], i.e. information ending at T-1.
    for i in range(1, len(rows) - 1):
        prev, cur, nxt = rows[i - 1], rows[i], rows[i + 1]
        if i < MIN_HISTORY:
            continue
        hist_widths = [x[1]["width"] for x in rows[:i]]
        boundary = nearest_rank(hist_widths, QUANTILE)
        received_mean, received_width = movement(prev[1], cur[1])
        actual_mean, actual_width = movement(cur[1], nxt[1])

        claim = "CLOSE" if cur[1]["width"] >= boundary else "NO_CLAIM"
        if claim == "CLOSE":
            verdict = "HIT" if nxt[1]["width"] < cur[1]["width"] else "MISS"
        else:
            verdict = "UNKNOWN"

        out.append({
            "t_minus_1": prev[0],
            "t": cur[0],
            "t_plus_1": nxt[0],
            "received_mean_move": received_mean,
            "received_width_move": received_width,
            "arrival_mean": round(cur[1]["mean"], 6),
            "arrival_width": cur[1]["width"],
            "wide_boundary_t": boundary,
            "claim_width_t_plus_1": claim,
            "boundary_rule": f"arrival_width >= q75_past({boundary}) => CLOSE",
            "possible_mean": "UNCLAIMED",
            "possible_exact_width": "UNCLAIMED",
            "actual_width_t_plus_1": nxt[1]["width"],
            "actual_width_move": actual_width,
            "actual_mean_move": actual_mean,
            "verdict": verdict,
        })
    return out


def main():
    rows = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    c = Counter(r["verdict"] for r in rows)
    decided = c["HIT"] + c["MISS"]
    n = len(rows)
    print("T+1 vNext / ARRIVAL-STATE CLAIM WALK")
    print("state=received Flow + arrival width; claim=CLOSE only at expanding-history q75 wide boundary")
    print("mean/exact width remain Possible; no target-fitted tolerance")
    print(f"n={n} CLAIMS={decided} HIT={c['HIT']} MISS={c['MISS']} UNKNOWN={c['UNKNOWN']}")
    print(f"coverage={(decided/n if n else 0):.4f} accuracy={(c['HIT']/decided if decided else 0):.4f} total_hit={(c['HIT']/n if n else 0):.4f}")
    by_entry = Counter()
    for r in rows:
        if r["verdict"] != "UNKNOWN":
            by_entry[(r["received_width_move"], r["verdict"])] += 1
    print("claimed cases by received width:")
    for entry in ("OPEN", "CLOSE", "SAME"):
        h = by_entry[(entry, "HIT")]
        m = by_entry[(entry, "MISS")]
        if h + m:
            print(f"  {entry}: HIT={h} MISS={m} accuracy={h/(h+m):.4f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
