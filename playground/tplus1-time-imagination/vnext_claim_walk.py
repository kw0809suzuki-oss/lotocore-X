"""T+1 vNext — Claim / Boundary walk-forward observation.

Independent playground only. Does not modify CORE/X/Random.
The purpose is to separate what is claimed from what remains Possible,
then score the pre-fixed claim on unseen T+1 as HIT/MISS/UNKNOWN.

G1 rule is intentionally minimal and parameter-free:
- State_T includes the movement received from T-1 -> T.
- Claim only the width component: the next width movement repeats the
  direction just received (OPEN/CLOSE/SAME).
- Mean is left Possible (unclaimed).
- Boundary is exact and fixed before T+1 is opened:
    OPEN  => width_(T+1) > width_T
    CLOSE => width_(T+1) < width_T
    SAME  => width_(T+1) == width_T
No tolerance is fitted from outcomes.
"""
from pathlib import Path
import csv
from collections import Counter

DATA = Path("data/loto7.csv")
OUT = Path("playground/tplus1-time-imagination/output/vnext_claim_walk.csv")


def state(nums):
    return {
        "mean": sum(nums) / 7.0,
        "width": max(nums) - min(nums),
    }


def movement(prev, cur):
    mean = "HIGH" if cur["mean"] > prev["mean"] else "LOW" if cur["mean"] < prev["mean"] else "SAME"
    width = "OPEN" if cur["width"] > prev["width"] else "CLOSE" if cur["width"] < prev["width"] else "SAME"
    return mean, width


def evaluate_width_claim(claim, current_width, next_width):
    if claim == "OPEN":
        return "HIT" if next_width > current_width else "MISS"
    if claim == "CLOSE":
        return "HIT" if next_width < current_width else "MISS"
    if claim == "SAME":
        return "HIT" if next_width == current_width else "MISS"
    return "UNKNOWN"


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
    # At T, only T-1 and T are used to create the claim. T+1 is opened afterward.
    for prev, cur, nxt in zip(rows, rows[1:], rows[2:]):
        mean_in, width_in = movement(prev[1], cur[1])
        claim = width_in
        verdict = evaluate_width_claim(claim, cur[1]["width"], nxt[1]["width"])
        _, actual_width_move = movement(cur[1], nxt[1])
        out.append({
            "t_minus_1": prev[0],
            "t": cur[0],
            "t_plus_1": nxt[0],
            "state_mean": round(cur[1]["mean"], 6),
            "state_width": cur[1]["width"],
            "received_mean_move": mean_in,
            "received_width_move": width_in,
            "claim_width_t_plus_1": claim,
            "boundary": f"next_width {'>' if claim == 'OPEN' else '<' if claim == 'CLOSE' else '=='} {cur[1]['width']}",
            "possible_mean": "UNCLAIMED",
            "actual_width_t_plus_1": nxt[1]["width"],
            "actual_width_move": actual_width_move,
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
    accuracy = c["HIT"] / decided if decided else 0.0
    coverage = decided / len(rows) if rows else 0.0
    print("T+1 vNext / CLAIM-BOUNDARY WALK")
    print("claim=repeat received WIDTH direction; mean remains Possible")
    print("boundary=exact sign only; no fitted tolerance")
    print(f"n={len(rows)} HIT={c['HIT']} MISS={c['MISS']} UNKNOWN={c['UNKNOWN']}")
    print(f"coverage={coverage:.4f} accuracy={accuracy:.4f} total_hit={(c['HIT']/len(rows) if rows else 0):.4f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
