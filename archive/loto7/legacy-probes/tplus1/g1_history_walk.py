"""G1: observe verified LOTO7 history as a State trajectory.

Observation only. A visible pattern may become a hypothesis, but this script assigns
no mechanism, causality, predictive power, or meaning to it.
"""

from pathlib import Path
import csv
from collections import Counter

DATA = Path("data/loto7.csv")
OUT = Path("playground/tplus1-time-imagination/output/g1_state_walk.txt")


def state(nums):
    return sum(nums) / 7.0, max(nums) - min(nums)


def direction(a, b):
    mean = "HIGH" if b[0] > a[0] else "LOW" if b[0] < a[0] else "SAME"
    width = "OPEN" if b[1] > a[1] else "CLOSE" if b[1] < a[1] else "SAME"
    return f"{mean}+{width}"


def main():
    rows = []
    with DATA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nums = [int(r[f"n{i}"]) for i in range(1, 8)]
            rows.append((int(r["round"]), nums, state(nums)))

    transitions = []
    counts = Counter()
    for prev, cur in zip(rows, rows[1:]):
        d = direction(prev[2], cur[2])
        counts[d] += 1
        transitions.append((prev, cur, d))

    lines = [
        "T+1 TIME IMAGINATION / G1 STATE WALK",
        "Observation only: visible structure is not automatically meaningful or predictive.",
        f"draws={len(rows)} transitions={len(transitions)} rounds={rows[0][0]}..{rows[-1][0]}",
        "",
        "DIRECTION COUNTS",
    ]
    for k, v in counts.most_common():
        lines.append(f"{k}: {v}")

    lines += ["", "RECENT 40 TRANSITIONS"]
    for prev, cur, d in transitions[-40:]:
        pm, pw = prev[2]
        cm, cw = cur[2]
        lines.append(
            f"{prev[0]}->{cur[0]} {d:11s} mean={pm:.2f}->{cm:.2f} width={pw}->{cw}"
        )

    # Purely descriptive: repeated direction strings. No significance test here.
    seq = [d for _, _, d in transitions]
    lines += ["", "RECENT DIRECTION SEQUENCE", " -> ".join(seq[-30:])]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
