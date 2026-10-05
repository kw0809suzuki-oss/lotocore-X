"""G2: glance at what follows a wide-open LOTO7 state.

Same phenomenon, different viewing coordinate. Descriptive only.
Threshold is deliberately anchored to current round 694 width=33, not optimized.
"""
from pathlib import Path
import csv
from collections import Counter

DATA = Path("data/loto7.csv")
OUT = Path("playground/tplus1-time-imagination/output/g2_open_afterglance.txt")
THRESHOLD = 33


def state(nums):
    return sum(nums) / 7.0, max(nums) - min(nums)


def direction(a, b):
    mean = "HIGH" if b[0] > a[0] else "LOW" if b[0] < a[0] else "SAME"
    width = "OPEN" if b[1] > a[1] else "CLOSE" if b[1] < a[1] else "SAME"
    return f"{mean}+{width}"


def main():
    rows=[]
    with DATA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nums=[int(r[f"n{i}"]) for i in range(1,8)]
            rows.append((int(r["round"]), state(nums)))

    cases=[]
    for cur,nxt in zip(rows,rows[1:]):
        if cur[1][1] >= THRESHOLD:
            d=direction(cur[1],nxt[1])
            cases.append((cur,nxt,d))

    counts=Counter(d for _,_,d in cases)
    close=sum(v for k,v in counts.items() if k.endswith("CLOSE"))
    open_=sum(v for k,v in counts.items() if k.endswith("OPEN"))
    same=sum(v for k,v in counts.items() if k.endswith("SAME"))

    lines=[
        "G2 / WIDE-OPEN AFTERGLANCE",
        f"coordinate: current width >= {THRESHOLD} (694 has width 33)",
        "Descriptive only; this threshold was not searched for predictive performance.",
        f"cases_with_next={len(cases)}",
        f"next width: CLOSE={close} OPEN={open_} SAME={same}",
        "",
        "DIRECTION COUNTS",
    ]
    for k,v in counts.most_common(): lines.append(f"{k}: {v}")
    lines += ["", "CASES"]
    for cur,nxt,d in cases:
        lines.append(f"{cur[0]}->{nxt[0]} {d:11s} mean={cur[1][0]:.2f}->{nxt[1][0]:.2f} width={cur[1][1]}->{nxt[1][1]}")
    lines += ["", "CURRENT EDGE", "694 width=33; its next state is not yet in this dataset."]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__": main()
