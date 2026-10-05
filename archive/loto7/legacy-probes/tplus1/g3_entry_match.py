"""G3: find historical entries shaped like 693->694.

Current entry: LOW+OPEN arriving at width >= 33.
Observation only; no threshold optimization.
"""
from pathlib import Path
import csv
from collections import Counter

DATA = Path("data/loto7.csv")
OUT = Path("playground/tplus1-time-imagination/output/g3_entry_match.txt")
THRESHOLD = 33


def state(nums):
    return sum(nums) / 7.0, max(nums) - min(nums)


def direction(a,b):
    mean="HIGH" if b[0]>a[0] else "LOW" if b[0]<a[0] else "SAME"
    width="OPEN" if b[1]>a[1] else "CLOSE" if b[1]<a[1] else "SAME"
    return f"{mean}+{width}"


def main():
    rows=[]
    with DATA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nums=[int(r[f"n{i}"]) for i in range(1,8)]
            rows.append((int(r["round"]),state(nums)))

    cases=[]
    for prev,cur,nxt in zip(rows,rows[1:],rows[2:]):
        entry=direction(prev[1],cur[1])
        if entry=="LOW+OPEN" and cur[1][1]>=THRESHOLD:
            cases.append((prev,cur,nxt,direction(cur[1],nxt[1])))

    counts=Counter(x[3] for x in cases)
    lines=["G3 / ENTRY-SHAPE GLANCE",
           f"gate: entry LOW+OPEN and arrival width >= {THRESHOLD}",
           "current 693->694 matches this gate; 694->695 is not yet known.",
           f"historical_cases_with_next={len(cases)}","","NEXT DIRECTION COUNTS"]
    for k,v in counts.most_common(): lines.append(f"{k}: {v}")
    lines += ["","CASES"]
    for prev,cur,nxt,d in cases:
        lines.append(f"{prev[0]}->{cur[0]}->{nxt[0]} entry=LOW+OPEN next={d:11s} mean={cur[1][0]:.2f}->{nxt[1][0]:.2f} width={cur[1][1]}->{nxt[1][1]}")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__": main()
