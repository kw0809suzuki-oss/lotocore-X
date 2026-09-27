from __future__ import annotations

from pathlib import Path
from itertools import combinations
import statistics
import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
WINDOW = 100
NUMBERS = list(range(1, 38))

def truth(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))

def center(xs):
    return sum(xs) / 7.0

def span(xs):
    return max(xs) - min(xs)

def sgn(x, eps=1e-12):
    return 1 if x > eps else (-1 if x < -eps else 0)

def nearest_distance(actual, picks):
    return sum(min(abs(a-p) for p in picks) for a in actual) / 7.0

def hits(actual, picks):
    return len(set(actual) & set(picks))

def snapshot(history):
    core = tuple(sorted(lotocore.predict(history).numbers))
    snap = lotocore.score_snapshot(history)
    scores = {int(k): float(v) for k, v in snap["scores"].items()}
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    ranked = sorted(NUMBERS, key=lambda n: (ranks[n], n))
    noncore = tuple(n for n in ranked if n not in set(core))
    return {
        "core": core, "scores": scores, "noncore": noncore,
        "center": center(core), "span": span(core),
    }

def choose(core, noncore, scores, boundary_k, max_swaps, target_center):
    support = tuple(sorted(set(core) | set(noncore[:boundary_k])))
    core_set = set(core)
    best = core
    best_key = None
    for comb in combinations(support, 7):
        swaps = 7 - len(core_set & set(comb))
        if max_swaps is not None and swaps > max_swaps:
            continue
        err = abs(center(comb)-target_center) / max(span(core), 1.0)
        mass = sum(scores[n] for n in comb)
        key = (err, swaps, -mass, comb)
        if best_key is None or key < best_key:
            best_key = key
            best = comb
    return tuple(best)

ARMS = []
for k in (3,6,10):
    for sw in (1,2,None):
        for mag in (0.5,1.0,1.5,2.0):
            ARMS.append((k,sw,mag))

def name(k,sw,mag):
    s = "all" if sw is None else str(sw)
    m = str(mag).replace(".","p")
    return f"k{k}_sw{s}_m{m}"

def summarize(rows, label):
    print(f"--- {label} n={len(rows)} ---")
    bh = statistics.mean(r["base_h"] for r in rows)
    bd = statistics.mean(r["base_d"] for r in rows)
    stats=[]
    for k,sw,mag in ARMS:
        n=name(k,sw,mag)
        mh=statistics.mean(r[n+"_h"] for r in rows)
        md=statistics.mean(r[n+"_d"] for r in rows)
        p3=statistics.mean(r[n+"_h"]>=3 for r in rows)
        stats.append((mh-bh, bd-md, p3, n, mh, md))
    print("TOP exact delta")
    for dh,g,p3,n,mh,md in sorted(stats, reverse=True)[:10]:
        print(f"{n:16s} hits={mh:.6f} dh={dh:+.6f} dist={md:.6f} gain={g:+.6f} 3+={p3:.6f}")
    print("TOP distance gain")
    for dh,g,p3,n,mh,md in sorted(stats, key=lambda z:z[1], reverse=True)[:10]:
        print(f"{n:16s} hits={mh:.6f} dh={dh:+.6f} dist={md:.6f} gain={g:+.6f} 3+={p3:.6f}")
    print()

def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    boxes={}
    for i in range(WINDOW,len(df)):
        r=int(df.iloc[i]["round"])
        boxes[r]=snapshot(df.iloc[i-WINDOW:i])

    rows=[]
    for i in range(WINDOW+1,len(df)):
        r=int(df.iloc[i]["round"]); pr=int(df.iloc[i-1]["round"])
        cur=boxes[r]; prev=boxes[pr]; act=truth(df.iloc[i])
        cdir=sgn(cur["center"]-prev["center"])
        rec={"round":r,"base_h":hits(act,cur["core"]),"base_d":nearest_distance(act,cur["core"])}
        for k,sw,mag in ARMS:
            target=cur["center"]-cdir*mag
            picks=choose(cur["core"],cur["noncore"],cur["scores"],k,sw,target)
            n=name(k,sw,mag)
            rec[n+"_h"]=hits(act,picks)
            rec[n+"_d"]=nearest_distance(act,picks)
        rows.append(rec)

    print("=== LOTO7 REVERSE CENTER GEOMETRY PACK v2 ===")
    print(f"targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    print("k = Boundary size; sw = max number of replacements; m = reverse-center magnitude.")
    print()
    summarize(rows,"ALL")
    summarize(rows[-200:],"RECENT200")
    summarize(rows[-100:],"RECENT100")

if __name__=="__main__":
    main()
