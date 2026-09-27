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


def box(history):
    core = tuple(sorted(lotocore.predict(history).numbers))
    snap = lotocore.score_snapshot(history)
    scores = {int(k): float(v) for k, v in snap["scores"].items()}
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    ranked = sorted(NUMBERS, key=lambda n: (ranks[n], n))
    boundary = tuple(n for n in ranked if n not in set(core))[:6]
    support = tuple(sorted(set(core) | set(boundary)))
    return {
        "core": core,
        "scores": scores,
        "support": support,
        "center": center(core),
        "span": span(core),
    }


def choose(support, scores, target_center, target_span):
    best = None
    best_key = None
    cscale = max(target_span, 1.0)
    sscale = max(target_span, 1.0)
    for comb in combinations(support, 7):
        c = center(comb)
        sp = span(comb)
        err = abs(c-target_center)/cscale + abs(sp-target_span)/sscale
        mass = sum(scores[n] for n in comb)
        key = (err, -mass, comb)
        if best_key is None or key < best_key:
            best_key = key
            best = comb
    return tuple(best)


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        r = int(df.iloc[i]["round"])
        boxes[r] = box(df.iloc[i-WINDOW:i])

    rows = []
    for i in range(WINDOW+1, len(df)):
        r = int(df.iloc[i]["round"])
        pr = int(df.iloc[i-1]["round"])
        if r not in boxes or pr not in boxes:
            continue

        cur = boxes[r]
        prev = boxes[pr]
        actual = truth(df.iloc[i])

        cdir = sgn(cur["center"] - prev["center"])
        rdir = sgn(cur["span"] - prev["span"])

        tc = cur["center"]
        ts = cur["span"]

        rules = {
            "base": cur["core"],
            "same_c1": choose(cur["support"], cur["scores"], tc + cdir*1.0, ts),
            "opp_c1": choose(cur["support"], cur["scores"], tc - cdir*1.0, ts),
            "same_c2": choose(cur["support"], cur["scores"], tc + cdir*2.0, ts),
            "same_c1r2": choose(cur["support"], cur["scores"], tc + cdir*1.0, max(1.0, ts + rdir*2.0)),
            "opp_c1r2": choose(cur["support"], cur["scores"], tc - cdir*1.0, max(1.0, ts - rdir*2.0)),
        }

        rec = {"round": r, "cdir": cdir, "rdir": rdir, "actual": actual}
        for name, picks in rules.items():
            rec[name] = picks
            rec[name+"_hits"] = hits(actual, picks)
            rec[name+"_dist"] = nearest_distance(actual, picks)
        rows.append(rec)

    print("=== LOTO7 TIME NUDGE TOY v0 ===")
    print(f"targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    print("Support = current CORE7 + current Boundary6.")
    print("Time direction = Box(t-1)->Box(t), using history only before target t.")
    print()

    names = ["base","same_c1","opp_c1","same_c2","same_c1r2","opp_c1r2"]
    base_h = statistics.mean(r["base_hits"] for r in rows)
    base_d = statistics.mean(r["base_dist"] for r in rows)
    for name in names:
        mh = statistics.mean(r[name+"_hits"] for r in rows)
        md = statistics.mean(r[name+"_dist"] for r in rows)
        p3 = statistics.mean(1.0 if r[name+"_hits"] >= 3 else 0.0 for r in rows)
        better = statistics.mean(1.0 if r[name+"_dist"] < r["base_dist"] else 0.0 for r in rows)
        worse = statistics.mean(1.0 if r[name+"_dist"] > r["base_dist"] else 0.0 for r in rows)
        print(f"{name:11s} hits={mh:.6f} delta_hits={mh-base_h:+.6f} dist={md:.6f} gain={base_d-md:+.6f} 3+={p3:.6f} better={better:.6f} worse={worse:.6f}")

    print()
    print("RECENT 15")
    for r in rows[-15:]:
        print(
            f"r{r['round']} cdir={r['cdir']:+d} rdir={r['rdir']:+d} actual={'-'.join(f'{x:02d}' for x in r['actual'])} | "
            + " | ".join(
                f"{n}:{'-'.join(f'{x:02d}' for x in r[n])} h{r[n+'_hits']} d{r[n+'_dist']:.2f}"
                for n in names
            )
        )


if __name__ == "__main__":
    main()
