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

def eval_rows(rows, label):
    print(f"--- {label} n={len(rows)} ---")
    base_h = statistics.mean(r["base_hits"] for r in rows)
    base_d = statistics.mean(r["base_dist"] for r in rows)
    for name in RULES:
        mh = statistics.mean(r[name+"_hits"] for r in rows)
        md = statistics.mean(r[name+"_dist"] for r in rows)
        p3 = statistics.mean(1.0 if r[name+"_hits"] >= 3 else 0.0 for r in rows)
        p4 = statistics.mean(1.0 if r[name+"_hits"] >= 4 else 0.0 for r in rows)
        better = statistics.mean(1.0 if r[name+"_dist"] < r["base_dist"] else 0.0 for r in rows)
        worse = statistics.mean(1.0 if r[name+"_dist"] > r["base_dist"] else 0.0 for r in rows)
        print(
            f"{name:16s} hits={mh:.6f} dh={mh-base_h:+.6f} "
            f"dist={md:.6f} gain={base_d-md:+.6f} "
            f"3+={p3:.6f} 4+={p4:.6f} better={better:.6f} worse={worse:.6f}"
        )
    print()

RULES = [
    "base",
    "same_c1",
    "opp_c05",
    "opp_c1",
    "opp_c15",
    "opp_c2",
    "opp_c3",
    "opp_actual_delta",
    "opp_big_only",
    "opp_turn2_only",
    "opp_streak_only",
    "opp_flip_only",
]

def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        r = int(df.iloc[i]["round"])
        boxes[r] = box(df.iloc[i-WINDOW:i])

    rows = []
    prev_cdir = None

    for i in range(WINDOW+1, len(df)):
        r = int(df.iloc[i]["round"])
        pr = int(df.iloc[i-1]["round"])
        if r not in boxes or pr not in boxes:
            continue

        cur = boxes[r]
        prev = boxes[pr]
        actual = truth(df.iloc[i])

        delta_c = cur["center"] - prev["center"]
        cdir = sgn(delta_c)
        turnover = 7 - len(set(prev["core"]) & set(cur["core"]))

        def shifted(amount):
            return choose(cur["support"], cur["scores"], cur["center"] + amount, cur["span"])

        rules = {}
        rules["base"] = cur["core"]
        rules["same_c1"] = shifted(+cdir * 1.0)
        rules["opp_c05"] = shifted(-cdir * 0.5)
        rules["opp_c1"] = shifted(-cdir * 1.0)
        rules["opp_c15"] = shifted(-cdir * 1.5)
        rules["opp_c2"] = shifted(-cdir * 2.0)
        rules["opp_c3"] = shifted(-cdir * 3.0)

        # Reverse the actual box center movement amount, clipped to 3.
        amt = max(-3.0, min(3.0, -delta_c))
        rules["opp_actual_delta"] = shifted(amt)

        # Conditional toys.
        rules["opp_big_only"] = shifted(-cdir * 1.0) if abs(delta_c) >= 0.5 else cur["core"]
        rules["opp_turn2_only"] = shifted(-cdir * 1.0) if turnover >= 2 else cur["core"]

        same_streak = prev_cdir is not None and cdir != 0 and cdir == prev_cdir
        flip = prev_cdir is not None and cdir != 0 and prev_cdir != 0 and cdir == -prev_cdir
        rules["opp_streak_only"] = shifted(-cdir * 1.0) if same_streak else cur["core"]
        rules["opp_flip_only"] = shifted(-cdir * 1.0) if flip else cur["core"]

        rec = {
            "round": r,
            "cdir": cdir,
            "delta_c": delta_c,
            "turnover": turnover,
            "actual": actual,
        }
        for name, picks in rules.items():
            rec[name] = picks
            rec[name+"_hits"] = hits(actual, picks)
            rec[name+"_dist"] = nearest_distance(actual, picks)
        rows.append(rec)
        prev_cdir = cdir

    print("=== LOTO7 REVERSE CENTER PUNCH PACK v1 ===")
    print(f"targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    print("Support = current CORE7 + Boundary6; only center target is nudged.")
    print("opp_* means reverse the observed Box center motion.")
    print()
    eval_rows(rows, "ALL")
    eval_rows(rows[-200:], "RECENT200")
    eval_rows(rows[-100:], "RECENT100")

    print("RECENT 12")
    for r in rows[-12:]:
        parts = []
        for n in ["base","opp_c05","opp_c1","opp_c15","opp_c2","opp_actual_delta"]:
            parts.append(f"{n}:h{r[n+'_hits']} d{r[n+'_dist']:.2f}")
        print(
            f"r{r['round']} dc={r['delta_c']:+.3f} turn={r['turnover']} "
            + " | ".join(parts)
        )

if __name__ == "__main__":
    main()
