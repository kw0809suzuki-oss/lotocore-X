from __future__ import annotations

from pathlib import Path
from itertools import combinations
import statistics
import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
WINDOW = 100
NUMBERS = list(range(1, 38))
MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

def truth(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))

def center(xs):
    return sum(xs) / 7.0

def span(xs):
    return max(xs) - min(xs)

def sgn(x, eps=1e-12):
    return 1 if x > eps else (-1 if x < -eps else 0)

def hits(actual, picks):
    return len(set(actual) & set(picks))

def dist(actual, picks):
    return sum(min(abs(a-p) for p in picks) for a in actual) / 7.0

def snap(history):
    core = tuple(sorted(lotocore.predict(history).numbers))
    ss = lotocore.score_snapshot(history)
    scores = {int(k): float(v) for k, v in ss["scores"].items()}
    ranks = {int(k): int(v) for k, v in ss["ranks"].items()}
    ranked = sorted(NUMBERS, key=lambda n: (ranks[n], n))
    boundary = tuple(n for n in ranked if n not in set(core))[:6]
    support = tuple(sorted(set(core) | set(boundary)))
    return core, scores, support, center(core), span(core)

def candidates(support, scores):
    out = []
    for comb in combinations(support, 7):
        out.append((comb, center(comb), sum(scores[n] for n in comb)))
    return out

def pick(cands, target_center, scale):
    best = None
    best_key = None
    for comb, c, mass in cands:
        key = (abs(c-target_center)/scale, -mass, comb)
        if best_key is None or key < best_key:
            best_key = key
            best = comb
    return tuple(best)

def band(abs_delta):
    if abs_delta < 1e-12:
        return "zero"
    if abs_delta < 0.75:
        return "small"
    if abs_delta <= 2.0:
        return "medium"
    return "large"

def summarize_group(rows, label):
    if not rows:
        return
    print(f"--- {label} n={len(rows)} ---")
    bh = statistics.mean(r["base_h"] for r in rows)
    bd = statistics.mean(r["base_d"] for r in rows)
    good = []
    for m in MAGS:
        k = f"m{str(m).replace('.','p')}"
        mh = statistics.mean(r[k+"_h"] for r in rows)
        md = statistics.mean(r[k+"_d"] for r in rows)
        dh = mh - bh
        gd = bd - md
        if dh >= 0 and gd >= 0:
            good.append(m)
        print(f"reverse {m:>3.1f}: hits={mh:.6f} dh={dh:+.6f} dist={md:.6f} gain={gd:+.6f}")
    print("both-better-or-equal band:", good if good else "none")
    print()

def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        r = int(df.iloc[i]["round"])
        boxes[r] = snap(df.iloc[i-WINDOW:i])

    rows = []
    for i in range(WINDOW+1, len(df)):
        r = int(df.iloc[i]["round"])
        pr = int(df.iloc[i-1]["round"])
        core, scores, support, c, sp = boxes[r]
        _, _, _, pc, _ = boxes[pr]
        delta = c - pc
        direction = sgn(delta)
        actual = truth(df.iloc[i])

        cands = candidates(support, scores)
        rec = {
            "round": r,
            "delta": delta,
            "abs_delta": abs(delta),
            "band": band(abs(delta)),
            "base_h": hits(actual, core),
            "base_d": dist(actual, core),
        }
        for m in MAGS:
            picks = core if m == 0.0 or direction == 0 else pick(cands, c - direction*m, max(sp,1.0))
            k = f"m{str(m).replace('.','p')}"
            rec[k+"_h"] = hits(actual, picks)
            rec[k+"_d"] = dist(actual, picks)
            rec[k+"_picks"] = picks
        rows.append(rec)

    print("=== LOTO7 DYNAMIC REVERSE BAND TOY v0 ===")
    print(f"targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    print("One draw is added at each step; Box(t) uses only history before target t.")
    print("Bands use absolute Box-center move: small <0.75, medium 0.75..2.0, large >2.0.")
    print()

    summarize_group(rows, "ALL")
    for b in ("zero","small","medium","large"):
        summarize_group([r for r in rows if r["band"] == b], b.upper())

    # Three deliberately simple dynamic policies; not optimized per round.
    policies = {
        "soft": {"zero":0.0, "small":0.5, "medium":1.0, "large":1.0},
        "elastic": {"zero":0.0, "small":0.5, "medium":1.5, "large":2.0},
        "damped_large": {"zero":0.0, "small":0.5, "medium":1.5, "large":0.5},
    }

    def policy_stats(sub, label):
        print(f"--- POLICY {label} n={len(sub)} ---")
        bh = statistics.mean(r["base_h"] for r in sub)
        bd = statistics.mean(r["base_d"] for r in sub)
        for pname, pmap in policies.items():
            hs=[]; ds=[]
            for r in sub:
                m=pmap[r["band"]]
                k=f"m{str(m).replace('.','p')}"
                hs.append(r[k+"_h"]); ds.append(r[k+"_d"])
            mh=statistics.mean(hs); md=statistics.mean(ds)
            print(f"{pname:14s} hits={mh:.6f} dh={mh-bh:+.6f} dist={md:.6f} gain={bd-md:+.6f}")
        print()

    policy_stats(rows, "ALL")
    policy_stats(rows[-200:], "RECENT200")
    policy_stats(rows[-100:], "RECENT100")

if __name__ == "__main__":
    main()
