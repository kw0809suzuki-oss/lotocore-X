from __future__ import annotations

from pathlib import Path
from itertools import combinations
import statistics
import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
WINDOW = 100
NUMBERS = list(range(1, 38))
MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
MIN_HISTORY = 20

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

def band(abs_delta):
    if abs_delta < 1e-12:
        return "zero"
    if abs_delta < 0.75:
        return "small"
    if abs_delta <= 2.0:
        return "medium"
    return "large"

def snap(history):
    core = tuple(sorted(lotocore.predict(history).numbers))
    ss = lotocore.score_snapshot(history)
    scores = {int(k): float(v) for k, v in ss["scores"].items()}
    ranks = {int(k): int(v) for k, v in ss["ranks"].items()}
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

def precompute(box):
    out = []
    for comb in combinations(box["support"], 7):
        out.append((comb, center(comb), span(comb), sum(box["scores"][n] for n in comb)))
    return out

def pick(cands, target_center, target_span):
    scale = max(target_span, 1.0)
    best, best_key = None, None
    for comb, c, sp, mass in cands:
        err = abs(c-target_center)/scale + abs(sp-target_span)/scale
        key = (err, -mass, comb)
        if best_key is None or key < best_key:
            best_key = key
            best = comb
    return tuple(best)

def candidate_sets(cur, prev):
    delta = cur["center"] - prev["center"]
    d = sgn(delta)
    cands = precompute(cur)
    out = {}
    for m in MAGS:
        if d == 0 or m == 0.0:
            out[m] = cur["core"]
        else:
            out[m] = pick(cands, cur["center"] - d*m, cur["span"])
    return delta, out

def empty_stats():
    return {b: {"n":0, "base_h":0.0, "base_d":0.0,
                "arms":{m:{"h":0.0,"d":0.0} for m in MAGS}}
            for b in ("small","medium","large")}

def live_band(stats_for_band):
    n = stats_for_band["n"]
    if n < MIN_HISTORY:
        return [], 0.0, 0.0
    base_h = stats_for_band["base_h"] / n
    base_d = stats_for_band["base_d"] / n
    alive = []
    for m in MAGS:
        mh = stats_for_band["arms"][m]["h"] / n
        md = stats_for_band["arms"][m]["d"] / n
        if mh >= base_h and md <= base_d:
            alive.append(m)
    if not alive:
        return [], 0.0, 0.0
    lo, hi = min(alive), max(alive)
    center_band = (lo + hi) / 2.0
    chosen = min(MAGS, key=lambda m: (abs(m-center_band), m))
    return alive, chosen, hi-lo

def summarize(rows, label):
    if not rows:
        return
    print(f"--- {label} n={len(rows)} ---")
    for key in ("base","fixed1","online"):
        mh = statistics.mean(r[key+"_h"] for r in rows)
        md = statistics.mean(r[key+"_d"] for r in rows)
        p3 = statistics.mean(r[key+"_h"] >= 3 for r in rows)
        p4 = statistics.mean(r[key+"_h"] >= 4 for r in rows)
        print(f"{key:8s} hits={mh:.6f} dist={md:.6f} 3+={p3:.6f} 4+={p4:.6f}")
    bh = statistics.mean(r["base_h"] for r in rows)
    bd = statistics.mean(r["base_d"] for r in rows)
    oh = statistics.mean(r["online_h"] for r in rows)
    od = statistics.mean(r["online_d"] for r in rows)
    print(f"ONLINE vs BASE: dh={oh-bh:+.6f} distance_gain={bd-od:+.6f}")
    print()

def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        boxes[int(df.iloc[i]["round"])] = snap(df.iloc[i-WINDOW:i])

    stats = empty_stats()
    rows = []

    for i in range(WINDOW+1, len(df)):
        r = int(df.iloc[i]["round"])
        pr = int(df.iloc[i-1]["round"])
        cur, prev = boxes[r], boxes[pr]
        actual = truth(df.iloc[i])

        delta, sets = candidate_sets(cur, prev)
        b = band(abs(delta))

        base = sets[0.0]
        fixed1 = sets[1.0]

        if b == "zero":
            alive, chosen_mag, width = [0.0], 0.0, 0.0
            mature = True
        else:
            alive, chosen_mag, width = live_band(stats[b])
            mature = stats[b]["n"] >= MIN_HISTORY
            if not alive:
                chosen_mag = 0.0

        online = sets[chosen_mag]

        rec = {
            "round": r,
            "band": b,
            "delta": delta,
            "history_n": 0 if b=="zero" else stats[b]["n"],
            "mature": mature,
            "alive": tuple(alive),
            "chosen_mag": chosen_mag,
            "band_width": width,
            "base_h": hits(actual, base),
            "base_d": dist(actual, base),
            "fixed1_h": hits(actual, fixed1),
            "fixed1_d": dist(actual, fixed1),
            "online_h": hits(actual, online),
            "online_d": dist(actual, online),
        }
        rows.append(rec)

        # Only AFTER scoring target r do we update the state for r+1.
        if b != "zero":
            stats[b]["n"] += 1
            stats[b]["base_h"] += rec["base_h"]
            stats[b]["base_d"] += rec["base_d"]
            for m in MAGS:
                stats[b]["arms"][m]["h"] += hits(actual, sets[m])
                stats[b]["arms"][m]["d"] += dist(actual, sets[m])

    mature_rows = [r for r in rows if r["mature"] and r["band"] != "zero"]

    print("=== LOTO7 DYNAMIC BAND ONLINE REPLAY v0 ===")
    print(f"targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    print("STRICT ROUND ORDER: target r uses only outcomes from rounds < r to update its live band.")
    print(f"candidate corrections={MAGS}; min same-band history={MIN_HISTORY}")
    print("If no live correction exists yet, online output falls back to Base.")
    print()

    summarize(rows, "ALL INCLUDING BOOTSTRAP")
    summarize(mature_rows, "MATURE ONLINE STATE")
    summarize(mature_rows[-200:], "LAST 200 MATURE")
    summarize(mature_rows[-100:], "LAST 100 MATURE")

    print("--- CHOSEN MAGNITUDES, MATURE ---")
    for b in ("small","medium","large"):
        sub=[r for r in mature_rows if r["band"]==b]
        if not sub: continue
        counts={m:sum(r["chosen_mag"]==m for r in sub) for m in MAGS}
        avg_width=statistics.mean(r["band_width"] for r in sub)
        print(f"{b:6s} n={len(sub)} counts={counts} avg_live_width={avg_width:.3f}")
    print()

    print("RECENT 20 MATURE")
    for r in mature_rows[-20:]:
        print(
            f"r{r['round']} {r['band']} dc={r['delta']:+.3f} hist={r['history_n']} "
            f"alive={r['alive']} choose={r['chosen_mag']:.1f} | "
            f"base h{r['base_h']} d{r['base_d']:.2f} | "
            f"online h{r['online_h']} d{r['online_d']:.2f}"
        )

if __name__ == "__main__":
    main()
