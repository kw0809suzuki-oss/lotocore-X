from __future__ import annotations

from math import comb
import statistics
import pandas as pd

import probe_loto7_dynamic_band_online_replay_v0 as box

DATA = box.DATA
WINDOW = box.WINDOW
MAGS = box.MAGS
MIN_HISTORY = box.MIN_HISTORY

def random_prob_exact(k: int) -> float:
    return comb(7,k) * comb(30, 7-k) / comb(37,7)

def random_prob_at_least(k: int) -> float:
    return sum(random_prob_exact(j) for j in range(k,8))

def rate(rows, key, k):
    return sum(r[key] >= k for r in rows) / len(rows)

def mean(rows,key):
    return statistics.mean(r[key] for r in rows)

def uplift(model_rate, base_rate):
    return (model_rate/base_rate - 1.0) if base_rate > 0 else float("nan")

def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        r = int(df.iloc[i]["round"])
        boxes[r] = box.snap(df.iloc[i-WINDOW:i])

    stats = box.empty_stats()
    rows = []

    for i in range(WINDOW+1, len(df)):
        r = int(df.iloc[i]["round"])
        pr = int(df.iloc[i-1]["round"])
        cur, prev = boxes[r], boxes[pr]
        actual = box.truth(df.iloc[i])

        delta, sets = box.candidate_sets(cur, prev)
        b = box.band(abs(delta))

        base_pick = sets[0.0]

        if b == "zero":
            alive, chosen_mag, width = [0.0], 0.0, 0.0
            mature = True
        else:
            alive, chosen_mag, width = box.live_band(stats[b])
            mature = stats[b]["n"] >= MIN_HISTORY
            if not alive:
                chosen_mag = 0.0

        model_pick = sets[chosen_mag]

        rec = {
            "round":r,
            "base_hits":box.hits(actual, base_pick),
            "model_hits":box.hits(actual, model_pick),
            "mature":mature,
            "band":b,
        }
        rows.append(rec)

        # Preserve original online update order exactly: update only after prediction is scored.
        if b != "zero":
            stats[b]["n"] += 1
            stats[b]["base_h"] += rec["base_hits"]
            stats[b]["base_d"] += box.dist(actual, base_pick)
            for m in MAGS:
                stats[b]["arms"][m]["h"] += box.hits(actual, sets[m])
                stats[b]["arms"][m]["d"] += box.dist(actual, sets[m])

    mature = [r for r in rows if r["mature"] and r["band"] != "zero"]

    print("=== LOTO7 FROZEN BOX WIN-RATE EVALUATION v0 ===")
    print("Model under test: Dynamic Band Online Replay v0, unchanged.")
    print("No parameter search. No re-selection after seeing target. One 7-number output per round.")
    print(f"all targets={len(rows)} rounds={rows[0]['round']}..{rows[-1]['round']}")
    print(f"mature targets={len(mature)}")
    print()

    for label, sub in [("ALL",rows),("MATURE",mature),("LAST200_MATURE",mature[-200:]),("LAST100_MATURE",mature[-100:])]:
        print(f"--- {label} n={len(sub)} ---")
        model_mean = mean(sub,"model_hits")
        base_mean = mean(sub,"base_hits")
        random_mean = 49/37
        print(f"mean main-number hits: BOX={model_mean:.6f} CORE={base_mean:.6f} RANDOM={random_mean:.6f}")
        print(f"BOX uplift vs CORE={(model_mean/base_mean-1)*100:+.3f}%")
        print(f"BOX uplift vs RANDOM={(model_mean/random_mean-1)*100:+.3f}%")
        for k in (3,4,5,6,7):
            mr=rate(sub,"model_hits",k)
            br=rate(sub,"base_hits",k)
            rr=random_prob_at_least(k)
            print(
                f"{k}+ main hits: BOX={mr:.6%} CORE={br:.6%} RANDOM={rr:.6%} "
                f"| uplift_vs_CORE={uplift(mr,br)*100:+.2f}% "
                f"uplift_vs_RANDOM={uplift(mr,rr)*100:+.2f}%"
            )
        print()

    print("--- EXACT HIT COUNTS, MATURE ---")
    for key,label in [("model_hits","BOX"),("base_hits","CORE")]:
        counts={k:sum(r[key]==k for r in mature) for k in range(8)}
        print(label, counts)

    print()
    print("--- RANDOM THEORETICAL EXACT HIT PROBABILITIES ---")
    for k in range(8):
        print(f"{k}: {random_prob_exact(k):.8%}")

if __name__ == "__main__":
    main()
