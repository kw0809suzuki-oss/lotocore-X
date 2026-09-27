from __future__ import annotations

from math import comb
from pathlib import Path
import statistics

import pandas as pd

import probe_loto7_dynamic_band_online_replay_v0 as box

DATA = Path("data/loto7.csv")
WINDOW = box.WINDOW
MAGS = box.MAGS
MIN_HISTORY = box.MIN_HISTORY

# Same expansion rule as the first Floot A-BOX implementation:
# ticket 1 = frozen Final 7
# tickets 2..10 = Final 7 x state-transition blend
SPECS = (
    (6, 0.5),
    (6, 1.0),
    (5, 0.5),
    (5, 1.0),
    (5, 1.5),
    (4, 1.0),
    (4, 1.5),
    (4, 2.0),
    (4, 2.5),
)


def max_hits(actual, tickets):
    return max(box.hits(actual, t) for t in tickets)


def blend10(cur, final7, delta):
    d = box.sgn(delta)
    final_set = set(final7)
    used = {tuple(final7)}
    out = [tuple(final7)]
    cands = box.precompute(cur)
    scale = max(cur["span"], 1.0)

    for keep, mag in SPECS:
        target_center = cur["center"] - d * mag
        ranked = []
        for nums, c, sp, mass in cands:
            if len(set(nums) & final_set) != keep:
                continue
            err = abs(c - target_center) / scale + abs(sp - cur["span"]) / scale
            ranked.append((err, -mass, tuple(nums)))
        ranked.sort()
        for _, _, nums in ranked:
            if nums not in used:
                used.add(nums)
                out.append(nums)
                break

    if len(out) < 10:
        fallback = []
        for nums, c, sp, mass in cands:
            nums = tuple(nums)
            overlap = len(set(nums) & final_set)
            if 4 <= overlap <= 6 and nums not in used:
                fallback.append((-mass, nums))
        fallback.sort()
        for _, nums in fallback:
            if len(out) >= 10:
                break
            used.add(nums)
            out.append(nums)

    return out[:10]


def random_single_exact(k):
    return comb(7, k) * comb(30, 7 - k) / comb(37, 7)


def random_single_at_least(k):
    return sum(random_single_exact(j) for j in range(k, 8))


def random_pack_at_least(k, tickets):
    p = random_single_at_least(k)
    return 1.0 - (1.0 - p) ** tickets


def random_pack_expected_max(tickets):
    # E[max] = sum P(max >= k), k=1..7
    return sum(random_pack_at_least(k, tickets) for k in range(1, 8))


def anchored_random_at_least(final_hits, threshold):
    if final_hits >= threshold:
        return 1.0
    return random_pack_at_least(threshold, 9)


def anchored_random_expected_max(final_hits):
    # Final 7 is fixed at final_hits. Add 9 independent uniform-random tickets.
    return final_hits + sum(
        random_pack_at_least(k, 9)
        for k in range(final_hits + 1, 8)
    )


def summarize(rows, label):
    print(f"--- {label} n={len(rows)} ---")
    if not rows:
        print()
        return

    fh = statistics.mean(r["final_h"] for r in rows)
    ah = statistics.mean(r["abox10_h"] for r in rows)
    gain = statistics.mean(r["abox10_h"] - r["final_h"] for r in rows)
    improve = statistics.mean(r["abox10_h"] > r["final_h"] for r in rows)

    anchor_mean = statistics.mean(
        anchored_random_expected_max(r["final_h"]) for r in rows
    )

    print(f"Final7 mean hits={fh:.6f}")
    print(f"A-BOX10 mean max hits={ah:.6f}")
    print(f"A-BOX10 mean gain over Final7={gain:+.6f}")
    print(f"A-BOX10 improves over Final7 in {improve:.6%} of rounds")
    print(f"CONTROL Final7+9 random expected max={anchor_mean:.6f}")
    print(f"delta A-BOX10 vs anchored-random={ah-anchor_mean:+.6f}")

    for k in (3, 4, 5, 6):
        ar = statistics.mean(r["abox10_h"] >= k for r in rows)
        cr = statistics.mean(
            anchored_random_at_least(r["final_h"], k) for r in rows
        )
        print(
            f"{k}+ max: A-BOX10={ar:.6%} "
            f"Final7+9random={cr:.6%} delta={(ar-cr)*100:+.3f}pt"
        )
    print()


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        boxes[int(df.iloc[i]["round"])] = box.snap(df.iloc[i-WINDOW:i])

    stats = box.empty_stats()
    rows = []

    for i in range(WINDOW + 1, len(df)):
        r = int(df.iloc[i]["round"])
        pr = int(df.iloc[i - 1]["round"])
        cur, prev = boxes[r], boxes[pr]
        actual = box.truth(df.iloc[i])

        delta, sets = box.candidate_sets(cur, prev)
        b = box.band(abs(delta))

        if b == "zero":
            chosen_mag = 0.0
            mature = True
        else:
            alive, chosen_mag, _ = box.live_band(stats[b])
            mature = stats[b]["n"] >= MIN_HISTORY
            if not alive:
                chosen_mag = 0.0

        final7 = sets[chosen_mag]
        tickets = blend10(cur, final7, delta)

        rec = {
            "round": r,
            "band": b,
            "mature": mature,
            "final_h": box.hits(actual, final7),
            "abox10_h": max_hits(actual, tickets),
        }
        rows.append(rec)

        # Strict online order: update only after target r was scored.
        if b != "zero":
            base = sets[0.0]
            stats[b]["n"] += 1
            stats[b]["base_h"] += box.hits(actual, base)
            stats[b]["base_d"] += box.dist(actual, base)
            for m in MAGS:
                stats[b]["arms"][m]["h"] += box.hits(actual, sets[m])
                stats[b]["arms"][m]["d"] += box.dist(actual, sets[m])

    mature = [r for r in rows if r["mature"] and r["band"] != "zero"]

    print("=== LOTO7 FINAL7 -> A-BOX10 AMPLIFICATION v0 ===")
    print("Question: when Final7 is strong, do the nine transition-blend tickets amplify it?")
    print("Control: the SAME Final7 plus 9 uniform-random tickets.")
    print("No target-aware tuning. Frozen Final7 logic. Strict online chronology.")
    print()
    print(
        f"Reference pure 10-random: expected max={random_pack_expected_max(10):.6f} "
        f"3+={random_pack_at_least(3,10):.6%} "
        f"4+={random_pack_at_least(4,10):.6%} "
        f"5+={random_pack_at_least(5,10):.6%}"
    )
    print()

    summarize(rows, "ALL")
    summarize(mature, "MATURE")
    summarize(mature[-200:], "LAST200 MATURE")
    summarize(mature[-100:], "LAST100 MATURE")

    print("=== CONDITIONAL ON FINAL7 HITS | MATURE ===")
    for k in range(4):
        summarize([r for r in mature if r["final_h"] == k], f"Final7={k}")
    summarize([r for r in mature if r["final_h"] >= 4], "Final7=4+")

    print("=== TRANSITION AMPLIFICATION COUNTS | MATURE ===")
    for k in range(5):
        sub = [r for r in mature if (r["final_h"] == k if k < 4 else r["final_h"] >= 4)]
        if not sub:
            continue
        improved = sum(r["abox10_h"] > r["final_h"] for r in sub)
        same = sum(r["abox10_h"] == r["final_h"] for r in sub)
        print(
            f"Final7 {'4+' if k == 4 else k}: n={len(sub)} "
            f"improved={improved} ({improved/len(sub):.2%}) "
            f"same={same} ({same/len(sub):.2%})"
        )


if __name__ == "__main__":
    main()
