from __future__ import annotations

from pathlib import Path
import random
import statistics

import pandas as pd

import lotocore
import x_agent
import probe_loto7_dynamic_band_online_replay_v0 as box
from probe_loto7_final7_abox10_amplification_v0 import blend10

DATA = Path("data/loto7.csv")
WINDOW = 100
ALLOCATOR_REPS = 100


def ranked_from_snapshot(snapshot):
    ranks = {int(n): int(r) for n, r in snapshot["ranks"].items()}
    return sorted(ranks, key=lambda n: (ranks[n], n))


def blend_proxy_pool(history):
    core_ranked = ranked_from_snapshot(lotocore.score_snapshot(history))
    x_ranked = ranked_from_snapshot(
        x_agent.score_snapshot(history, competition_gate=True)
    )
    core10 = core_ranked[:10]
    x10 = x_ranked[:10]
    union = sorted(set(core10) | set(x10))
    shared = set(core10) & set(x10)
    weights = {n: (2.0 if n in shared else 1.0) for n in union}
    return core10, x10, union, shared, weights


def weighted_ticket(pool, weights, rng):
    available = list(pool)
    out = []
    while len(out) < 7 and available:
        total = sum(weights[n] for n in available)
        r = rng.random() * total
        idx = 0
        for idx, n in enumerate(available):
            r -= weights[n]
            if r <= 0:
                break
        out.append(available.pop(idx))
    return tuple(sorted(out))


def blend_proxy_pack(pool, weights, seed):
    rng = random.Random(seed)
    out = []
    seen = set()
    for _ in range(3000):
        if len(out) >= 10:
            break
        t = weighted_ticket(pool, weights, rng)
        if t not in seen:
            seen.add(t)
            out.append(t)
    if len(out) < 10:
        raise RuntimeError("could not build 10 unique Blend proxy tickets")
    return out


def max_hits(actual, tickets):
    aset = set(actual)
    return max(len(aset & set(t)) for t in tickets)


def summarize(rows, label):
    if not rows:
        return
    print(f"--- {label} n={len(rows)} ---")
    print(
        f"A-BOX10 mean_max={statistics.mean(r['abox_max'] for r in rows):.6f} "
        f"3+={statistics.mean(r['abox_max']>=3 for r in rows):.6%} "
        f"4+={statistics.mean(r['abox_max']>=4 for r in rows):.6%} "
        f"5+={statistics.mean(r['abox_max']>=5 for r in rows):.6%}"
    )
    print(
        f"BLEND_PROXY mean_max={statistics.mean(r['blend_mean_max'] for r in rows):.6f} "
        f"3+={statistics.mean(r['blend_p3'] for r in rows):.6%} "
        f"4+={statistics.mean(r['blend_p4'] for r in rows):.6%} "
        f"5+={statistics.mean(r['blend_p5'] for r in rows):.6%}"
    )
    print(
        f"delta mean_max={statistics.mean(r['abox_max']-r['blend_mean_max'] for r in rows):+.6f} "
        f"mean_union={statistics.mean(r['union_size'] for r in rows):.3f} "
        f"mean_shared={statistics.mean(r['shared_size'] for r in rows):.3f}"
    )
    paired = [
        (
            r["abox_max"] > r["blend_mean_max"],
            abs(r["abox_max"] - r["blend_mean_max"]) < 1e-12,
            r["abox_max"] < r["blend_mean_max"],
        )
        for r in rows
    ]
    print(
        f"paired against Blend expected max: "
        f"A>B={sum(x[0] for x in paired)} "
        f"A=B={sum(x[1] for x in paired)} "
        f"A<B={sum(x[2] for x in paired)}"
    )
    print()


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    # Frozen A-BOX online state.
    boxes = {}
    for i in range(WINDOW, len(df)):
        boxes[int(df.iloc[i]["round"])] = box.snap(df.iloc[i-WINDOW:i])
    stats = box.empty_stats()

    rows = []
    for i in range(WINDOW + 1, len(df)):
        rnd = int(df.iloc[i]["round"])
        pr = int(df.iloc[i - 1]["round"])
        history = df.iloc[i-WINDOW:i]
        actual = box.truth(df.iloc[i])

        # A-BOX10, strict online chronology.
        cur, prev = boxes[rnd], boxes[pr]
        delta, sets = box.candidate_sets(cur, prev)
        band = box.band(abs(delta))
        if band == "zero":
            chosen_mag = 0.0
            mature = True
        else:
            alive, chosen_mag, _ = box.live_band(stats[band])
            mature = stats[band]["n"] >= box.MIN_HISTORY
            if not alive:
                chosen_mag = 0.0
        final7 = sets[chosen_mag]
        abox_tickets = blend10(cur, final7, delta)
        abox_max = max_hits(actual, abox_tickets)

        # GitHub-reproducible proxy for current Blend concept.
        core10, x10, pool, shared, weights = blend_proxy_pool(history)
        blend_maxes = []
        for rep in range(ALLOCATOR_REPS):
            seed = rnd * 1_000_003 + rep * 97 + 20260927
            pack = blend_proxy_pack(pool, weights, seed)
            blend_maxes.append(max_hits(actual, pack))

        rows.append({
            "round": rnd,
            "band": band,
            "mature": mature,
            "final_hits": box.hits(actual, final7),
            "abox_max": abox_max,
            "blend_mean_max": statistics.mean(blend_maxes),
            "blend_p3": statistics.mean(v >= 3 for v in blend_maxes),
            "blend_p4": statistics.mean(v >= 4 for v in blend_maxes),
            "blend_p5": statistics.mean(v >= 5 for v in blend_maxes),
            "union_size": len(pool),
            "shared_size": len(shared),
        })

        if band != "zero":
            base = sets[0.0]
            stats[band]["n"] += 1
            stats[band]["base_h"] += box.hits(actual, base)
            stats[band]["base_d"] += box.dist(actual, base)
            for m in box.MAGS:
                stats[band]["arms"][m]["h"] += box.hits(actual, sets[m])
                stats[band]["arms"][m]["d"] += box.dist(actual, sets[m])

    mature = [r for r in rows if r["mature"] and r["band"] != "zero"]

    print("=== LOTO7 GITHUB BLEND PROXY vs A-BOX10 v0 ===")
    print("BLEND_PROXY = CORE top10 union X top10, shared candidates weight x2, 10 tickets.")
    print(f"Allocator averaged over {ALLOCATOR_REPS} deterministic seeds per round.")
    print("A-BOX10 = frozen Final7 + 9 state-transition blend tickets.")
    print("No target outcome is used in prediction or allocation.")
    print("BOUNDARY: this is a reproducible proxy for the current Blend concept, not a historical replay of Floot AI outputs.")
    print()

    summarize(rows, "ALL")
    summarize(mature, "MATURE")
    summarize(mature[-200:], "LAST200 MATURE")
    summarize(mature[-100:], "LAST100 MATURE")

    print("=== BY A-BOX FINAL7 HIT COUNT | MATURE ===")
    for k in range(4):
        summarize([r for r in mature if r["final_hits"] == k], f"Final7={k}")
    summarize([r for r in mature if r["final_hits"] >= 4], "Final7=4+")


if __name__ == "__main__":
    main()
