from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

import lotocore

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_region_shuffle_null_v0.csv")
SUMMARY = Path("results/loto7_region_shuffle_null_summary_v0.csv")

WINDOW = 100
COVERAGE = 13
SHUFFLES = 200
SEED = 20260927
NUMBERS = list(range(1, 38))


def actual(row):
    return {int(row[f"n{i}"]) for i in range(1, 8)}


def local_density(scores):
    dens = {}
    for n in NUMBERS:
        dens[n] = sum(scores[k] for k in (n - 1, n, n + 1) if 1 <= k <= 37)
    return dens


def select_coverage(scores, k=COVERAGE):
    dens = local_density(scores)
    picked = sorted(
        sorted(NUMBERS, key=lambda n: (-dens[n], -scores[n], n))[:k]
    )
    return picked, dens


def merge_regions(picked):
    regions = []
    if not picked:
        return regions
    start = prev = picked[0]
    for n in picked[1:]:
        if n == prev + 1:
            prev = n
            continue
        regions.append((start, prev))
        start = prev = n
    regions.append((start, prev))
    return regions


def nearest_distance(truth, covered):
    vals = list(covered)
    return sum(min(abs(a - n) for n in vals) for a in truth) / 7.0


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(WINDOW, len(df)):
        history = df.iloc[i-WINDOW:i]
        truth = actual(df.iloc[i])

        snap = lotocore.score_snapshot(history)
        scores = {int(k): float(v) for k, v in snap["scores"].items()}

        original, _ = select_coverage(scores)
        regions = merge_regions(original)

        original_hits = len(set(original) & truth)
        original_dist = nearest_distance(truth, original)

        vals = [scores[n] for n in NUMBERS]
        sh_hits = []
        sh_dist = []
        sh_region_counts = []

        for _ in range(SHUFFLES):
            perm = vals[:]
            rng.shuffle(perm)
            shuffled_scores = {n: perm[j] for j, n in enumerate(NUMBERS)}
            covered, _ = select_coverage(shuffled_scores)
            sh_hits.append(len(set(covered) & truth))
            sh_dist.append(nearest_distance(truth, covered))
            sh_region_counts.append(len(merge_regions(covered)))

        shuffle_mean_hits = sum(sh_hits) / SHUFFLES
        shuffle_mean_dist = sum(sh_dist) / SHUFFLES
        shuffle_ge_hits_rate = sum(h >= original_hits for h in sh_hits) / SHUFFLES
        shuffle_le_dist_rate = sum(d <= original_dist for d in sh_dist) / SHUFFLES

        rows.append({
            "target_round": int(df.iloc[i]["round"]),
            "coverage": COVERAGE,
            "original_hits": original_hits,
            "shuffle_mean_hits": shuffle_mean_hits,
            "hit_advantage": original_hits - shuffle_mean_hits,
            "original_distance": original_dist,
            "shuffle_mean_distance": shuffle_mean_dist,
            "distance_advantage": shuffle_mean_dist - original_dist,
            "original_region_count": len(regions),
            "shuffle_mean_region_count": sum(sh_region_counts) / SHUFFLES,
            "shuffle_ge_original_hits_rate": shuffle_ge_hits_rate,
            "shuffle_le_original_distance_rate": shuffle_le_dist_rate,
            "covered": "-".join(f"{n:02d}" for n in original),
            "regions": "|".join(
                f"{a:02d}" if a == b else f"{a:02d}-{b:02d}"
                for a, b in regions
            ),
            "actual": "-".join(f"{n:02d}" for n in sorted(truth)),
        })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    summary = pd.DataFrame([{
        "targets": len(res),
        "coverage": COVERAGE,
        "shuffles_per_target": SHUFFLES,
        "original_mean_hits": res.original_hits.mean(),
        "shuffle_mean_hits": res.shuffle_mean_hits.mean(),
        "mean_hit_advantage": res.hit_advantage.mean(),
        "original_mean_distance": res.original_distance.mean(),
        "shuffle_mean_distance": res.shuffle_mean_distance.mean(),
        "mean_distance_advantage": res.distance_advantage.mean(),
        "original_mean_region_count": res.original_region_count.mean(),
        "shuffle_mean_region_count": res.shuffle_mean_region_count.mean(),
        "rounds_original_beats_shuffle_hits_rate": (res.hit_advantage > 0).mean(),
        "rounds_original_beats_shuffle_distance_rate": (res.distance_advantage > 0).mean(),
        "mean_shuffle_ge_original_hits_rate": res.shuffle_ge_original_hits_rate.mean(),
        "mean_shuffle_le_original_distance_rate": res.shuffle_le_original_distance_rate.mean(),
    }])
    summary.to_csv(SUMMARY, index=False)

    print("=== LOTO7 REGION SHUFFLE NULL v0 ===")
    print(f"targets={len(res)} rounds={int(res.target_round.min())}..{int(res.target_round.max())}")
    print(f"coverage={COVERAGE} shuffles_per_target={SHUFFLES}")
    print("Region extractor: local density score[n-1]+score[n]+score[n+1],")
    print("take top coverage integers, then merge adjacent selected integers into regions.")
    print("Shuffle null preserves the 37 score values but permutes their number labels.")
    print()
    print(f"mean hits: original={res.original_hits.mean():.6f} shuffle={res.shuffle_mean_hits.mean():.6f} advantage={res.hit_advantage.mean():+.6f}")
    print(f"mean nearest distance: original={res.original_distance.mean():.6f} shuffle={res.shuffle_mean_distance.mean():.6f} advantage={res.distance_advantage.mean():+.6f}")
    print(f"mean region count: original={res.original_region_count.mean():.6f} shuffle={res.shuffle_mean_region_count.mean():.6f}")
    print(f"rounds original > shuffle mean hits={(res.hit_advantage > 0).mean():.6f}")
    print(f"rounds original < shuffle mean distance={(res.distance_advantage > 0).mean():.6f}")
    print(f"mean per-round null tail P(hit >= original)={res.shuffle_ge_original_hits_rate.mean():.6f}")
    print(f"mean per-round null tail P(distance <= original)={res.shuffle_le_original_distance_rate.mean():.6f}")
    print()
    print("RECENT 20")
    print(res.tail(20).to_string(index=False))
    print(f"saved -> {OUT}")
    print(f"saved -> {SUMMARY}")


if __name__ == "__main__":
    main()
