from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
OUT_JSON = Path("results/loto7_capture56_layout_headroom_v0_summary.json")
OUT_CSV = Path("results/loto7_capture56_layout_headroom_v0_rounds.csv")


def actual_numbers(row):
    return set(int(row[f"n{i}"]) for i in range(1, 8))


def core18(history):
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return [n for n, _ in sorted(ranks.items(), key=lambda kv: (kv[1], kv[0]))[:18]]


def structured_bundle(candidates):
    target = {n: 3 for n in candidates}
    for n in candidates[:16]:
        target[n] = 4
    remaining = target.copy()
    pair = defaultdict(int)
    out = []
    for slot in range(10):
        left = 10 - slot
        chosen = [n for n in candidates if remaining[n] == left]
        while len(chosen) < 7:
            pool = [n for n in candidates if remaining[n] > 0 and n not in chosen]
            def key(n):
                return (
                    sum(pair[tuple(sorted((n, x)))] for x in chosen),
                    -remaining[n],
                    candidates.index(n),
                )
            chosen.append(min(pool, key=key))
        chosen = tuple(sorted(chosen))
        out.append(chosen)
        for n in chosen:
            remaining[n] -= 1
        for a, b in itertools.combinations(chosen, 2):
            pair[tuple(sorted((a, b)))] += 1
    return tuple(out)


def mutate_bundle(base, candidates, rng):
    out = []
    for ticket in base:
        s = set(ticket)
        swaps = rng.choice([1, 1, 2, 2, 3])
        for _ in range(swaps):
            drop = rng.choice(sorted(s))
            add = rng.choice([n for n in candidates if n not in s])
            s.remove(drop)
            s.add(add)
        out.append(tuple(sorted(s)))
    return tuple(out)


def a_features(bundle, candidates):
    degree = Counter(n for t in bundle for n in t)
    degs = [degree[n] for n in candidates]
    mean_deg = sum(degs) / len(degs)
    degree_variance = sum((x - mean_deg) ** 2 for x in degs) / len(degs)

    pair = Counter()
    for t in bundle:
        pair.update(itertools.combinations(t, 2))
    pair_reuse_sq = sum(v * v for v in pair.values())
    return degree_variance, pair_reuse_sq


def zscores(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    s = math.sqrt(v)
    return [0.0] * len(xs) if s == 0 else [(x - m) / s for x in xs]


def generate_a_good_pool(candidates, seed, pool_size, keep):
    rng = random.Random(seed)
    base = structured_bundle(candidates)
    bundles = [base]
    seen = {base}
    while len(bundles) < pool_size:
        b = mutate_bundle(base, candidates, rng)
        if b not in seen:
            seen.add(b)
            bundles.append(b)

    degs, pairs, rows = [], [], []
    for b in bundles:
        d, p = a_features(b, candidates)
        degs.append(d)
        pairs.append(p)
        rows.append({"bundle": b, "degree_variance": d, "pair_reuse_sq": p})

    zd, zp = zscores(degs), zscores(pairs)
    for r, a, b in zip(rows, zd, zp):
        r["a_score"] = a + b

    rows.sort(key=lambda r: r["a_score"])
    return rows[:keep]


def max_hits(bundle, actual):
    return max(len(set(t) & actual) for t in bundle)


def captured_set(candidates, actual):
    return tuple(sorted(set(candidates) & actual))


def captured_subset_coverage(bundle, captured, k):
    if len(captured) < k:
        return 0.0
    target = set(itertools.combinations(captured, k))
    covered = set()
    for t in bundle:
        hit = sorted(set(t) & set(captured))
        if len(hit) >= k:
            covered.update(itertools.combinations(hit, k))
    return len(covered & target) / len(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--keep", type=int, default=120)
    args = ap.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(args.window, len(df)):
        history = df.iloc[i-args.window:i]
        row = df.iloc[i]
        actual = actual_numbers(row)
        candidates = core18(history)
        cap = len(set(candidates) & actual)

        if cap not in (5, 6):
            continue

        base = structured_bundle(candidates)
        base_hits = max_hits(base, actual)
        captured = captured_set(candidates, actual)

        pool = generate_a_good_pool(
            candidates,
            seed=int(row["round"]) * 1000003 + 5656,
            pool_size=args.pool,
            keep=args.keep,
        )

        pool_hits = [max_hits(r["bundle"], actual) for r in pool]
        reach5 = [h >= 5 for h in pool_hits]

        best = max(pool_hits)
        reach5_share = sum(reach5) / len(reach5)

        base_pair_cov = captured_subset_coverage(base, captured, 2)
        base_triple_cov = captured_subset_coverage(base, captured, 3)
        base_quad_cov = captured_subset_coverage(base, captured, 4)

        best_rows = [r for r, h in zip(pool, pool_hits) if h == best]
        best_bundle = best_rows[0]["bundle"]
        best_pair_cov = captured_subset_coverage(best_bundle, captured, 2)
        best_triple_cov = captured_subset_coverage(best_bundle, captured, 3)
        best_quad_cov = captured_subset_coverage(best_bundle, captured, 4)

        rows.append({
            "round": int(row["round"]),
            "capture": cap,
            "baseline_max_hits": base_hits,
            "pool_best_max_hits": best,
            "pool_mean_max_hits": sum(pool_hits) / len(pool_hits),
            "pool_reach5_share": reach5_share,
            "layout_headroom": best - base_hits,
            "baseline_pair_cov": base_pair_cov,
            "baseline_triple_cov": base_triple_cov,
            "baseline_quad_cov": base_quad_cov,
            "best_pair_cov": best_pair_cov,
            "best_triple_cov": best_triple_cov,
            "best_quad_cov": best_quad_cov,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "probe": "LOTO7 Capture 5/6 Layout Headroom v0",
        "history_draws": int(len(df)),
        "usable_targets": int(len(df) - args.window),
        "capture56_rounds": int(len(out)),
        "pool_per_round": args.pool,
        "a_good_keep": args.keep,
        "overall": {
            "baseline_reach5": float((out["baseline_max_hits"] >= 5).mean()) if len(out) else None,
            "pool_oracle_reach5": float((out["pool_best_max_hits"] >= 5).mean()) if len(out) else None,
            "mean_pool_reach5_share": float(out["pool_reach5_share"].mean()) if len(out) else None,
            "mean_layout_headroom": float(out["layout_headroom"].mean()) if len(out) else None,
            "headroom_positive_share": float((out["layout_headroom"] > 0).mean()) if len(out) else None,
        },
        "by_capture": {},
    }

    for cap, g in out.groupby("capture"):
        summary["by_capture"][str(int(cap))] = {
            "rounds": int(len(g)),
            "baseline_mean_max": float(g["baseline_max_hits"].mean()),
            "pool_best_mean_max": float(g["pool_best_max_hits"].mean()),
            "baseline_reach5": float((g["baseline_max_hits"] >= 5).mean()),
            "pool_oracle_reach5": float((g["pool_best_max_hits"] >= 5).mean()),
            "mean_pool_reach5_share": float(g["pool_reach5_share"].mean()),
            "mean_layout_headroom": float(g["layout_headroom"].mean()),
            "headroom_positive_share": float((g["layout_headroom"] > 0).mean()),
            "baseline_pair_cov": float(g["baseline_pair_cov"].mean()),
            "baseline_triple_cov": float(g["baseline_triple_cov"].mean()),
            "baseline_quad_cov": float(g["baseline_quad_cov"].mean()),
            "best_pair_cov": float(g["best_pair_cov"].mean()),
            "best_triple_cov": float(g["best_triple_cov"].mean()),
            "best_quad_cov": float(g["best_quad_cov"].mean()),
        }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 CAPTURE 5/6 LAYOUT HEADROOM v0 ===")
    print(f"history_draws={len(df)} usable_targets={len(df)-args.window} capture56_rounds={len(out)}")
    print(f"pool={args.pool} A_good_keep={args.keep}")
    x = summary["overall"]
    print(
        f"overall baseline_5+={x['baseline_reach5']:.6f} "
        f"pool_oracle_5+={x['pool_oracle_reach5']:.6f} "
        f"mean_pool_5+_share={x['mean_pool_reach5_share']:.6f} "
        f"mean_headroom={x['mean_layout_headroom']:.6f} "
        f"positive_headroom_share={x['headroom_positive_share']:.6f}"
    )
    for k in sorted(summary["by_capture"], key=int):
        y = summary["by_capture"][k]
        print(
            f"capture={k} n={y['rounds']} "
            f"baseline_mean={y['baseline_mean_max']:.6f} best_mean={y['pool_best_mean_max']:.6f} "
            f"baseline_5+={y['baseline_reach5']:.6f} oracle_5+={y['pool_oracle_reach5']:.6f} "
            f"pool_5+_share={y['mean_pool_reach5_share']:.6f} "
            f"headroom={y['mean_layout_headroom']:.6f} positive={y['headroom_positive_share']:.6f}"
        )
        print(
            f"  subset coverage baseline pair/triple/quad="
            f"{y['baseline_pair_cov']:.4f}/{y['baseline_triple_cov']:.4f}/{y['baseline_quad_cov']:.4f} "
            f"best={y['best_pair_cov']:.4f}/{y['best_triple_cov']:.4f}/{y['best_quad_cov']:.4f}"
        )
    print("NOTE: pool_oracle is diagnostic only; it selects the best pre-generated A-good bundle after seeing the result.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
