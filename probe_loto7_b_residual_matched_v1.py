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
OUT_JSON = Path("results/loto7_b_residual_matched_v1_summary.json")
OUT_CSV = Path("results/loto7_b_residual_matched_v1_pairs.csv")


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
    return out


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
    return out


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


def b_value(bundle, candidates):
    rank = {n: i + 1 for i, n in enumerate(candidates)}
    return max(sum(rank[n] <= 3 for n in t) for t in bundle)


def zscores(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    s = math.sqrt(v)
    return [0.0] * len(xs) if s == 0 else [(x - m) / s for x in xs]


def generate_pool(candidates, seed, n):
    rng = random.Random(seed)
    base = structured_bundle(candidates)
    bundles = [base]
    seen = {tuple(base)}
    while len(bundles) < n:
        b = mutate_bundle(base, candidates, rng)
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            bundles.append(b)

    deg, pair, rows = [], [], []
    for b in bundles:
        d, p = a_features(b, candidates)
        deg.append(d); pair.append(p)
        rows.append({"bundle": b, "degree_variance": d, "pair_reuse_sq": p, "b": b_value(b, candidates)})

    zd, zp = zscores(deg), zscores(pair)
    for r, x, y in zip(rows, zd, zp):
        r["a_score"] = x + y
    rows.sort(key=lambda r: r["a_score"])
    return rows[:max(40, len(rows)//3)]


def max_hits(bundle, actual):
    return max(len(set(t) & actual) for t in bundle)


def match_pairs(pool, eps):
    high = [r for r in pool if r["b"] >= 2]
    low = [r for r in pool if r["b"] <= 1]
    used = set()
    pairs = []
    for h in sorted(high, key=lambda r: r["a_score"]):
        best = None
        best_j = None
        for j, l in enumerate(low):
            if j in used:
                continue
            d = abs(h["a_score"] - l["a_score"])
            if d <= eps and (best is None or d < best):
                best = d
                best_j = j
        if best_j is not None:
            used.add(best_j)
            pairs.append((h, low[best_j], best))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []
    total_pairs = 0

    for i in range(args.window, len(df)):
        history = df.iloc[i-args.window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = core18(history)
        pool = generate_pool(cand, rnd * 1000003 + 123, args.pool)
        pairs = match_pairs(pool, args.eps)
        total_pairs += len(pairs)
        for h, l, dist in pairs:
            hh = max_hits(h["bundle"], actual)
            lh = max_hits(l["bundle"], actual)
            rows.append({
                "round": rnd,
                "core18_capture": len(set(cand) & actual),
                "a_high": h["a_score"],
                "a_low": l["a_score"],
                "a_distance": dist,
                "b_high": h["b"],
                "b_low": l["b"],
                "high_max_hits": hh,
                "low_max_hits": lh,
                "high_5plus": int(hh >= 5),
                "low_5plus": int(lh >= 5),
                "high_6plus": int(hh >= 6),
                "low_6plus": int(lh >= 6),
                "high_7": int(hh >= 7),
                "low_7": int(lh >= 7),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "probe": "LOTO7 B Residual Matched v1",
        "targets": int(len(df) - args.window),
        "pairs": int(len(out)),
        "epsilon": args.eps,
        "mean_a_distance": float(out["a_distance"].mean()) if len(out) else None,
        "max_a_distance": float(out["a_distance"].max()) if len(out) else None,
        "results": {}
    }
    for m in ["max_hits", "5plus", "6plus", "7"]:
        a = float(out[f"high_{m}"].mean()) if len(out) else None
        b = float(out[f"low_{m}"].mean()) if len(out) else None
        summary["results"][m] = {"B_high": a, "B_low": b, "delta": (a-b) if a is not None else None}

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 B RESIDUAL MATCHED v1 ===")
    print(f"history_draws={len(df)} usable_targets={len(df)-args.window} window={args.window}")
    print(f"matched_pairs={len(out)} epsilon={args.eps}")
    if len(out):
        print(f"A_distance mean={out['a_distance'].mean():.6f} max={out['a_distance'].max():.6f}")
        for m, x in summary["results"].items():
            print(f"{m}: B_high={x['B_high']:.6f} B_low={x['B_low']:.6f} delta={x['delta']:+.6f}")
    print("B = top3 cooccurrence only; A is pair-matched within epsilon.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
