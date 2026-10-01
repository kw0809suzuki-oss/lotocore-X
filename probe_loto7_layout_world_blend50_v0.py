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
OUT_JSON = Path("results/loto7_layout_world_blend50_v0_summary.json")
OUT_CSV = Path("results/loto7_layout_world_blend50_v0_rounds.csv")


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


def covered_subsets(bundle, k):
    out = set()
    for t in bundle:
        out.update(itertools.combinations(sorted(t), k))
    return out


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
        rows.append({
            "bundle": b,
            "degree_variance": d,
            "pair_reuse_sq": p,
            "pairs": covered_subsets(b, 2),
            "triples": covered_subsets(b, 3),
            "quads": covered_subsets(b, 4),
            "quints": covered_subsets(b, 5),
        })

    zd, zp = zscores(degs), zscores(pairs)
    for r, a, b in zip(rows, zd, zp):
        r["a_score"] = a + b

    rows.sort(key=lambda r: r["a_score"])
    return rows[:keep]


def diverse_layout_blend(blocks, count=5):
    chosen = []
    used = set()
    q5, q4, q3 = set(), set(), set()

    for _ in range(count):
        best_i, best_key = None, None
        for i, r in enumerate(blocks):
            if i in used:
                continue
            new5 = len(r["quints"] - q5)
            new4 = len(r["quads"] - q4)
            new3 = len(r["triples"] - q3)
            # We do not know the winning numbers. Diversify the layout worlds
            # themselves: maximize new 5-way co-location opportunities first,
            # then 4-way and 3-way opportunities, while staying in A-good.
            key = (new5, new4, new3, -r["a_score"])
            if best_key is None or key > best_key:
                best_key, best_i = key, i

        used.add(best_i)
        r = blocks[best_i]
        chosen.append(r)
        q5.update(r["quints"])
        q4.update(r["quads"])
        q3.update(r["triples"])

    tickets = [t for r in chosen for t in r["bundle"]]
    return tickets, chosen


def random_layout_blend(blocks, rng, count=5):
    chosen = rng.sample(blocks, count)
    tickets = [t for r in chosen for t in r["bundle"]]
    return tickets, chosen


def stats(tickets, actual):
    hits = [len(set(t) & actual) for t in tickets]
    mx = max(hits)
    return {
        "max_hits": mx,
        "reach3": int(mx >= 3),
        "reach4": int(mx >= 4),
        "reach5": int(mx >= 5),
        "reach6": int(mx >= 6),
        "reach7": int(mx >= 7),
    }


def subset_union_size(chosen, key):
    u = set()
    for r in chosen:
        u.update(r[key])
    return len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--keep", type=int, default=120)
    ap.add_argument("--random-reps", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(args.window, len(df)):
        history = df.iloc[i-args.window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        candidates = core18(history)
        capture = len(set(candidates) & actual)

        blocks = generate_a_good_pool(
            candidates,
            seed=rnd * 1000003 + 5050,
            pool_size=args.pool,
            keep=args.keep,
        )

        smart_tickets, smart_blocks = diverse_layout_blend(blocks, 5)
        smart = stats(smart_tickets, actual)

        rng = random.Random(rnd * 1000003 + 9090)
        rand_acc = {k: 0.0 for k in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]}
        rand_u5 = rand_u4 = rand_u3 = 0.0

        for _ in range(args.random_reps):
            rt, rb = random_layout_blend(blocks, rng, 5)
            s = stats(rt, actual)
            for k in rand_acc:
                rand_acc[k] += s[k]
            rand_u5 += subset_union_size(rb, "quints")
            rand_u4 += subset_union_size(rb, "quads")
            rand_u3 += subset_union_size(rb, "triples")

        for k in rand_acc:
            rand_acc[k] /= args.random_reps
        rand_u5 /= args.random_reps
        rand_u4 /= args.random_reps
        rand_u3 /= args.random_reps

        rows.append({
            "round": rnd,
            "capture": capture,
            "smart_quint_union": subset_union_size(smart_blocks, "quints"),
            "random_quint_union": rand_u5,
            "smart_quad_union": subset_union_size(smart_blocks, "quads"),
            "random_quad_union": rand_u4,
            "smart_triple_union": subset_union_size(smart_blocks, "triples"),
            "random_triple_union": rand_u3,
            **{f"smart_{k}": v for k, v in smart.items()},
            **{f"random_{k}": v for k, v in rand_acc.items()},
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    def summarize(g):
        result = {}
        for m in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]:
            a = float(g[f"smart_{m}"].mean())
            b = float(g[f"random_{m}"].mean())
            result[m] = {"smart": a, "random": b, "delta": a - b}
        result["layout_union"] = {
            "quint_smart": float(g["smart_quint_union"].mean()),
            "quint_random": float(g["random_quint_union"].mean()),
            "quad_smart": float(g["smart_quad_union"].mean()),
            "quad_random": float(g["random_quad_union"].mean()),
            "triple_smart": float(g["smart_triple_union"].mean()),
            "triple_random": float(g["random_triple_union"].mean()),
        }
        return result

    summary = {
        "probe": "LOTO7 Layout-World Blend50 v0",
        "definition": {
            "both_sides": "Same CORE18 and same A-good 10-ticket block pool.",
            "smart_50": "Choose 5 distinct A-good blocks to maximize new 5-subset co-location opportunities, then 4-subset, then 3-subset opportunities.",
            "random_50": "Choose 5 A-good blocks randomly from the same pool; averaged over repetitions.",
            "target_leakage": False,
        },
        "history_draws": int(len(df)),
        "usable_targets": int(len(out)),
        "pool": args.pool,
        "keep": args.keep,
        "random_reps": args.random_reps,
        "overall": summarize(out),
        "capture_5_6": summarize(out[out["capture"].isin([5, 6])]),
        "by_capture": {},
    }

    for cap, g in out.groupby("capture"):
        summary["by_capture"][str(int(cap))] = {"rounds": int(len(g)), **summarize(g)}

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 LAYOUT-WORLD BLEND50 v0 ===")
    print(f"history_draws={len(df)} usable_targets={len(out)} pool={args.pool} keep={args.keep} random_reps={args.random_reps}")

    for label, g in [("overall", out), ("capture56", out[out["capture"].isin([5,6])])]:
        print(f"[{label}] n={len(g)}")
        for m in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]:
            a = g[f"smart_{m}"].mean()
            b = g[f"random_{m}"].mean()
            print(f"{m}: smart={a:.6f} random={b:.6f} delta={a-b:+.6f}")
        print(
            "layout_union: "
            f"quint {g['smart_quint_union'].mean():.2f}/{g['random_quint_union'].mean():.2f} "
            f"quad {g['smart_quad_union'].mean():.2f}/{g['random_quad_union'].mean():.2f} "
            f"triple {g['smart_triple_union'].mean():.2f}/{g['random_triple_union'].mean():.2f}"
        )

    for cap in sorted(out["capture"].unique()):
        if cap < 4:
            continue
        g = out[out["capture"] == cap]
        print(
            f"capture={cap} n={len(g)} "
            f"5+ smart={g['smart_reach5'].mean():.6f} random={g['random_reach5'].mean():.6f} "
            f"delta={g['smart_reach5'].mean()-g['random_reach5'].mean():+.6f} "
            f"6+ smart={g['smart_reach6'].mean():.6f} random={g['random_reach6'].mean():.6f}"
        )

    print("NOTE: This tests diversification across 10-ticket layout worlds, not characteristic-number diversification.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
