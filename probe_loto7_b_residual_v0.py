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
OUT_JSON = Path("results/loto7_b_residual_v0_summary.json")
OUT_CSV = Path("results/loto7_b_residual_v0_holdout.csv")


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
                pair_cost = sum(pair[tuple(sorted((n, x)))] for x in chosen)
                return (pair_cost, -remaining[n], candidates.index(n))
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


def b_features(bundle, candidates):
    rank = {n: i + 1 for i, n in enumerate(candidates)}
    ticket_means = []
    spans = []
    adjacent_density = []
    near_pair_density = []
    top3_counts = []
    top6_counts = []
    top9_counts = []

    for t in bundle:
        rs = sorted(rank[n] for n in t)
        ticket_means.append(sum(rs) / 7)
        spans.append(rs[-1] - rs[0])
        adjacent_density.append(sum((b - a) == 1 for a, b in zip(rs, rs[1:])) / 6)
        pairs = list(itertools.combinations(rs, 2))
        near_pair_density.append(sum(abs(a - b) <= 3 for a, b in pairs) / len(pairs))
        top3_counts.append(sum(r <= 3 for r in rs))
        top6_counts.append(sum(r <= 6 for r in rs))
        top9_counts.append(sum(r <= 9 for r in rs))

    def var(xs):
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    return {
        "rank_mean_variance": var(ticket_means),
        "rank_span_mean": sum(spans) / len(spans),
        "adjacent_rank_density": sum(adjacent_density) / len(adjacent_density),
        "near_rank_pair_density": sum(near_pair_density) / len(near_pair_density),
        "top3_cooccurrence_max": max(top3_counts),
        "top6_cooccurrence_max": max(top6_counts),
        "top9_cooccurrence_max": max(top9_counts),
        "top6_slot_share": sum(top6_counts) / 70,
        "top9_slot_share": sum(top9_counts) / 70,
    }


def max_hits(bundle, actual):
    return max(len(set(t) & actual) for t in bundle)


def zscores(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    s = math.sqrt(v)
    if s == 0:
        return [0.0] * len(xs)
    return [(x - m) / s for x in xs]


def generate_pool(candidates, seed, n):
    rng = random.Random(seed)
    base = structured_bundle(candidates)
    bundles = [base]
    seen = {tuple(base)}
    while len(bundles) < n:
        b = mutate_bundle(base, candidates, rng)
        key = tuple(b)
        if key in seen:
            continue
        seen.add(key)
        bundles.append(b)

    degs, pairs = [], []
    rows = []
    for b in bundles:
        d, p = a_features(b, candidates)
        degs.append(d)
        pairs.append(p)
        rows.append({"bundle": b, "degree_variance": d, "pair_reuse_sq": p, **b_features(b, candidates)})

    zd = zscores(degs)
    zp = zscores(pairs)
    for r, a, b in zip(rows, zd, zp):
        r["a_score"] = a + b  # lower = more A-like: balanced degree + low pair reuse

    rows.sort(key=lambda r: r["a_score"])
    keep = max(20, len(rows) // 4)
    return rows[:keep]


def effect(good, bad):
    if not good or not bad:
        return 0.0
    allv = good + bad
    m = sum(allv) / len(allv)
    s = math.sqrt(sum((x - m) ** 2 for x in allv) / len(allv))
    if s == 0:
        return 0.0
    return ((sum(good) / len(good)) - (sum(bad) / len(bad))) / s


def discover(df, window, discovery_targets, bundles_per_round):
    feats = [
        "rank_mean_variance",
        "rank_span_mean",
        "adjacent_rank_density",
        "near_rank_pair_density",
        "top3_cooccurrence_max",
        "top6_cooccurrence_max",
        "top9_cooccurrence_max",
        "top6_slot_share",
        "top9_slot_share",
    ]
    records = []

    for i in range(window, window + discovery_targets):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = core18(history)
        pool = generate_pool(cand, rnd * 1000003 + 41, bundles_per_round)
        for r in pool:
            h = max_hits(r["bundle"], actual)
            records.append((i, h, r))

    block_size = discovery_targets // 4
    summaries = []
    for f in feats:
        es = []
        event_counts = []
        for b in range(4):
            lo = window + b * block_size
            hi = window + (b + 1) * block_size if b < 3 else window + discovery_targets
            block = [x for x in records if lo <= x[0] < hi]
            good = [x[2][f] for x in block if x[1] >= 5]
            bad = [x[2][f] for x in block if x[1] < 5]
            es.append(effect(good, bad))
            event_counts.append(len(good))
        stable = all(e > 0 for e in es) or all(e < 0 for e in es)
        summaries.append({
            "feature": f,
            "effects": es,
            "five_plus_counts": event_counts,
            "stable_direction": stable,
            "direction": 1 if sum(es) >= 0 else -1,
            "min_abs_effect": min(abs(e) for e in es),
            "mean_abs_effect": sum(abs(e) for e in es) / 4,
        })

    stable = [x for x in summaries if x["stable_direction"]]
    chosen = max(stable, key=lambda x: (x["min_abs_effect"], x["mean_abs_effect"])) if stable else max(
        summaries, key=lambda x: x["mean_abs_effect"]
    )
    return chosen, summaries


def holdout(df, window, start_i, bundles_per_round, chosen):
    f = chosen["feature"]
    reverse = chosen["direction"] > 0
    rows = []

    for i in range(start_i, len(df)):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = core18(history)
        pool = generate_pool(cand, rnd * 1000003 + 911, bundles_per_round)

        pool.sort(key=lambda r: r[f], reverse=reverse)
        q = max(5, len(pool) // 4)
        b_high = pool[:q]
        b_low = pool[-q:]

        def stats(group):
            hs = [max_hits(r["bundle"], actual) for r in group]
            return {
                "mean_max": sum(hs) / len(hs),
                "five_plus": sum(h >= 5 for h in hs) / len(hs),
                "six_plus": sum(h >= 6 for h in hs) / len(hs),
                "seven": sum(h >= 7 for h in hs) / len(hs),
                "a_score": sum(r["a_score"] for r in group) / len(group),
                "b_value": sum(r[f] for r in group) / len(group),
            }

        hi = stats(b_high)
        lo = stats(b_low)
        rows.append({
            "round": rnd,
            "date": row.get("date", ""),
            "core18_capture": len(set(cand) & actual),
            "feature": f,
            **{f"B_{k}": v for k, v in hi.items()},
            **{f"notB_{k}": v for k, v in lo.items()},
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--discovery-targets", type=int, default=400)
    ap.add_argument("--discovery-bundles", type=int, default=240)
    ap.add_argument("--holdout-bundles", type=int, default=400)
    args = ap.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    chosen, candidates = discover(df, args.window, args.discovery_targets, args.discovery_bundles)
    hold = holdout(df, args.window, args.window + args.discovery_targets, args.holdout_bundles, chosen)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    hold.to_csv(OUT_CSV, index=False)

    summary = {
        "probe": "LOTO7 B Residual v0",
        "definition": "A = known 3+/4+ geometry (low degree variance + low pair reuse). B = rank-relationship feature that separates 5+ vs <5 inside A-good bundles on discovery only.",
        "chosen": chosen,
        "discovery_candidates": candidates,
        "holdout_rounds": int(len(hold)),
        "holdout": {},
    }
    for m in ["mean_max", "five_plus", "six_plus", "seven"]:
        b = float(hold[f"B_{m}"].mean())
        n = float(hold[f"notB_{m}"].mean())
        summary["holdout"][m] = {"B": b, "notB": n, "delta": b - n}

    summary["a_match"] = {
        "B_mean_a_score": float(hold["B_a_score"].mean()),
        "notB_mean_a_score": float(hold["notB_a_score"].mean()),
        "delta": float(hold["B_a_score"].mean() - hold["notB_a_score"].mean()),
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 B RESIDUAL v0 ===")
    print("A is held by structural geometry; B is discovered only inside the A-good region.")
    print(f"chosen_B={chosen['feature']} direction={'high' if chosen['direction'] > 0 else 'low'}")
    print("discovery_effects=" + ",".join(f"{x:+.4f}" for x in chosen["effects"]))
    print("discovery_5plus_counts=" + ",".join(str(x) for x in chosen["five_plus_counts"]))
    print(f"stable_direction={chosen['stable_direction']}")
    print(f"holdout_rounds={len(hold)}")
    print(f"A_match: B={summary['a_match']['B_mean_a_score']:.4f} notB={summary['a_match']['notB_mean_a_score']:.4f} delta={summary['a_match']['delta']:+.4f}")
    for m, x in summary["holdout"].items():
        print(f"{m}: B={x['B']:.6f} notB={x['notB']:.6f} delta={x['delta']:+.6f}")
    print("NOTE: B uses rank-relations, not lottery-number identities. Holdout target values do not select bundles.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
