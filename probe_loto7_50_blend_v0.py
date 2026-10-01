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
OUT_JSON = Path("results/loto7_50_blend_v0_summary.json")
OUT_CSV = Path("results/loto7_50_blend_v0_rounds.csv")


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
    if s == 0:
        return [0.0] * len(xs)
    return [(x - m) / s for x in xs]


def covered_subsets(bundle, k):
    out = set()
    for t in bundle:
        out.update(itertools.combinations(sorted(t), k))
    return out


def characteristic_numbers(bundle, candidates, topn=3):
    # Structural role only: how many distinct partner contexts this number bridges
    # inside one 10-ticket A bundle. No target/winning numbers are used.
    score = {}
    for n in candidates:
        pair_context = set()
        triple_context = set()
        for t in bundle:
            if n not in t:
                continue
            others = sorted(x for x in t if x != n)
            pair_context.update(itertools.combinations(others, 2))
            triple_context.update(itertools.combinations(others, 3))
        score[n] = len(pair_context) + 0.25 * len(triple_context)
    ranked = sorted(candidates, key=lambda n: (-score[n], candidates.index(n)))
    return tuple(ranked[:topn]), score


def generate_a_good_blocks(candidates, seed, pool_size, keep):
    rng = random.Random(seed)
    base = tuple(structured_bundle(candidates))
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
        sig, sig_scores = characteristic_numbers(b, candidates)
        rows.append({
            "bundle": b,
            "degree_variance": d,
            "pair_reuse_sq": p,
            "signature": sig,
            "signature_scores": sig_scores,
            "triples": covered_subsets(b, 3),
            "quads": covered_subsets(b, 4),
        })

    zd, zp = zscores(degs), zscores(pairs)
    for r, a, b in zip(rows, zd, zp):
        r["a_score"] = a + b

    rows.sort(key=lambda r: r["a_score"])
    return rows[:keep]


def smart_blend(blocks, count=5):
    chosen = []
    used = set()
    sig_union = set()
    triple_union = set()
    quad_union = set()

    for _ in range(count):
        best_i = None
        best_key = None
        for i, r in enumerate(blocks):
            if i in used:
                continue
            new_sig = len(set(r["signature"]) - sig_union)
            new_q = len(r["quads"] - quad_union)
            new_t = len(r["triples"] - triple_union)
            # Primary: diversify structural characteristic numbers.
            # Secondary: preserve complementary 4- and 3-subset coverage.
            key = (new_sig, new_q, new_t, -r["a_score"])
            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        used.add(best_i)
        r = blocks[best_i]
        chosen.append(r)
        sig_union.update(r["signature"])
        triple_union.update(r["triples"])
        quad_union.update(r["quads"])

    tickets = []
    for r in chosen:
        tickets.extend(r["bundle"])
    return tickets, chosen


def plain_blend(blocks, rng, count=5):
    chosen = rng.sample(blocks, count)
    tickets = []
    for r in chosen:
        tickets.extend(r["bundle"])
    return tickets, chosen


def stats_for_tickets(tickets, actual):
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


def signature_union_size(chosen):
    u = set()
    for r in chosen:
        u.update(r["signature"])
    return len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--plain-reps", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(args.window, len(df)):
        history = df.iloc[i-args.window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        candidates = core18(history)

        blocks = generate_a_good_blocks(
            candidates,
            seed=rnd * 1000003 + 4242,
            pool_size=args.pool,
            keep=args.keep,
        )

        smart_tickets, smart_blocks = smart_blend(blocks, 5)
        smart = stats_for_tickets(smart_tickets, actual)

        rng = random.Random(rnd * 1000003 + 9898)
        plain_acc = {k: 0.0 for k in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]}
        plain_sig = 0.0
        for _ in range(args.plain_reps):
            plain_tickets, plain_blocks = plain_blend(blocks, rng, 5)
            s = stats_for_tickets(plain_tickets, actual)
            for k in plain_acc:
                plain_acc[k] += s[k]
            plain_sig += signature_union_size(plain_blocks)

        for k in plain_acc:
            plain_acc[k] /= args.plain_reps
        plain_sig /= args.plain_reps

        rows.append({
            "round": rnd,
            "core18_capture": len(set(candidates) & actual),
            "smart_signature_union": signature_union_size(smart_blocks),
            "plain_signature_union": plain_sig,
            **{f"smart_{k}": v for k, v in smart.items()},
            **{f"plain_{k}": v for k, v in plain_acc.items()},
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "probe": "LOTO7 50-ticket structural blend v0",
        "definition": {
            "A": "Each 10-ticket block is selected from the A-good region (low degree variance + low pair reuse).",
            "characteristic_number": "Within each 10-ticket block, numbers are ranked only by distinct partner-context bridging in triples/quads.",
            "smart_50": "Five A-good 10-ticket blocks chosen to diversify characteristic numbers first, then complementary quad/triple coverage.",
            "plain_50": "Five A-good 10-ticket blocks sampled randomly from the exact same candidate block pool.",
            "no_target_leakage": True,
        },
        "history_draws": int(len(df)),
        "usable_targets": int(len(out)),
        "pool_per_round": args.pool,
        "a_good_keep_per_round": args.keep,
        "plain_reps": args.plain_reps,
        "results": {},
    }

    for m in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]:
        a = float(out[f"smart_{m}"].mean())
        b = float(out[f"plain_{m}"].mean())
        summary["results"][m] = {"smart": a, "plain": b, "delta": a - b}

    summary["signature_union"] = {
        "smart": float(out["smart_signature_union"].mean()),
        "plain": float(out["plain_signature_union"].mean()),
        "delta": float(out["smart_signature_union"].mean() - out["plain_signature_union"].mean()),
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 50-TICKET STRUCTURAL BLEND v0 ===")
    print(f"history_draws={len(df)} usable_targets={len(out)}")
    print(f"pool={args.pool} A_good_keep={args.keep} plain_reps={args.plain_reps}")
    print(
        "signature_union: "
        f"smart={summary['signature_union']['smart']:.4f} "
        f"plain={summary['signature_union']['plain']:.4f} "
        f"delta={summary['signature_union']['delta']:+.4f}"
    )
    for m, x in summary["results"].items():
        print(f"{m}: smart={x['smart']:.6f} plain={x['plain']:.6f} delta={x['delta']:+.6f}")
    print("NOTE: Both sides use the same A-good 10-ticket block pool. Only the way five blocks are blended differs.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
