from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
OUT_JSON = Path("results/loto7_closure_structure_v0_summary.json")
OUT_CSV = Path("results/loto7_closure_structure_v0_holdout.csv")


def actual_numbers(row):
    return set(int(row[f"n{i}"]) for i in range(1, 8))


def core18(history):
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return [n for n, _ in sorted(ranks.items(), key=lambda kv: (kv[1], kv[0]))[:18]]


def random_bundle(candidates, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < 10:
        t = tuple(sorted(rng.sample(candidates, 7)))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def bundle_features(bundle, candidates):
    degree = Counter(n for t in bundle for n in t)
    degs = [degree[n] for n in candidates]
    mean_deg = sum(degs) / len(degs)
    degree_variance = sum((x - mean_deg) ** 2 for x in degs) / len(degs)

    pair = Counter()
    triple = Counter()
    overlaps = []
    for t in bundle:
        pair.update(itertools.combinations(t, 2))
        triple.update(itertools.combinations(t, 3))
    for a, b in itertools.combinations(bundle, 2):
        overlaps.append(len(set(a) & set(b)))

    return {
        "degree_variance": degree_variance,
        "pair_reuse_sq": sum(v * v for v in pair.values()),
        "triple_reuse_sq": sum(v * v for v in triple.values()),
        "ticket_overlap_mean": sum(overlaps) / len(overlaps),
        "ticket_overlap_max": max(overlaps),
    }


def max_hits(bundle, actual):
    return max(len(set(t) & actual) for t in bundle)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def effect(good, bad):
    allv = good + bad
    s = std(allv)
    if s == 0 or not good or not bad:
        return 0.0
    return ((sum(good) / len(good)) - (sum(bad) / len(bad))) / s


def discover(df, window, discovery_targets, bundles_per_round):
    feature_names = [
        "degree_variance",
        "pair_reuse_sq",
        "triple_reuse_sq",
        "ticket_overlap_mean",
        "ticket_overlap_max",
    ]

    records = []
    for i in range(window, window + discovery_targets):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = core18(history)
        for rep in range(bundles_per_round):
            bundle = random_bundle(cand, rnd * 1000003 + rep * 7919 + 77)
            feats = bundle_features(bundle, cand)
            h = max_hits(bundle, actual)
            records.append((i, h, feats))

    # 4 chronological blocks; compare bundles that reached 4+ against those that did not.
    block_size = max(1, discovery_targets // 4)
    effects = {f: [] for f in feature_names}
    counts = []
    for b in range(4):
        lo = window + b * block_size
        hi = window + (b + 1) * block_size if b < 3 else window + discovery_targets
        block = [r for r in records if lo <= r[0] < hi]
        good_n = sum(r[1] >= 4 for r in block)
        bad_n = len(block) - good_n
        counts.append({"block": b + 1, "good_4plus": good_n, "bad_lt4": bad_n})
        for f in feature_names:
            good = [r[2][f] for r in block if r[1] >= 4]
            bad = [r[2][f] for r in block if r[1] < 4]
            effects[f].append(effect(good, bad))

    candidates = []
    for f, es in effects.items():
        nonzero = [e for e in es if abs(e) > 1e-12]
        stable = bool(nonzero) and (all(e > 0 for e in nonzero) or all(e < 0 for e in nonzero)) and len(nonzero) == 4
        direction = 1 if sum(es) >= 0 else -1
        candidates.append({
            "feature": f,
            "effects": es,
            "stable_direction": stable,
            "direction": direction,
            "min_abs_effect": min(abs(e) for e in es),
            "mean_abs_effect": sum(abs(e) for e in es) / len(es),
        })

    stable = [x for x in candidates if x["stable_direction"]]
    if stable:
        chosen = max(stable, key=lambda x: (x["min_abs_effect"], x["mean_abs_effect"]))
    else:
        chosen = max(candidates, key=lambda x: x["mean_abs_effect"])

    return chosen, candidates, counts


def holdout(df, window, start_i, bundles_per_round, chosen):
    f = chosen["feature"]
    direction = chosen["direction"]
    rows = []

    for i in range(start_i, len(df)):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = core18(history)

        bundles = []
        for rep in range(bundles_per_round):
            bundle = random_bundle(cand, rnd * 1000003 + rep * 7919 + 991)
            feats = bundle_features(bundle, cand)
            h = max_hits(bundle, actual)
            bundles.append((feats[f], h))

        bundles.sort(key=lambda x: x[0], reverse=(direction > 0))
        q = max(1, len(bundles) // 4)
        closure = bundles[:q]
        ablated = bundles[-q:]

        def rate(group, k):
            return sum(h >= k for _, h in group) / len(group)

        rows.append({
            "round": rnd,
            "date": row.get("date", ""),
            "core18_capture": len(set(cand) & actual),
            "feature": f,
            "closure_feature_mean": sum(x for x, _ in closure) / len(closure),
            "ablated_feature_mean": sum(x for x, _ in ablated) / len(ablated),
            "closure_4plus": rate(closure, 4),
            "ablated_4plus": rate(ablated, 4),
            "closure_5plus": rate(closure, 5),
            "ablated_5plus": rate(ablated, 5),
            "closure_6plus": rate(closure, 6),
            "ablated_6plus": rate(ablated, 6),
            "closure_7": rate(closure, 7),
            "ablated_7": rate(ablated, 7),
            "closure_mean_max": sum(h for _, h in closure) / len(closure),
            "ablated_mean_max": sum(h for _, h in ablated) / len(ablated),
        })

    return pd.DataFrame(rows)


def summarize(hold, chosen, all_candidates, discovery_counts):
    metrics = ["4plus", "5plus", "6plus", "7", "mean_max"]
    result = {
        "probe": "LOTO7 Closure Structure v0",
        "scope": "Discover relation-level structure on an early period, then test enriched vs depleted structure on a later holdout. No target data is used to choose holdout bundles.",
        "chosen_structure": chosen,
        "discovery_candidates": all_candidates,
        "discovery_block_counts": discovery_counts,
        "holdout_rounds": int(len(hold)),
        "holdout": {},
    }
    for m in metrics:
        c = float(hold[f"closure_{m}"].mean())
        a = float(hold[f"ablated_{m}"].mean())
        result["holdout"][m] = {"closure": c, "ablated": a, "delta": c - a}

    result["by_core18_capture"] = {}
    for cap, g in hold.groupby("core18_capture"):
        block = {"n": int(len(g))}
        for m in ["4plus", "5plus", "6plus"]:
            c = float(g[f"closure_{m}"].mean())
            a = float(g[f"ablated_{m}"].mean())
            block[m] = {"closure": c, "ablated": a, "delta": c-a}
        result["by_core18_capture"][str(int(cap))] = block
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--discovery-targets", type=int, default=400)
    p.add_argument("--discovery-bundles", type=int, default=160)
    p.add_argument("--holdout-bundles", type=int, default=400)
    args = p.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    max_targets = len(df) - args.window
    if args.discovery_targets >= max_targets:
        raise ValueError("discovery period leaves no holdout")

    chosen, candidates, counts = discover(
        df, args.window, args.discovery_targets, args.discovery_bundles
    )
    hold = holdout(
        df,
        args.window,
        args.window + args.discovery_targets,
        args.holdout_bundles,
        chosen,
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    hold.to_csv(OUT_CSV, index=False)
    summary = summarize(hold, chosen, candidates, counts)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 CLOSURE STRUCTURE v0 ===")
    print("Discovery: compare 4+ bundles vs <4 bundles using relation-level bundle features.")
    print("Holdout: enrich/deplete only the chosen structural feature; no target leakage.")
    print(f"chosen={chosen['feature']} direction={'high' if chosen['direction']>0 else 'low'}")
    print("effects_by_discovery_block=" + ",".join(f"{x:+.4f}" for x in chosen["effects"]))
    print(f"stable_direction={chosen['stable_direction']}")
    print(f"holdout_rounds={summary['holdout_rounds']}")
    for m, x in summary["holdout"].items():
        print(f"{m}: closure={x['closure']:.6f} ablated={x['ablated']:.6f} delta={x['delta']:+.6f}")
    print("--- by CORE18 capture ---")
    for cap in sorted(summary["by_core18_capture"], key=int):
        b = summary["by_core18_capture"][cap]
        print(
            f"capture={cap} n={b['n']} "
            f"5+ {b['5plus']['closure']:.4f} vs {b['5plus']['ablated']:.4f}; "
            f"6+ {b['6plus']['closure']:.4f} vs {b['6plus']['ablated']:.4f}"
        )
    print("NOTE: this tests a structural relation, not specific lottery numbers.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
