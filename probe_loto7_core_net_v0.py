from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CORE18 = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 25, 28, 30, 31, 34, 35]
OUT = Path("results/loto7_core_net_v0.json")


def ticket_mask(ticket):
    m = 0
    for i, n in enumerate(CORE18):
        if n in ticket:
            m |= 1 << i
    return m


def random_bundle(seed):
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < 10:
        t = tuple(sorted(rng.sample(CORE18, 7)))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def structured_bundle():
    # 70 total slots over 18 candidates => 16 numbers appear 4 times, 2 appear 3 times.
    target = {n: (4 if i < 16 else 3) for i, n in enumerate(CORE18)}
    remaining = target.copy()
    pair_count = defaultdict(int)
    out = []

    for slot in range(10):
        tickets_left = 10 - slot
        forced = [n for n in CORE18 if remaining[n] == tickets_left]
        chosen = list(forced)
        if len(chosen) > 7:
            raise RuntimeError("infeasible degree schedule")

        while len(chosen) < 7:
            candidates = [n for n in CORE18 if remaining[n] > 0 and n not in chosen]
            def key(n):
                pair_penalty = sum(pair_count[tuple(sorted((n, x)))] for x in chosen)
                # Prefer numbers with more remaining degree, then lower pair reuse, then stable order.
                return (pair_penalty, -remaining[n], CORE18.index(n))
            chosen.append(min(candidates, key=key))

        chosen = sorted(chosen)
        out.append(tuple(chosen))
        for n in chosen:
            remaining[n] -= 1
        for a, b in itertools.combinations(chosen, 2):
            pair_count[(a, b)] += 1

    if any(remaining.values()):
        raise RuntimeError(f"unspent degrees: {remaining}")
    return out


def pair_score(bundle):
    c = Counter()
    for t in bundle:
        c.update(itertools.combinations(t, 2))
    return sum(v * v for v in c.values())


def degree_stats(bundle):
    c = Counter(n for t in bundle for n in t)
    vals = [c[n] for n in CORE18]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "counts": {str(n): c[n] for n in CORE18},
        "min": min(vals),
        "max": max(vals),
        "variance": var,
    }


def exact_metrics(bundle):
    masks = [ticket_mask(t) for t in bundle]
    total_universe = math.comb(37, 7)
    reach = Counter()
    sum_max = 0
    by_capture = {}

    for k in range(0, 8):
        if k > 18 or 7 - k > 19:
            continue
        outside_weight = math.comb(19, 7 - k)
        count = 0
        sum_k = 0
        rk = Counter()
        hist = Counter()
        for combo in itertools.combinations(range(18), k):
            cm = 0
            for idx in combo:
                cm |= 1 << idx
            best = max((cm & tm).bit_count() for tm in masks)
            hist[best] += 1
            sum_k += best
            count += 1
            for h in (3, 4, 5, 6, 7):
                if best >= h:
                    rk[h] += 1

            weighted = outside_weight
            sum_max += best * weighted
            for h in (3, 4, 5, 6, 7):
                if best >= h:
                    reach[h] += weighted

        if count:
            by_capture[str(k)] = {
                "candidate_subsets": count,
                "mean_max": sum_k / count,
                "reach3": rk[3] / count,
                "reach4": rk[4] / count,
                "reach5": rk[5] / count,
                "reach6": rk[6] / count,
                "reach7": rk[7] / count,
                "max_hit_hist": {str(h): hist[h] for h in sorted(hist)},
            }

    return {
        "expected_max": sum_max / total_universe,
        "reach3": reach[3] / total_universe,
        "reach4": reach[4] / total_universe,
        "reach5": reach[5] / total_universe,
        "reach6": reach[6] / total_universe,
        "reach7": reach[7] / total_universe,
        "by_capture": by_capture,
    }


def summarize_random(reps):
    rows = []
    for r in range(reps):
        b = random_bundle(20261001 + r * 7919)
        m = exact_metrics(b)
        rows.append({
            "expected_max": m["expected_max"],
            "reach3": m["reach3"],
            "reach4": m["reach4"],
            "reach5": m["reach5"],
            "reach6": m["reach6"],
            "reach7": m["reach7"],
            "pair_score": pair_score(b),
            "degree_variance": degree_stats(b)["variance"],
            "by_capture": m["by_capture"],
        })

    keys = ["expected_max", "reach3", "reach4", "reach5", "reach6", "reach7", "pair_score", "degree_variance"]
    summary = {}
    for key in keys:
        xs = sorted(float(x[key]) for x in rows)
        summary[key] = {
            "mean": sum(xs) / len(xs),
            "min": xs[0],
            "max": xs[-1],
            "median": xs[len(xs)//2],
        }

    cap_summary = {}
    for k in range(8):
        kk = str(k)
        vals = [x["by_capture"][kk] for x in rows if kk in x["by_capture"]]
        if not vals:
            continue
        cap_summary[kk] = {}
        for key in ["mean_max", "reach3", "reach4", "reach5", "reach6", "reach7"]:
            xs = sorted(float(v[key]) for v in vals)
            cap_summary[kk][key] = {
                "mean": sum(xs) / len(xs),
                "min": xs[0],
                "max": xs[-1],
                "median": xs[len(xs)//2],
            }
    summary["by_capture"] = cap_summary
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random-reps", type=int, default=200)
    args = p.parse_args()

    structured = structured_bundle()
    sm = exact_metrics(structured)
    random_summary = summarize_random(args.random_reps)

    result = {
        "probe": "LOTO7 CORE Net v0",
        "scope": "Candidate set frozen; compare ticket allocation only. No prediction claim.",
        "core18": CORE18,
        "tickets": 10,
        "ticket_size": 7,
        "structured": {
            "bundle": structured,
            "pair_score": pair_score(structured),
            "degree_stats": degree_stats(structured),
            **sm,
        },
        "random_baseline": {
            "reps": args.random_reps,
            **random_summary,
        },
    }

    # Convenient deltas against the random mean.
    result["delta_vs_random_mean"] = {
        k: result["structured"][k] - result["random_baseline"][k]["mean"]
        for k in ["expected_max", "reach3", "reach4", "reach5", "reach6", "reach7"]
    }
    result["delta_vs_random_mean"]["pair_score"] = (
        result["structured"]["pair_score"] - result["random_baseline"]["pair_score"]["mean"]
    )
    result["delta_vs_random_mean"]["degree_variance"] = (
        result["structured"]["degree_stats"]["variance"] - result["random_baseline"]["degree_variance"]["mean"]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 CORE NET v0 ===")
    print("scope: fixed CORE18; allocator-only comparison; exact outcome weighting")
    print(f"CORE18={CORE18}")
    print(f"structured_pair_score={result['structured']['pair_score']}")
    print(f"random_pair_score_mean={result['random_baseline']['pair_score']['mean']:.3f}")
    print(f"structured_degree_variance={result['structured']['degree_stats']['variance']:.6f}")
    print(f"random_degree_variance_mean={result['random_baseline']['degree_variance']['mean']:.6f}")
    for k in ["expected_max", "reach3", "reach4", "reach5", "reach6", "reach7"]:
        s = result["structured"][k]
        r = result["random_baseline"][k]["mean"]
        print(f"{k}: structured={s:.8f} random_mean={r:.8f} delta={s-r:+.8f}")
    for cap in ("3","4","5","6","7"):
        s = result["structured"]["by_capture"][cap]
        r = result["random_baseline"]["by_capture"][cap]
        print(
            f"capture={cap}: mean_max {s['mean_max']:.6f} vs {r['mean_max']['mean']:.6f}; "
            f"reach4 {s['reach4']:.6f} vs {r['reach4']['mean']:.6f}; "
            f"reach5 {s['reach5']:.6f} vs {r['reach5']['mean']:.6f}"
        )
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
