from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
OUT_CSV = Path("results/loto7_core_net_walkforward_v0.csv")
OUT_JSON = Path("results/loto7_core_net_walkforward_v0_summary.json")


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


def structured_bundle(candidates):
    # 10 tickets x 7 = 70 slots. Spread 70 uses across 18 candidates as evenly as possible.
    base = 70 // len(candidates)
    extra = 70 % len(candidates)
    target = {n: base + (1 if i < extra else 0) for i, n in enumerate(candidates)}
    rem = target.copy()
    pair_count = defaultdict(int)
    out = []

    for slot in range(10):
        tickets_left = 10 - slot
        forced = [n for n in candidates if rem[n] == tickets_left]
        chosen = list(forced)
        if len(chosen) > 7:
            raise RuntimeError("infeasible degree schedule")

        while len(chosen) < 7:
            pool = [n for n in candidates if rem[n] > 0 and n not in chosen]
            def key(n):
                pair_penalty = sum(pair_count[tuple(sorted((n, x)))] for x in chosen)
                return (pair_penalty, -rem[n], candidates.index(n))
            chosen.append(min(pool, key=key))

        chosen = sorted(chosen)
        out.append(tuple(chosen))
        for n in chosen:
            rem[n] -= 1
        for a, b in itertools.combinations(chosen, 2):
            pair_count[(a, b)] += 1

    if any(rem.values()):
        raise RuntimeError(f"unspent degrees: {rem}")
    return out


def max_hits(bundle, actual):
    return max(len(set(t) & actual) for t in bundle)


def pair_score(bundle):
    c = Counter()
    for t in bundle:
        c.update(itertools.combinations(t, 2))
    return sum(v * v for v in c.values())


def degree_variance(bundle, candidates):
    c = Counter(n for t in bundle for n in t)
    xs = [c[n] for n in candidates]
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def run(window=100, random_reps=200):
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(window, len(df)):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = core18(history)

        structured = structured_bundle(cand)
        sh = max_hits(structured, actual)
        capture = len(set(cand) & actual)

        random_hits = []
        random_pairs = []
        random_degvar = []
        for rep in range(random_reps):
            seed = rnd * 1000003 + 20261001 + rep * 7919
            rb = random_bundle(cand, seed)
            random_hits.append(max_hits(rb, actual))
            random_pairs.append(pair_score(rb))
            random_degvar.append(degree_variance(rb, cand))

        rows.append({
            "round": rnd,
            "date": row.get("date", ""),
            "core18": " ".join(map(str, cand)),
            "candidate_capture": capture,
            "structured_max_hits": sh,
            "random_mean_max_hits": sum(random_hits)/len(random_hits),
            "random_reach3": sum(h >= 3 for h in random_hits)/len(random_hits),
            "random_reach4": sum(h >= 4 for h in random_hits)/len(random_hits),
            "random_reach5": sum(h >= 5 for h in random_hits)/len(random_hits),
            "structured_reach3": int(sh >= 3),
            "structured_reach4": int(sh >= 4),
            "structured_reach5": int(sh >= 5),
            "structured_pair_score": pair_score(structured),
            "random_pair_score_mean": sum(random_pairs)/len(random_pairs),
            "structured_degree_variance": degree_variance(structured, cand),
            "random_degree_variance_mean": sum(random_degvar)/len(random_degvar),
        })

    return pd.DataFrame(rows)


def summarize(df):
    def pct(series):
        return float(series.mean())

    out = {
        "probe": "LOTO7 CORE Net Walk-forward v0",
        "scope": "Historical walk-forward. CORE18 frozen per target from prior 100 draws only. Allocator-only comparison.",
        "target_rounds": int(len(df)),
        "structured": {
            "mean_max_hits": float(df["structured_max_hits"].mean()),
            "reach3": pct(df["structured_reach3"]),
            "reach4": pct(df["structured_reach4"]),
            "reach5": pct(df["structured_reach5"]),
            "pair_score_mean": float(df["structured_pair_score"].mean()),
            "degree_variance_mean": float(df["structured_degree_variance"].mean()),
        },
        "random_baseline": {
            "mean_max_hits": float(df["random_mean_max_hits"].mean()),
            "reach3": float(df["random_reach3"].mean()),
            "reach4": float(df["random_reach4"].mean()),
            "reach5": float(df["random_reach5"].mean()),
            "pair_score_mean": float(df["random_pair_score_mean"].mean()),
            "degree_variance_mean": float(df["random_degree_variance_mean"].mean()),
        },
        "by_capture": {},
    }

    for cap, g in df.groupby("candidate_capture"):
        out["by_capture"][str(int(cap))] = {
            "n": int(len(g)),
            "structured_mean_max": float(g["structured_max_hits"].mean()),
            "random_mean_max": float(g["random_mean_max_hits"].mean()),
            "structured_reach3": float(g["structured_reach3"].mean()),
            "random_reach3": float(g["random_reach3"].mean()),
            "structured_reach4": float(g["structured_reach4"].mean()),
            "random_reach4": float(g["random_reach4"].mean()),
            "structured_reach5": float(g["structured_reach5"].mean()),
            "random_reach5": float(g["random_reach5"].mean()),
        }

    out["delta_vs_random"] = {
        k: out["structured"][k] - out["random_baseline"][k]
        for k in ["mean_max_hits", "reach3", "reach4", "reach5", "pair_score_mean", "degree_variance_mean"]
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--random-reps", type=int, default=200)
    args = p.parse_args()

    df = run(args.window, args.random_reps)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = summarize(df)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 CORE NET WALK-FORWARD v0 ===")
    print(f"targets={summary['target_rounds']} window={args.window} random_reps={args.random_reps}")
    for side in ("structured", "random_baseline"):
        s = summary[side]
        print(
            f"{side}: mean_max={s['mean_max_hits']:.6f} "
            f"3+={s['reach3']:.6f} 4+={s['reach4']:.6f} 5+={s['reach5']:.6f} "
            f"pair={s['pair_score_mean']:.3f} degvar={s['degree_variance_mean']:.6f}"
        )
    d = summary["delta_vs_random"]
    print(
        f"delta: mean_max={d['mean_max_hits']:+.6f} "
        f"3+={d['reach3']:+.6f} 4+={d['reach4']:+.6f} 5+={d['reach5']:+.6f}"
    )
    for cap in sorted(summary["by_capture"], key=int):
        x = summary["by_capture"][cap]
        print(
            f"capture={cap} n={x['n']} "
            f"mean_max {x['structured_mean_max']:.4f} vs {x['random_mean_max']:.4f}; "
            f"4+ {x['structured_reach4']:.4f} vs {x['random_reach4']:.4f}; "
            f"5+ {x['structured_reach5']:.4f} vs {x['random_reach5']:.4f}"
        )
    print("NOTE: historical walk-forward observation; not evidence of future lottery advantage by itself.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
