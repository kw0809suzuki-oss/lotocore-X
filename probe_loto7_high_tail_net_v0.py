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
OUT_CSV = Path("results/loto7_high_tail_net_v0.csv")
OUT_JSON = Path("results/loto7_high_tail_net_v0_summary.json")


def actual_numbers(row):
    return set(int(row[f"n{i}"]) for i in range(1, 8))


def ranked_core18(history):
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
    # Frozen 3+/4+ reference net from the previous experiment.
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
    return out


def anchor_bundle(candidates, anchor_count):
    # High-tail allocator: keep the top-ranked CORE numbers together and rotate the rest.
    anchors = candidates[:anchor_count]
    rest = candidates[anchor_count:]
    need = 7 - anchor_count
    if need <= 0:
        raise ValueError("anchor_count must be < 7")

    # Greedy rotation minimizes rest-number and pair reuse while anchors stay fixed.
    use = Counter()
    pair = Counter()
    out = []
    for _ in range(10):
        chosen = list(anchors)
        while len(chosen) < 7:
            pool = [n for n in rest if n not in chosen]
            def key(n):
                p = sum(pair[tuple(sorted((n, x)))] for x in chosen if x in rest)
                return (use[n], p, candidates.index(n))
            pick = min(pool, key=key)
            chosen.append(pick)
        t = tuple(sorted(chosen))
        out.append(t)
        for n in chosen:
            use[n] += 1
        for a, b in itertools.combinations([n for n in chosen if n in rest], 2):
            pair[tuple(sorted((a, b)))] += 1
    return out


def max_hits(bundle, actual):
    return max(len(set(t) & actual) for t in bundle)


def metrics_for_bundle(bundle, actual):
    h = max_hits(bundle, actual)
    return {
        "max_hits": h,
        "r5": int(h >= 5),
        "r6": int(h >= 6),
        "r7": int(h >= 7),
    }


def run(window=100, random_reps=200):
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(window, len(df)):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        cand = ranked_core18(history)
        capture = len(set(cand) & actual)

        packs = {
            "structured": structured_bundle(cand),
            "anchor2": anchor_bundle(cand, 2),
            "anchor3": anchor_bundle(cand, 3),
            "anchor4": anchor_bundle(cand, 4),
        }
        rec = {
            "round": rnd,
            "date": row.get("date", ""),
            "candidate_capture": capture,
            "top7_capture": len(set(cand[:7]) & actual),
        }

        for name, bundle in packs.items():
            m = metrics_for_bundle(bundle, actual)
            for k, v in m.items():
                rec[f"{name}_{k}"] = v

        rh = []
        for rep in range(random_reps):
            rb = random_bundle(cand, rnd * 1000003 + 20261001 + rep * 7919)
            rh.append(max_hits(rb, actual))
        rec["random_mean_max_hits"] = sum(rh) / len(rh)
        rec["random_r5"] = sum(h >= 5 for h in rh) / len(rh)
        rec["random_r6"] = sum(h >= 6 for h in rh) / len(rh)
        rec["random_r7"] = sum(h >= 7 for h in rh) / len(rh)
        rows.append(rec)

    return pd.DataFrame(rows)


def summarize(df):
    models = ["structured", "anchor2", "anchor3", "anchor4"]
    out = {
        "probe": "LOTO7 High-Tail Net v0",
        "scope": "Separate 5+/6+/7+ experiment. Previous 3+/4+ Structured Net remains frozen.",
        "target_rounds": int(len(df)),
        "models": {},
        "random_baseline": {
            "mean_max_hits": float(df["random_mean_max_hits"].mean()),
            "reach5": float(df["random_r5"].mean()),
            "reach6": float(df["random_r6"].mean()),
            "reach7": float(df["random_r7"].mean()),
        },
        "by_core18_capture": {},
        "by_top7_capture": {},
    }
    for name in models:
        out["models"][name] = {
            "mean_max_hits": float(df[f"{name}_max_hits"].mean()),
            "reach5": float(df[f"{name}_r5"].mean()),
            "reach6": float(df[f"{name}_r6"].mean()),
            "reach7": float(df[f"{name}_r7"].mean()),
        }

    for col, dest in [("candidate_capture", "by_core18_capture"), ("top7_capture", "by_top7_capture")]:
        for cap, g in df.groupby(col):
            block = {"n": int(len(g)), "random": {
                "reach5": float(g["random_r5"].mean()),
                "reach6": float(g["random_r6"].mean()),
                "reach7": float(g["random_r7"].mean()),
            }}
            for name in models:
                block[name] = {
                    "reach5": float(g[f"{name}_r5"].mean()),
                    "reach6": float(g[f"{name}_r6"].mean()),
                    "reach7": float(g[f"{name}_r7"].mean()),
                }
            out[dest][str(int(cap))] = block
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--random-reps", type=int, default=200)
    args = p.parse_args()

    df = run(args.window, args.random_reps)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    s = summarize(df)
    OUT_JSON.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 HIGH-TAIL NET v0 ===")
    print("scope: separate 5+/6+/7+ experiment; 3+/4+ Structured Net is frozen")
    print(f"targets={s['target_rounds']} window={args.window} random_reps={args.random_reps}")
    r = s["random_baseline"]
    print(f"random: mean_max={r['mean_max_hits']:.6f} 5+={r['reach5']:.6f} 6+={r['reach6']:.6f} 7={r['reach7']:.6f}")
    for name, m in s["models"].items():
        print(f"{name}: mean_max={m['mean_max_hits']:.6f} 5+={m['reach5']:.6f} 6+={m['reach6']:.6f} 7={m['reach7']:.6f}")
    print("--- conditioned on CORE18 capture ---")
    for cap in sorted(s["by_core18_capture"], key=int):
        b = s["by_core18_capture"][cap]
        print(f"capture={cap} n={b['n']} random5={b['random']['reach5']:.4f} "
              f"structured5={b['structured']['reach5']:.4f} a2={b['anchor2']['reach5']:.4f} "
              f"a3={b['anchor3']['reach5']:.4f} a4={b['anchor4']['reach5']:.4f}")
    print("--- conditioned on top7 capture ---")
    for cap in sorted(s["by_top7_capture"], key=int):
        b = s["by_top7_capture"][cap]
        print(f"top7_capture={cap} n={b['n']} random5={b['random']['reach5']:.4f} "
              f"structured5={b['structured']['reach5']:.4f} a2={b['anchor2']['reach5']:.4f} "
              f"a3={b['anchor3']['reach5']:.4f} a4={b['anchor4']['reach5']:.4f}")
    print("NOTE: observational walk-forward only. No future lottery advantage is implied.")
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
