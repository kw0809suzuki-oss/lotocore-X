from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import lotocore

DATA = Path("data/loto7.csv")
OUT_JSON = Path("results/loto7_bundle_capture_conversion_v0_summary.json")
OUT_CSV = Path("results/loto7_bundle_capture_conversion_v0_rounds.csv")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []

    for i in range(args.window, len(df)):
        history = df.iloc[i-args.window:i]
        row = df.iloc[i]
        actual = actual_numbers(row)
        candidates = core18(history)
        bundle = structured_bundle(candidates)

        union_numbers = set().union(*(set(t) for t in bundle))
        bundle_capture = len(union_numbers & actual)
        core18_capture = len(set(candidates) & actual)
        hits = [len(set(t) & actual) for t in bundle]
        max_hits = max(hits)

        rows.append({
            "round": int(row["round"]),
            "core18_capture": core18_capture,
            "bundle_union_size": len(union_numbers),
            "bundle_capture": bundle_capture,
            "max_hits": max_hits,
            "conversion_ratio": (max_hits / bundle_capture) if bundle_capture else None,
            "reach3": int(max_hits >= 3),
            "reach4": int(max_hits >= 4),
            "reach5": int(max_hits >= 5),
            "reach6": int(max_hits >= 6),
            "reach7": int(max_hits >= 7),
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    buckets = {}
    for capture, g in out.groupby("bundle_capture"):
        buckets[str(int(capture))] = {
            "rounds": int(len(g)),
            "share": float(len(g) / len(out)),
            "mean_max_hits": float(g["max_hits"].mean()),
            "mean_conversion_ratio": float(g["conversion_ratio"].dropna().mean()) if capture else None,
            "reach3": float(g["reach3"].mean()),
            "reach4": float(g["reach4"].mean()),
            "reach5": float(g["reach5"].mean()),
            "reach6": float(g["reach6"].mean()),
            "reach7": float(g["reach7"].mean()),
        }

    summary = {
        "probe": "LOTO7 10-ticket bundle capture -> conversion v0",
        "history_draws": int(len(df)),
        "usable_targets": int(len(out)),
        "window": args.window,
        "bundle_union_size_unique": sorted(int(x) for x in out["bundle_union_size"].unique()),
        "bundle_capture_equals_core18_capture_all_rounds": bool((out["bundle_capture"] == out["core18_capture"]).all()),
        "overall": {
            "mean_bundle_capture": float(out["bundle_capture"].mean()),
            "mean_max_hits": float(out["max_hits"].mean()),
            "reach3": float(out["reach3"].mean()),
            "reach4": float(out["reach4"].mean()),
            "reach5": float(out["reach5"].mean()),
            "reach6": float(out["reach6"].mean()),
            "reach7": float(out["reach7"].mean()),
        },
        "by_bundle_capture": buckets,
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 BUNDLE CAPTURE -> CONVERSION v0 ===")
    print(f"history_draws={len(df)} usable_targets={len(out)}")
    print(f"union_sizes={summary['bundle_union_size_unique']}")
    print(f"bundle_capture_equals_core18_capture={summary['bundle_capture_equals_core18_capture_all_rounds']}")
    print(
        f"overall mean_capture={summary['overall']['mean_bundle_capture']:.6f} "
        f"mean_max={summary['overall']['mean_max_hits']:.6f} "
        f"3+={summary['overall']['reach3']:.6f} "
        f"4+={summary['overall']['reach4']:.6f} "
        f"5+={summary['overall']['reach5']:.6f}"
    )
    for k in sorted(buckets, key=int):
        x = buckets[k]
        conv = "NA" if x["mean_conversion_ratio"] is None else f"{x['mean_conversion_ratio']:.6f}"
        print(
            f"capture={k} n={x['rounds']} share={x['share']:.6f} "
            f"mean_max={x['mean_max_hits']:.6f} conversion={conv} "
            f"3+={x['reach3']:.6f} 4+={x['reach4']:.6f} "
            f"5+={x['reach5']:.6f} 6+={x['reach6']:.6f} 7={x['reach7']:.6f}"
        )
    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
