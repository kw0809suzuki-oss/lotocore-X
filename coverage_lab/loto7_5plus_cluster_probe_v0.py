#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
from coverage_lab.loto7_portfolio20_v0 import (
    A_TICKETS,
    MODEL_WINDOW,
    POOL_K,
    main_numbers,
    portfolio_bundle,
    unique_structured_bundle,
)


def parse_hits(ticket, actual):
    return tuple(sorted(set(ticket) & actual))


def cluster_components(hit_sets: list[tuple[int, ...]], min_shared: int = 4) -> list[list[int]]:
    n = len(hit_sets)
    seen = set()
    comps = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            sx = set(hit_sets[x])
            for j in range(n):
                if j in seen:
                    continue
                if len(sx & set(hit_sets[j])) >= min_shared:
                    seen.add(j)
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def jaccard(a, b):
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def summarize_high_hits(tickets, actual):
    records = []
    for i, t in enumerate(tickets, 1):
        hs = parse_hits(t, actual)
        if len(hs) >= 5:
            records.append({
                "ticket_id": i,
                "ticket": list(t),
                "hit_count": len(hs),
                "hit_set": list(hs),
            })

    hit_sets = [tuple(r["hit_set"]) for r in records]
    if not hit_sets:
        return {
            "ticket_count_5plus": 0,
            "cluster_count_shared4": 0,
            "unique_hit_structures": 0,
            "largest_cluster": 0,
            "mean_pair_shared_hits": None,
            "mean_pair_jaccard": None,
            "records": [],
            "clusters": [],
        }

    comps = cluster_components(hit_sets, min_shared=4)
    pair_shared = []
    pair_j = []
    for a, b in itertools.combinations(hit_sets, 2):
        pair_shared.append(len(set(a) & set(b)))
        pair_j.append(jaccard(a, b))

    return {
        "ticket_count_5plus": len(records),
        "cluster_count_shared4": len(comps),
        "unique_hit_structures": len(set(hit_sets)),
        "largest_cluster": max(len(c) for c in comps),
        "mean_pair_shared_hits": (
            sum(pair_shared) / len(pair_shared) if pair_shared else None
        ),
        "mean_pair_jaccard": (
            sum(pair_j) / len(pair_j) if pair_j else None
        ),
        "records": records,
        "clusters": comps,
    }


def run(data_path: Path, model_window: int = MODEL_WINDOW):
    df = pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    rows = []
    detail = []

    for idx in range(model_window, len(df)):
        history = df.iloc[idx-model_window:idx]
        target = df.iloc[idx]
        rnd = int(target["round"])
        actual = main_numbers(target)

        snap = lotocore.score_snapshot(history)
        ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
        scores = {int(k): float(v) for k, v in snap["scores"].items()}
        ranked = sorted(range(1, 38), key=lambda n: (ranks[n], n))
        pool = ranked[:POOL_K]

        seed = rnd * 100_003 + 20261002
        a, _ = unique_structured_bundle(pool, A_TICKETS, seed + 1)
        b, _ = portfolio_bundle(pool, scores, seed + 2)

        for strategy, tickets in (("structured20", a), ("portfolio20", b)):
            s = summarize_high_hits(tickets, actual)
            if s["ticket_count_5plus"] == 0:
                continue
            rows.append({
                "round": rnd,
                "strategy": strategy,
                "ticket_count_5plus": s["ticket_count_5plus"],
                "cluster_count_shared4": s["cluster_count_shared4"],
                "unique_hit_structures": s["unique_hit_structures"],
                "largest_cluster": s["largest_cluster"],
                "mean_pair_shared_hits": s["mean_pair_shared_hits"],
                "mean_pair_jaccard": s["mean_pair_jaccard"],
            })
            detail.append({
                "round": rnd,
                "strategy": strategy,
                "actual": sorted(actual),
                **s,
            })

    res = pd.DataFrame(rows)
    summary = {}
    for strategy in ("structured20", "portfolio20"):
        g = res[res.strategy == strategy]
        summary[strategy] = {
            "rounds_with_5plus": int(len(g)),
            "total_5plus_tickets": int(g.ticket_count_5plus.sum()) if len(g) else 0,
            "total_shared4_clusters": int(g.cluster_count_shared4.sum()) if len(g) else 0,
            "total_unique_hit_structures": int(g.unique_hit_structures.sum()) if len(g) else 0,
            "multi_5plus_rounds": int((g.ticket_count_5plus >= 2).sum()) if len(g) else 0,
            "mean_tickets_per_5plus_round": float(g.ticket_count_5plus.mean()) if len(g) else None,
            "mean_clusters_per_5plus_round": float(g.cluster_count_shared4.mean()) if len(g) else None,
            "mean_unique_structures_per_5plus_round": float(g.unique_hit_structures.mean()) if len(g) else None,
            "mean_largest_cluster": float(g.largest_cluster.mean()) if len(g) else None,
            "mean_pair_shared_hits_multi_only": float(
                g[g.ticket_count_5plus >= 2].mean_pair_shared_hits.mean()
            ) if (g.ticket_count_5plus >= 2).any() else None,
            "mean_pair_jaccard_multi_only": float(
                g[g.ticket_count_5plus >= 2].mean_pair_jaccard.mean()
            ) if (g.ticket_count_5plus >= 2).any() else None,
        }

    multi = res[res.ticket_count_5plus >= 2].copy()
    return res, {
        "experiment": "loto7_5plus_cluster_probe_v0",
        "cluster_rule": "5+ tickets are connected when their winning-number hit sets share >=4 winning numbers; connected components are clusters.",
        "summary": summary,
        "multi_5plus_rows": multi.to_dict(orient="records"),
        "detail": detail,
        "boundary": [
            "This replays the frozen Structured20 and Portfolio20 generators; it does not change ticket generation.",
            "Only rounds with at least one 5+ ticket are analyzed.",
            "Cluster count is an observation of redundancy among realized high-hit tickets, not evidence of future win probability.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("coverage_lab/fixtures/loto7_1_696.csv"))
    p.add_argument("--out", type=Path, default=Path("results/loto7_5plus_cluster_probe_v0.csv"))
    p.add_argument("--json", type=Path, default=Path("results/loto7_5plus_cluster_probe_v0.json"))
    a = p.parse_args()

    res, payload = run(a.data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    a.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== LOTO7 5+ CLUSTER PROBE v0 ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    print("MULTI", json.dumps(payload["multi_5plus_rows"], ensure_ascii=False, sort_keys=True))
    print("BOUNDARY", " | ".join(payload["boundary"]))
    print(f"saved -> {a.out}")
    print(f"saved -> {a.json}")


if __name__ == "__main__":
    main()
