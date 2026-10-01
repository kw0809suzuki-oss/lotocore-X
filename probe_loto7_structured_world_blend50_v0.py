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
OUT_JSON = Path("results/loto7_structured_world_blend50_v0_summary.json")
OUT_CSV = Path("results/loto7_structured_world_blend50_v0_rounds.csv")


def actual_numbers(row):
    return set(int(row[f"n{i}"]) for i in range(1, 8))


def core18(history):
    snap = lotocore.score_snapshot(history)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return [n for n, _ in sorted(ranks.items(), key=lambda kv: (kv[1], kv[0]))[:18]]


def structured_bundle(candidates, seed):
    """
    Same Structured Net principle used in the original 10-ticket experiment:
    - 70 slots spread as evenly as possible over 18 candidates
    - pair reuse minimized greedily
    The only freedom is seeded tie-breaking among equally good candidates.
    """
    rng = random.Random(seed)
    base = 70 // len(candidates)
    extra = 70 % len(candidates)

    order = list(candidates)
    # The degree schedule remains 16x4 + 2x3; which candidates get the two
    # lower-degree slots is randomized so multiple legal Structured worlds exist.
    low_degree = set(rng.sample(order, len(candidates) - extra))
    target = {n: (base if n in low_degree else base + 1) for n in candidates}

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
            scored = []
            for n in pool:
                pair_penalty = sum(pair_count[tuple(sorted((n, x)))] for x in chosen)
                scored.append((pair_penalty, -rem[n], n))

            best_pair = min(x[0] for x in scored)
            best_rem = min(x[1] for x in scored if x[0] == best_pair)
            tied = [n for p, r, n in scored if p == best_pair and r == best_rem]
            chosen.append(rng.choice(sorted(tied)))

        chosen = tuple(sorted(chosen))
        out.append(chosen)

        for n in chosen:
            rem[n] -= 1
        for a, b in itertools.combinations(chosen, 2):
            pair_count[tuple(sorted((a, b)))] += 1

    if any(rem.values()):
        raise RuntimeError(f"unspent degrees: {rem}")

    return tuple(out)


def degree_variance(bundle, candidates):
    c = Counter(n for t in bundle for n in t)
    xs = [c[n] for n in candidates]
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def pair_score(bundle):
    c = Counter()
    for t in bundle:
        c.update(itertools.combinations(t, 2))
    return sum(v * v for v in c.values())


def subset_coverage(bundle, k):
    s = set()
    for t in bundle:
        s.update(itertools.combinations(sorted(t), k))
    return s


def generate_structured_worlds(candidates, round_no, count):
    worlds = []
    seen = set()
    attempt = 0
    while len(worlds) < count:
        seed = round_no * 1000003 + 20261001 + attempt * 7919
        attempt += 1
        b = structured_bundle(candidates, seed)
        if b in seen:
            continue
        seen.add(b)
        worlds.append({
            "bundle": b,
            "degvar": degree_variance(b, candidates),
            "pair_score": pair_score(b),
            "triples": subset_coverage(b, 3),
            "quads": subset_coverage(b, 4),
            "quints": subset_coverage(b, 5),
        })
    return worlds


def choose_diversified(worlds, count=5):
    chosen = []
    used = set()
    u5, u4, u3 = set(), set(), set()

    for _ in range(count):
        best_i = None
        best_key = None

        for i, w in enumerate(worlds):
            if i in used:
                continue

            new5 = len(w["quints"] - u5)
            new4 = len(w["quads"] - u4)
            new3 = len(w["triples"] - u3)

            # No target information. Only diversify co-location possibilities
            # across otherwise-valid Structured 10-ticket worlds.
            key = (
                new5,
                new4,
                new3,
                -w["pair_score"],
                -w["degvar"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        used.add(best_i)
        w = worlds[best_i]
        chosen.append(w)
        u5.update(w["quints"])
        u4.update(w["quads"])
        u3.update(w["triples"])

    tickets = [t for w in chosen for t in w["bundle"]]
    return tickets, chosen


def choose_plain(worlds, rng, count=5):
    chosen = rng.sample(worlds, count)
    tickets = [t for w in chosen for t in w["bundle"]]
    return tickets, chosen


def score50(tickets, actual):
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


def union_size(worlds, key):
    u = set()
    for w in worlds:
        u.update(w[key])
    return len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--worlds", type=int, default=120)
    ap.add_argument("--plain-reps", type=int, default=200)
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

        worlds = generate_structured_worlds(candidates, rnd, args.worlds)

        smart_tickets, smart_worlds = choose_diversified(worlds, 5)
        smart = score50(smart_tickets, actual)

        rng = random.Random(rnd * 1000003 + 424242)
        plain_acc = {k: 0.0 for k in smart}
        plain_u5 = plain_u4 = plain_u3 = 0.0

        for _ in range(args.plain_reps):
            pt, pw = choose_plain(worlds, rng, 5)
            s = score50(pt, actual)
            for k in plain_acc:
                plain_acc[k] += s[k]
            plain_u5 += union_size(pw, "quints")
            plain_u4 += union_size(pw, "quads")
            plain_u3 += union_size(pw, "triples")

        for k in plain_acc:
            plain_acc[k] /= args.plain_reps
        plain_u5 /= args.plain_reps
        plain_u4 /= args.plain_reps
        plain_u3 /= args.plain_reps

        rows.append({
            "round": rnd,
            "capture": capture,
            "smart_quint_union": union_size(smart_worlds, "quints"),
            "plain_quint_union": plain_u5,
            "smart_quad_union": union_size(smart_worlds, "quads"),
            "plain_quad_union": plain_u4,
            "smart_triple_union": union_size(smart_worlds, "triples"),
            "plain_triple_union": plain_u3,
            **{f"smart_{k}": v for k, v in smart.items()},
            **{f"plain_{k}": v for k, v in plain_acc.items()},
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    def summarize(g):
        result = {}
        for m in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]:
            a = float(g[f"smart_{m}"].mean())
            b = float(g[f"plain_{m}"].mean())
            result[m] = {"smart": a, "plain": b, "delta": a - b}
        result["layout_union"] = {
            "quint_smart": float(g["smart_quint_union"].mean()),
            "quint_plain": float(g["plain_quint_union"].mean()),
            "quad_smart": float(g["smart_quad_union"].mean()),
            "quad_plain": float(g["plain_quad_union"].mean()),
            "triple_smart": float(g["smart_triple_union"].mean()),
            "triple_plain": float(g["plain_triple_union"].mean()),
        }
        return result

    summary = {
        "probe": "LOTO7 Structured World Blend50 v0",
        "scope": (
            "Same deterministic CORE18 proxy as the original walk-forward evidence. "
            "Every 10-ticket world obeys the original Structured Net principle; "
            "no A-good/B/characteristic-number filter."
        ),
        "history_draws": int(len(df)),
        "usable_targets": int(len(out)),
        "worlds_per_round": args.worlds,
        "plain_reps": args.plain_reps,
        "overall": summarize(out),
        "capture_5_6": summarize(out[out["capture"].isin([5, 6])]),
        "by_capture": {},
    }

    for cap, g in out.groupby("capture"):
        summary["by_capture"][str(int(cap))] = {
            "rounds": int(len(g)),
            **summarize(g),
        }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LOTO7 STRUCTURED WORLD BLEND50 v0 ===")
    print(f"history_draws={len(df)} usable_targets={len(out)} worlds={args.worlds} plain_reps={args.plain_reps}")
    print("NO A-good / NO B residual / NO characteristic-number filter")
    print("All 10-ticket worlds preserve: 70 slots over 18 candidates + pair-reuse minimization")

    for label, g in [
        ("overall", out),
        ("capture56", out[out["capture"].isin([5, 6])]),
    ]:
        print(f"[{label}] n={len(g)}")
        for m in ["max_hits", "reach3", "reach4", "reach5", "reach6", "reach7"]:
            a = g[f"smart_{m}"].mean()
            b = g[f"plain_{m}"].mean()
            print(f"{m}: smart={a:.6f} plain={b:.6f} delta={a-b:+.6f}")
        print(
            "layout_union: "
            f"quint {g['smart_quint_union'].mean():.2f}/{g['plain_quint_union'].mean():.2f} "
            f"quad {g['smart_quad_union'].mean():.2f}/{g['plain_quad_union'].mean():.2f} "
            f"triple {g['smart_triple_union'].mean():.2f}/{g['plain_triple_union'].mean():.2f}"
        )

    for cap in sorted(out["capture"].unique()):
        if cap < 4:
            continue
        g = out[out["capture"] == cap]
        print(
            f"capture={cap} n={len(g)} "
            f"5+ smart={g['smart_reach5'].mean():.6f} plain={g['plain_reach5'].mean():.6f} "
            f"delta={g['smart_reach5'].mean()-g['plain_reach5'].mean():+.6f} "
            f"6+ smart={g['smart_reach6'].mean():.6f} plain={g['plain_reach6'].mean():.6f}"
        )

    print(f"saved -> {OUT_CSV}")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
