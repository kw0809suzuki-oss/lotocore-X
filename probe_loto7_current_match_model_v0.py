from __future__ import annotations

from math import comb, sqrt
from pathlib import Path
import random
import statistics

import pandas as pd

import lotocore
import x_agent
import probe_loto7_dynamic_band_online_replay_v0 as box
from probe_loto7_final7_abox10_amplification_v0 import blend10

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_current_match_model_v0.csv")
WINDOW = 100
POOL_K = 18
ALLOCATOR_REPS = 30
NEIGHBOR_KS = (30, 50, 70)
DEFAULT_K = 50
MODELS = ("core", "x", "blend", "narrow", "abox")


def ranked(snapshot):
    ranks = {int(n): int(r) for n, r in snapshot["ranks"].items()}
    return sorted(ranks, key=lambda n: (ranks[n], n))


def scores(snapshot):
    return {int(n): float(v) for n, v in snapshot["scores"].items()}


def weighted_top10_agreement(core_snap, x_snap):
    """0..1 continuous agreement using both shared identity and rank."""
    cr = ranked(core_snap)
    xr = ranked(x_snap)
    rc = {n: i + 1 for i, n in enumerate(cr[:10])}
    rx = {n: i + 1 for i, n in enumerate(xr[:10])}
    denom = sum(w * w for w in range(1, 11))  # 385
    num = 0.0
    for n in set(rc) & set(rx):
        num += (11 - rc[n]) * (11 - rx[n])
    return num / denom


def top10_concentration(snapshot):
    sc = scores(snapshot)
    rr = ranked(snapshot)
    total = sum(max(0.0, sc[n]) for n in sc)
    if total <= 0:
        return 0.0
    return sum(max(0.0, sc[n]) for n in rr[:10]) / total


def state_features(history):
    cs = lotocore.score_snapshot(history)
    xs = x_agent.score_snapshot(history, competition_gate=True)
    return {
        "agreement": weighted_top10_agreement(cs, xs),
        "concentration": (
            top10_concentration(cs) + top10_concentration(xs)
        ) / 2.0,
        "core_ranked": ranked(cs),
        "x_ranked": ranked(xs),
    }


def weighted_ticket(pool, weights, rng):
    available = list(pool)
    out = []
    while len(out) < 7 and available:
        total = sum(weights[n] for n in available)
        r = rng.random() * total
        idx = 0
        for idx, n in enumerate(available):
            r -= weights[n]
            if r <= 0:
                break
        out.append(available.pop(idx))
    return tuple(sorted(out))


def make_pack(pool, seed, weights=None):
    if len(pool) < 7:
        raise ValueError("pool too small")
    w = weights or {n: 1.0 for n in pool}
    rng = random.Random(seed)
    out = []
    seen = set()
    for _ in range(5000):
        if len(out) >= 10:
            break
        t = weighted_ticket(pool, w, rng)
        if t not in seen:
            seen.add(t)
            out.append(t)
    if len(out) != 10:
        raise RuntimeError("could not create 10 unique tickets")
    return out


def max_hits(actual, tickets):
    aset = set(actual)
    return max(len(aset & set(t)) for t in tickets)


def proxy_model_metrics(actual, state, rnd):
    core_pool = state["core_ranked"][:POOL_K]
    x_pool = state["x_ranked"][:POOL_K]
    blend_pool = sorted(set(core_pool) | set(x_pool))
    cset = set(core_pool)
    xset = set(x_pool)
    blend_weights = {
        n: (2.0 if n in cset and n in xset else 1.0)
        for n in blend_pool
    }
    narrow_pool = state["x_ranked"][:10]

    vals = {m: [] for m in ("core", "x", "blend", "narrow")}
    for rep in range(ALLOCATOR_REPS):
        base = rnd * 1_000_003 + rep * 97 + 20260927
        packs = {
            "core": make_pack(core_pool, base + 11),
            "x": make_pack(x_pool, base + 23),
            "blend": make_pack(blend_pool, base + 37, blend_weights),
            "narrow": make_pack(narrow_pool, base + 53),
        }
        for m, pack in packs.items():
            vals[m].append(max_hits(actual, pack))

    out = {}
    for m, arr in vals.items():
        out[m] = {
            "max": statistics.mean(arr),
            "p4": statistics.mean(v >= 4 for v in arr),
        }
    return out


def random_unique_pack_at_least(k, tickets=10):
    total = comb(37, 7)
    good = sum(comb(7, h) * comb(30, 7 - h) for h in range(k, 8))
    if good <= 0:
        return 0.0
    if total - good < tickets:
        return 1.0
    return 1.0 - comb(total - good, tickets) / comb(total, tickets)


def random_unique_pack_expected_max(tickets=10):
    return sum(random_unique_pack_at_least(k, tickets) for k in range(1, 8))


RANDOM_MEAN = random_unique_pack_expected_max(10)
RANDOM_P4 = random_unique_pack_at_least(4, 10)


def mean_sd(values):
    mu = statistics.mean(values)
    sd = statistics.pstdev(values)
    return mu, max(sd, 1e-12)


def distances(current, prior):
    am, asd = mean_sd([r["agreement"] for r in prior])
    cm, csd = mean_sd([r["concentration"] for r in prior])
    out = []
    for r in prior:
        da = (current["agreement"] - r["agreement"]) / asd
        dc = (current["concentration"] - r["concentration"]) / csd
        out.append((sqrt(da * da + dc * dc), r))
    out.sort(key=lambda x: (x[0], x[1]["round"]))
    return out


def local_stats(neighbors):
    out = {}
    for m in MODELS:
        out[m] = {
            "mean_max": statistics.mean(r[f"{m}_max"] for _, r in neighbors),
            "p4": statistics.mean(r[f"{m}_p4"] for _, r in neighbors),
        }
    return out


def choose_model(stats):
    # Fixed rule: local mean max first, then local 4+ rate, then stable name order.
    return max(
        MODELS,
        key=lambda m: (
            stats[m]["mean_max"],
            stats[m]["p4"],
            -MODELS.index(m),
        ),
    )


def summarize_eval(records, label, k):
    part = [r for r in records if r["k"] == k]
    if not part:
        return
    print(f"--- {label} K={k} n={len(part)} ---")
    print(
        f"selector mean_max={statistics.mean(r['selected_actual_max'] for r in part):.6f} "
        f"4+={statistics.mean(r['selected_actual_p4'] for r in part):.6%} "
        f"vs_random mean={statistics.mean(r['selected_actual_max'] for r in part)-RANDOM_MEAN:+.6f} "
        f"4+={(statistics.mean(r['selected_actual_p4'] for r in part)-RANDOM_P4)*100:+.3f}pt"
    )
    counts = {m: sum(r["selected_model"] == m for r in part) for m in MODELS}
    print(f"selection_counts={counts}")
    print(
        f"neighbor_distance mean={statistics.mean(r['neighbor_mean_distance'] for r in part):.4f} "
        f"max={statistics.mean(r['neighbor_max_distance'] for r in part):.4f}"
    )
    for m in MODELS:
        print(
            f"STATIC {m:6s} mean_max={statistics.mean(r[f'{m}_actual_max'] for r in part):.6f} "
            f"4+={statistics.mean(r[f'{m}_actual_p4'] for r in part):.6%}"
        )
    print()


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    boxes = {}
    for i in range(WINDOW, len(df)):
        boxes[int(df.iloc[i]["round"])] = box.snap(df.iloc[i-WINDOW:i])
    abox_stats = box.empty_stats()

    rows = []
    for i in range(WINDOW + 1, len(df)):
        rnd = int(df.iloc[i]["round"])
        prev_rnd = int(df.iloc[i - 1]["round"])
        history = df.iloc[i-WINDOW:i]
        actual = box.truth(df.iloc[i])

        state = state_features(history)
        proxy = proxy_model_metrics(actual, state, rnd)

        cur_box, prev_box = boxes[rnd], boxes[prev_rnd]
        delta, sets = box.candidate_sets(cur_box, prev_box)
        b = box.band(abs(delta))
        if b == "zero":
            chosen_mag = 0.0
        else:
            alive, chosen_mag, _ = box.live_band(abox_stats[b])
            if not alive:
                chosen_mag = 0.0
        final7 = sets[chosen_mag]
        abox_tickets = blend10(cur_box, final7, delta)
        ah = max_hits(actual, abox_tickets)

        row = {
            "round": rnd,
            "agreement": state["agreement"],
            "concentration": state["concentration"],
            "core_max": proxy["core"]["max"],
            "core_p4": proxy["core"]["p4"],
            "x_max": proxy["x"]["max"],
            "x_p4": proxy["x"]["p4"],
            "blend_max": proxy["blend"]["max"],
            "blend_p4": proxy["blend"]["p4"],
            "narrow_max": proxy["narrow"]["max"],
            "narrow_p4": proxy["narrow"]["p4"],
            "abox_max": float(ah),
            "abox_p4": float(ah >= 4),
        }
        rows.append(row)

        # Strict online update for A-BOX only after this target is scored.
        if b != "zero":
            base = sets[0.0]
            abox_stats[b]["n"] += 1
            abox_stats[b]["base_h"] += box.hits(actual, base)
            abox_stats[b]["base_d"] += box.dist(actual, base)
            for mag in box.MAGS:
                abox_stats[b]["arms"][mag]["h"] += box.hits(actual, sets[mag])
                abox_stats[b]["arms"][mag]["d"] += box.dist(actual, sets[mag])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    evals = []
    max_k = max(NEIGHBOR_KS)
    for idx, current in enumerate(rows):
        prior = rows[:idx]
        if len(prior) < max_k:
            continue
        ds = distances(current, prior)
        for k in NEIGHBOR_KS:
            neighbors = ds[:k]
            ls = local_stats(neighbors)
            selected = choose_model(ls)
            sorted_models = sorted(
                MODELS,
                key=lambda m: (ls[m]["mean_max"], ls[m]["p4"], -MODELS.index(m)),
                reverse=True,
            )
            second = sorted_models[1]
            rec = {
                "round": current["round"],
                "k": k,
                "selected_model": selected,
                "selected_actual_max": current[f"{selected}_max"],
                "selected_actual_p4": current[f"{selected}_p4"],
                "local_top_mean": ls[selected]["mean_max"],
                "local_second_mean": ls[second]["mean_max"],
                "local_margin": ls[selected]["mean_max"] - ls[second]["mean_max"],
                "neighbor_mean_distance": statistics.mean(d for d, _ in neighbors),
                "neighbor_max_distance": max(d for d, _ in neighbors),
            }
            for m in MODELS:
                rec[f"{m}_actual_max"] = current[f"{m}_max"]
                rec[f"{m}_actual_p4"] = current[f"{m}_p4"]
            evals.append(rec)

    print("=== LOTO7 CURRENT MATCH MODEL v0 ===")
    print("State = continuous CORE/X weighted top10 agreement + candidate concentration.")
    print("Distance = equal-weight Euclidean distance after PRIOR-only standardization.")
    print(f"Proxy pools: CORE/X top{POOL_K}; Blend union with shared x2; Narrow X top10.")
    print(f"Ticket allocator averaged over {ALLOCATOR_REPS} deterministic packs per round.")
    print("A-BOX uses frozen online chronology.")
    print("Historical CORE/X/Blend/Narrow are GitHub-reproducible proxies, not saved Floot AI outputs.")
    print()
    print(
        f"RANDOM STANDARD 10 unique tickets: expected_max={RANDOM_MEAN:.6f} "
        f"4+={RANDOM_P4:.6%}"
    )
    print()

    for k in NEIGHBOR_KS:
        summarize_eval(evals, "ALL ELIGIBLE", k)
        kpart = [r for r in evals if r["k"] == k]
        summarize_eval(kpart[-200:], "LAST200", k)

    # Current pre-draw read using all completed historical outcomes as neighbor evidence.
    current_state = state_features(df.tail(WINDOW))
    pseudo = {
        "round": int(df.iloc[-1]["round"]) + 1,
        "agreement": current_state["agreement"],
        "concentration": current_state["concentration"],
    }
    ds = distances(pseudo, rows)
    neighbors = ds[:DEFAULT_K]
    ls = local_stats(neighbors)
    order = sorted(
        MODELS,
        key=lambda m: (ls[m]["mean_max"], ls[m]["p4"], -MODELS.index(m)),
        reverse=True,
    )
    print("=== CURRENT PRE-DRAW MATCH | K=50 ===")
    print(
        f"history_end={int(df.iloc[-1]['round'])} next_round={int(df.iloc[-1]['round'])+1} "
        f"agreement={pseudo['agreement']:.6f} concentration={pseudo['concentration']:.6f}"
    )
    print(
        f"neighbors={DEFAULT_K} avg_distance={statistics.mean(d for d,_ in neighbors):.4f} "
        f"max_distance={max(d for d,_ in neighbors):.4f}"
    )
    for m in order:
        print(
            f"{m:6s} local_mean_max={ls[m]['mean_max']:.6f} "
            f"delta_random={ls[m]['mean_max']-RANDOM_MEAN:+.6f} "
            f"local_4+={ls[m]['p4']:.6%} "
            f"delta_random_4+={(ls[m]['p4']-RANDOM_P4)*100:+.3f}pt"
        )
    print(
        f"top={order[0]} margin_to_second="
        f"{ls[order[0]]['mean_max']-ls[order[1]]['mean_max']:+.6f}"
    )
    print()

    print("BOUNDARY: random is the fixed ordinary 10-ticket standard, not a forecast.")
    print("BOUNDARY: local past performance is not a future winning probability.")
    print("BOUNDARY: v0 tests whether the local-match mechanism adds stable out-of-sample value.")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
