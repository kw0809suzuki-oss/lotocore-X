from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import statistics

import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, ALPHA, NUMBERS,
    draws_from_df, frequency_distribution, mix_distribution,
    distribution_state, rank_numbers, choose_shape_aware,
)

OUT = Path("results/loto7_cworld_numeric_fit_v0.csv")
BOUNDARY_K = 3
ROUTE_K = 5
C3_ALPHAS = (0.75, 0.50, 0.25)
EPS = 1e-12


def truth(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def hits(actual, picks):
    return len(set(actual) & set(picks))


def normalize_score(d):
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo <= EPS:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def make_state(round_no, p, prev_p):
    ranked = rank_numbers(p)
    ac, asp = distribution_state(p)
    core, *_ = choose_shape_aware(ranked, p, ac, asp)
    core = tuple(sorted(core))
    core_set = set(core)
    boundary = tuple(n for n in ranked if n not in core_set)[:BOUNDARY_K]
    delta = {n: 0.0 if prev_p is None else p[n] - prev_p[n] for n in NUMBERS}

    if prev_p is None:
        dc = 0.0
        ds = 0.0
    else:
        pc, ps = distribution_state(prev_p)
        dc = ac - pc
        ds = asp - ps

    return {
        "round": round_no,
        "p": p,
        "core": core,
        "boundary": boundary,
        "delta": delta,
        "center": ac,
        "spread": asp,
        "d_center": dc,
        "d_spread": ds,
    }


def c1_a_hold_boundary(state):
    # A-hold candidate: preserve six or seven current core numbers.
    # Open at most one slot, only when a boundary number is moving upward
    # more strongly than the weakest-moving core member.
    core = list(state["core"])
    boundary = list(state["boundary"])
    if not boundary:
        return tuple(sorted(core))

    best_b = max(boundary, key=lambda n: (state["delta"][n], state["p"][n], -n))
    weakest_c = min(core, key=lambda n: (state["delta"][n], state["p"][n], n))

    if state["delta"][best_b] > max(0.0, state["delta"][weakest_c]) + EPS:
        return tuple(sorted((set(core) - {weakest_c}) | {best_b}))
    return tuple(sorted(core))


def feature_distance(cur, old, scales):
    vals = (
        (cur["center"] - old["center"]) / scales[0],
        (cur["spread"] - old["spread"]) / scales[1],
        (cur["d_center"] - old["d_center"]) / scales[2],
        (cur["d_spread"] - old["d_spread"]) / scales[3],
    )
    return math.sqrt(sum(v * v for v in vals))


def robust_scales(states):
    if len(states) < 8:
        return (1.0, 1.0, 1.0, 1.0)

    keys = ("center", "spread", "d_center", "d_spread")
    out = []
    for key in keys:
        xs = [float(s[key]) for s in states]
        sd = statistics.pstdev(xs)
        out.append(sd if sd > EPS else 1.0)
    return tuple(out)


def shifted_distribution(cur_p, old_p, old_next_p):
    # Transfer a past A-world movement vector onto the current A-world.
    raw = {}
    for n in NUMBERS:
        v = cur_p[n] + (old_next_p[n] - old_p[n])
        raw[n] = max(0.0, v)
    total = sum(raw.values())
    if total <= EPS:
        return dict(cur_p)
    return {n: raw[n] / total for n in NUMBERS}


def c2_common_survivor(cur, past_states):
    # B is not declared. We use several prior, already-observed A->next-A
    # movements that were locally similar to the current A state as plausible routes.
    # No target outcome is used.
    if len(past_states) < 12:
        return tuple(cur["core"])

    scales = robust_scales(past_states)
    usable = []
    # old index k needs old_next=k+1, and both must already be known.
    for k in range(len(past_states) - 1):
        old = past_states[k]
        nxt = past_states[k + 1]
        d = feature_distance(cur, old, scales)
        usable.append((d, old["round"], old, nxt))

    usable.sort(key=lambda x: (x[0], x[1]))
    routes = []
    for _, _, old, nxt in usable[:ROUTE_K]:
        q = shifted_distribution(cur["p"], old["p"], nxt["p"])
        routes.append(tuple(sorted(rank_numbers(q)[:7])))

    if not routes:
        return tuple(cur["core"])

    survive = Counter(n for route in routes for n in route)
    ranked = sorted(
        NUMBERS,
        key=lambda n: (-survive[n], -cur["p"][n], n),
    )
    return tuple(sorted(ranked[:7]))


def c3_balance(cur, alpha):
    # A mass + currently-open boundary freedom.
    # Freedom is only carried by the explicit boundary and positive local motion.
    support = set(cur["core"]) | set(cur["boundary"])
    a_raw = {n: cur["p"][n] if n in support else 0.0 for n in NUMBERS}
    f_raw = {
        n: (max(cur["delta"][n], 0.0) + 0.25 * cur["p"][n])
        if n in set(cur["boundary"]) else 0.0
        for n in NUMBERS
    }
    a = normalize_score(a_raw)
    f = normalize_score(f_raw)
    score = {n: alpha * a[n] + (1.0 - alpha) * f[n] for n in NUMBERS}
    ranked = sorted(NUMBERS, key=lambda n: (-score[n], -cur["p"][n], n))
    return tuple(sorted(ranked[:7]))


def summarize(rows, label):
    if not rows:
        return

    names = ["baseline_top7", "a_core", "c1", "c2"] + [f"c3_{a:.2f}" for a in C3_ALPHAS]
    print(f"--- {label} n={len(rows)} ---")
    base_mean = statistics.mean(r["baseline_top7_h"] for r in rows)

    for name in names:
        vals = [r[f"{name}_h"] for r in rows]
        mh = statistics.mean(vals)
        p3 = statistics.mean(v >= 3 for v in vals)
        p4 = statistics.mean(v >= 4 for v in vals)
        p5 = statistics.mean(v >= 5 for v in vals)
        print(
            f"{name:14s} mean={mh:.6f} "
            f"delta_vs_top7={mh-base_mean:+.6f} "
            f"3+={p3:.6%} 4+={p4:.6%} 5+={p5:.6%}"
        )
    print()


def main():
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws = draws_from_df(df)

    states = []
    prev_p = None
    for i in range(LONG_WINDOW, len(draws)):
        lp = frequency_distribution(draws[i - LONG_WINDOW:i])
        rp = frequency_distribution(draws[i - RECENT_WINDOW:i])
        p = mix_distribution(lp, rp)
        states.append(make_state(int(df.iloc[i]["round"]), p, prev_p))
        prev_p = p

    rows = []
    for sidx, cur in enumerate(states):
        # states[sidx] corresponds to target dataframe row LONG_WINDOW+sidx.
        i = LONG_WINDOW + sidx
        if sidx == 0:
            continue

        actual = truth(df.iloc[i])
        top7 = tuple(sorted(rank_numbers(cur["p"])[:7]))
        a_core = tuple(cur["core"])
        c1 = c1_a_hold_boundary(cur)

        # Critical chronology:
        # for target i, only A states up through current target-state are known.
        # c2 may use transitions old->old_next only when old_next is already in past/current A.
        past_known = states[: sidx + 1]
        c2 = c2_common_survivor(cur, past_known)

        c3s = {a: c3_balance(cur, a) for a in C3_ALPHAS}

        rec = {
            "round": cur["round"],
            "baseline_top7": "-".join(f"{n:02d}" for n in top7),
            "a_core": "-".join(f"{n:02d}" for n in a_core),
            "c1": "-".join(f"{n:02d}" for n in c1),
            "c2": "-".join(f"{n:02d}" for n in c2),
            "baseline_top7_h": hits(actual, top7),
            "a_core_h": hits(actual, a_core),
            "c1_h": hits(actual, c1),
            "c2_h": hits(actual, c2),
        }
        for a, picks in c3s.items():
            key = f"c3_{a:.2f}"
            rec[key] = "-".join(f"{n:02d}" for n in picks)
            rec[f"{key}_h"] = hits(actual, picks)
        rows.append(rec)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== LOTO7 C-WORLD NUMERIC FIT v0 ===")
    print("Parent objective: improve next-7-number prediction fit.")
    print("A/B/C are only candidate representations; none is promoted by construction.")
    print("A-world source: frozen Slow100 + Fast20 alpha=0.72 representation and shape-aware A-core.")
    print(f"Boundary={BOUNDARY_K}, C2 routes={ROUTE_K}, C3 alphas={C3_ALPHAS}")
    print("Target outcome is used only after each candidate 7 has been frozen.")
    print()

    summarize(rows, "ALL")
    summarize(rows[-200:], "LAST200")
    summarize(rows[-100:], "LAST100")

    cut = int(len(rows) * 0.60)
    summarize(rows[:cut], "EARLY60")
    summarize(rows[cut:], "LATE40")

    print("RECENT 20")
    cols = ["round", "baseline_top7_h", "a_core_h", "c1_h", "c2_h"] + [f"c3_{a:.2f}_h" for a in C3_ALPHAS]
    print(res[cols].tail(20).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
