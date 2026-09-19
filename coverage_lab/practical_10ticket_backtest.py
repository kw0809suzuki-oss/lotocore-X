#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import x_agent

TICKETS_PER_DRAW = 10
DRAW_SIZE = 7
PRICE_PER_TICKET = 300
GATE_KS = (10, 12)


def main_numbers(row):
    return {int(row[f"n{i}"]) for i in range(1, 8)}


def bonus_numbers(row):
    return {int(row["b1"]), int(row["b2"])}


def ranked_x(history: pd.DataFrame) -> list[int]:
    snap = x_agent.score_snapshot(history, competition_gate=True)
    ranks = {int(k): int(v) for k, v in snap["ranks"].items()}
    return sorted(range(1, 38), key=lambda n: (ranks[n], n))


def gate_diff(m: pd.DataFrame, hist_rounds: list[int]) -> float:
    z = m[m["round"].isin(hist_rounds)]
    diffs = []
    for k in GATE_KS:
        q = z[z.k == k]
        hi = q[q.state_bin == "high"].hit_lift.mean()
        lo = q[q.state_bin == "low"].hit_lift.mean()
        if pd.isna(hi) or pd.isna(lo):
            return float("nan")
        diffs.append(float(hi - lo))
    return sum(diffs) / len(diffs)


def strict_forward_blocks(rounds: list[int], history_window: int, step: int):
    for start in range(0, len(rounds) - history_window, step):
        hist = rounds[start : start + history_window]
        eva = rounds[start + history_window : start + history_window + step]
        if eva:
            yield hist, eva


def rank_degrees(pool_order: list[int]) -> Counter:
    """Translate one ranked candidate pool into exactly 70 ticket slots.

    Every candidate gets one slot first, then remaining slots are allocated
    with a fixed diminishing-return rank weight. Each number can appear at
    most once per ticket, hence degree <= 10.
    """
    total_slots = TICKETS_PER_DRAW * DRAW_SIZE
    if len(pool_order) < DRAW_SIZE:
        raise ValueError("candidate pool must contain at least 7 numbers")
    if len(pool_order) > total_slots:
        raise ValueError("pool too large for one-touch coverage")

    deg = Counter({n: 1 for n in pool_order})
    remaining = total_slots - len(pool_order)
    rank = {n: i for i, n in enumerate(pool_order)}
    weight = {n: len(pool_order) - i for i, n in enumerate(pool_order)}

    for _ in range(remaining):
        candidates = [n for n in pool_order if deg[n] < TICKETS_PER_DRAW]
        pick = max(
            candidates,
            key=lambda n: (
                weight[n] / (deg[n] + 1.0),
                -rank[n],
            ),
        )
        deg[pick] += 1
    if sum(deg.values()) != total_slots:
        raise RuntimeError("degree allocation failed")
    return deg


def choose_ticket(
    remaining: Counter,
    tickets_left: int,
    rank: dict[int, int],
    pair_count: defaultdict,
    rng: random.Random,
) -> tuple[int, ...]:
    chosen = [n for n, c in remaining.items() if c == tickets_left]
    if len(chosen) > DRAW_SIZE:
        raise RuntimeError("infeasible degree schedule")

    while len(chosen) < DRAW_SIZE:
        candidates = [n for n, c in remaining.items() if c > 0 and n not in chosen]
        if not candidates:
            raise RuntimeError("ran out of candidates")
        rng.shuffle(candidates)
        pick = min(
            candidates,
            key=lambda n: (
                sum(pair_count[tuple(sorted((n, x)))] for x in chosen),
                -remaining[n],
                rank[n],
            ),
        )
        chosen.append(pick)

    for n in chosen:
        remaining[n] -= 1
    return tuple(sorted(chosen))


def ranked_bundle(pool_order: list[int], seed: int) -> list[tuple[int, ...]]:
    degrees = rank_degrees(pool_order)
    remaining = Counter(degrees)
    pair_count = defaultdict(int)
    rank = {n: i for i, n in enumerate(pool_order)}
    rng = random.Random(seed)
    out = []

    for i in range(TICKETS_PER_DRAW):
        t = choose_ticket(
            remaining,
            TICKETS_PER_DRAW - i,
            rank,
            pair_count,
            rng,
        )
        for a in range(DRAW_SIZE):
            for b in range(a + 1, DRAW_SIZE):
                pair_count[(t[a], t[b])] += 1
        out.append(t)

    if any(remaining.values()):
        raise RuntimeError(f"nonzero degrees after bundle: {remaining}")
    return out


def random_pool_bundle(k: int, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    pool = list(range(1, 38))
    rng.shuffle(pool)
    if k < 37:
        pool = pool[:k]
    return ranked_bundle(pool, seed + 17)


def uniform_random_bundle(seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < TICKETS_PER_DRAW:
        t = tuple(sorted(rng.sample(range(1, 38), DRAW_SIZE)))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def prize_rank(ticket: tuple[int, ...], main: set[int], bonus: set[int]) -> int:
    s = set(ticket)
    h = len(s & main)
    b = len(s & bonus)
    if h == 7:
        return 1
    if h == 6 and b >= 1:
        return 2
    if h == 6:
        return 3
    if h == 5:
        return 4
    if h == 4:
        return 5
    if h == 3 and b >= 1:
        return 6
    return 0


def theoretical_prize_yen(round_no: int, rank: int) -> int:
    if rank == 0:
        return 0
    if round_no >= 613:
        table = {1: 700_000_000, 2: 6_100_000, 3: 500_000, 4: 6_500, 5: 1_400, 6: 1_000}
    else:
        table = {1: 600_000_000, 2: 7_300_000, 3: 730_000, 4: 9_100, 5: 1_400, 6: 1_000}
    return table[rank]


def pair_score(tickets: list[tuple[int, ...]]) -> int:
    c = Counter()
    for t in tickets:
        for i in range(DRAW_SIZE):
            for j in range(i + 1, DRAW_SIZE):
                c[(t[i], t[j])] += 1
    return sum(v * v for v in c.values())


def bundle_metrics(
    tickets: list[tuple[int, ...]],
    main: set[int],
    bonus: set[int],
    round_no: int,
) -> dict:
    main_hits = [len(set(t) & main) for t in tickets]
    ranks = [prize_rank(t, main, bonus) for t in tickets]
    rank_counts = Counter(r for r in ranks if r)
    best_rank = min((r for r in ranks if r), default=0)
    theo = sum(theoretical_prize_yen(round_no, r) for r in ranks)
    unique_numbers = len(set().union(*(set(t) for t in tickets)))

    return {
        "max_main_hits": max(main_hits),
        "mean_ticket_hits": sum(main_hits) / len(main_hits),
        "best_prize_rank": best_rank,
        "any_prize": int(best_rank > 0),
        "winning_tickets": sum(rank_counts.values()),
        "rank1": rank_counts[1],
        "rank2": rank_counts[2],
        "rank3": rank_counts[3],
        "rank4": rank_counts[4],
        "rank5": rank_counts[5],
        "rank6": rank_counts[6],
        "unique_numbers": unique_numbers,
        "pair_score": pair_score(tickets),
        "theoretical_gross_yen": theo,
        "cost_yen": TICKETS_PER_DRAW * PRICE_PER_TICKET,
        "theoretical_net_yen": theo - TICKETS_PER_DRAW * PRICE_PER_TICKET,
        "tickets": ";".join("-".join(f"{n:02d}" for n in t) for t in tickets),
    }


def run(
    data_path: Path,
    state_path: Path,
    history_window: int = 60,
    step: int = 20,
    model_window: int = 100,
) -> pd.DataFrame:
    df = pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    by_round = {int(r["round"]): i for i, r in df.iterrows()}

    obs = pd.read_csv(state_path).dropna(subset=["state_bin"])
    m = obs[(obs.model == "x") & (obs.metric == "field_center_abs")].copy()
    rounds = sorted(int(x) for x in m["round"].unique())

    rows = []
    signs: list[bool] = []

    for block_idx, (hist_rounds, eval_rounds) in enumerate(
        strict_forward_blocks(rounds, history_window, step)
    ):
        diff = gate_diff(m, hist_rounds)
        if pd.isna(diff):
            continue
        signs.append(diff > 0)
        fresh = bool(signs[-1])  # frozen Phase 10 candidate: last1

        for rnd in eval_rounds:
            z = m[(m["round"] == rnd) & (m.k == 10)]
            if z.empty:
                continue
            state_bin = str(z.iloc[0].state_bin)
            adaptive_k = 10 if (fresh and state_bin == "high") else 37

            idx = by_round[rnd]
            if idx < model_window:
                continue
            history = df.iloc[idx - model_window : idx]
            row = df.iloc[idx]
            main = main_numbers(row)
            bonus = bonus_numbers(row)
            ranked = ranked_x(history)

            seed = rnd * 100_003 + 20260919
            strategies = {
                "adaptive_x_last1_k10": ranked_bundle(ranked[:adaptive_k], seed + 1),
                "fixed_x_k10": ranked_bundle(ranked[:10], seed + 2),
                "fixed_x_k37": ranked_bundle(ranked, seed + 3),
                "same_k_random": random_pool_bundle(adaptive_k, seed + 4),
                "uniform_random_tickets": uniform_random_bundle(seed + 5),
            }

            for strategy, tickets in strategies.items():
                rec = {
                    "round": rnd,
                    "date": row.get("date", ""),
                    "block_idx": block_idx,
                    "gate_mean_diff": diff,
                    "fresh": int(fresh),
                    "state_bin": state_bin,
                    "adaptive_k": adaptive_k,
                    "strategy": strategy,
                }
                rec.update(bundle_metrics(tickets, main, bonus, rnd))
                rows.append(rec)

    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame):
    print("=== LOTO7 PHASE 10 PRACTICAL 10-TICKET BACKTEST ===")
    print("Frozen candidate: X / last1 / K10. 10 tickets per draw. Price=300 yen/ticket.")
    print("Prize ranks use main+bonus numbers. Money column uses official theoretical payout values, not actual historical payouts.")
    print(f"evaluation_rounds={res['round'].nunique()} strategies={res.strategy.nunique()}")

    for strategy in (
        "adaptive_x_last1_k10",
        "fixed_x_k10",
        "fixed_x_k37",
        "same_k_random",
        "uniform_random_tickets",
    ):
        g = res[res.strategy == strategy]
        spend = int(g.cost_yen.sum())
        gross = int(g.theoretical_gross_yen.sum())
        dist = g.max_main_hits.value_counts().sort_index().to_dict()
        ranks = {r: int(g[f"rank{r}"].sum()) for r in range(1, 7)}
        print(
            f"STRATEGY {strategy} n={len(g)} spend_yen={spend} "
            f"any_prize={g.any_prize.mean():.4f} no_prize={(1-g.any_prize.mean()):.4f} "
            f"mean_max_hits={g.max_main_hits.mean():.4f} "
            f"P(max>=4)={(g.max_main_hits>=4).mean():.4f} "
            f"P(max>=5)={(g.max_main_hits>=5).mean():.4f} "
            f"best_main={int(g.max_main_hits.max())} "
            f"rank_counts={ranks} max_hit_dist={dist} "
            f"theoretical_gross_yen={gross} theoretical_net_yen={gross-spend} "
            f"theoretical_return={gross/spend:.4f}"
        )

    pivot = res.pivot(index="round", columns="strategy", values="max_main_hits")
    a = pivot["adaptive_x_last1_k10"]
    for bname in ("fixed_x_k10", "fixed_x_k37", "same_k_random", "uniform_random_tickets"):
        b = pivot[bname]
        print(
            f"PAIRED max-hits adaptive vs {bname}: "
            f"A>B={(a>b).sum()} A=B={(a==b).sum()} A<B={(a<b).sum()}"
        )

    ag = res[res.strategy == "adaptive_x_last1_k10"]
    print(
        f"ADAPTIVE compression_rounds={(ag.adaptive_k==10).sum()}/{len(ag)} "
        f"rate={(ag.adaptive_k==10).mean():.4f} "
        f"mean_unique_numbers={ag.unique_numbers.mean():.2f} "
        f"mean_pair_score={ag.pair_score.mean():.2f}"
    )
    print("BOUNDARY: no future advantage is inferred; this is historical strict-forward ticket-level evidence.")
    print("BOUNDARY: theoretical payout uses official representative/theoretical amounts, not actual per-round historical payouts.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/loto7.csv"))
    p.add_argument(
        "--state",
        type=Path,
        default=Path("results/loto7_candidate_compression_state_observables.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/loto7_phase10_practical_10ticket_backtest.csv"),
    )
    p.add_argument("--history-window", type=int, default=60)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--model-window", type=int, default=100)
    a = p.parse_args()

    res = run(a.data, a.state, a.history_window, a.step, a.model_window)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    summarize(res)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
