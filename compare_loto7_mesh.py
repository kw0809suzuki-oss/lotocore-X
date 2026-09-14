from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import lotocore
import x_agent

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_mesh_compare.csv")
MODELS = (
    "lotocore",
    "x",
    "x_ungated",
    "x_spacing2",
    "x_actionpoint",
)


def actual_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def model_predictions(history) -> dict[str, tuple[int, ...]]:
    return {
        "lotocore": lotocore.predict(history).numbers,
        "x": x_agent.predict(history, competition_gate=True).numbers,
        "x_ungated": x_agent.predict(history, competition_gate=False).numbers,
        "x_spacing2": x_agent.predict(history, competition_gate=True, spacing_v2=True).numbers,
        "x_actionpoint": x_agent.predict(
            history, competition_gate=True, spacing_v2=True, action_point=True
        ).numbers,
    }


def make_degrees(preds: dict[str, tuple[int, ...]], total_slots: int = 140) -> Counter:
    base = Counter()
    for name in MODELS:
        base.update(preds[name])
    raw = {n: total_slots * c / sum(base.values()) for n, c in base.items()}
    deg = {n: min(20, int(v)) for n, v in raw.items()}
    used = sum(deg.values())
    order = sorted(base, key=lambda n: (raw[n] - int(raw[n]), base[n], -n), reverse=True)
    while used < total_slots:
        changed = False
        for n in order:
            if deg[n] < 20:
                deg[n] += 1
                used += 1
                changed = True
                if used == total_slots:
                    break
        if not changed:
            break
    while used > total_slots:
        for n in reversed(order):
            if deg[n] > 0:
                deg[n] -= 1
                used -= 1
                if used == total_slots:
                    break
    return Counter(deg)


def choose_feasible_ticket(remaining: Counter, tickets_left: int, rng: random.Random, pair_count=None) -> tuple[int, ...]:
    mandatory = [n for n, c in remaining.items() if c == tickets_left]
    if len(mandatory) > 7:
        raise RuntimeError("infeasible degree sequence: too many mandatory nodes")
    chosen = list(mandatory)
    while len(chosen) < 7:
        candidates = [n for n, c in remaining.items() if c > 0 and n not in chosen]
        if not candidates:
            raise RuntimeError("infeasible degree sequence: not enough distinct candidates")
        if pair_count is None:
            weights = [remaining[n] for n in candidates]
            total = sum(weights)
            r = rng.uniform(0, total)
            acc = 0.0
            pick = candidates[-1]
            for n, w in zip(candidates, weights):
                acc += w
                if r <= acc:
                    pick = n
                    break
        else:
            rng.shuffle(candidates)
            def score(n: int):
                pair_penalty = sum(pair_count[tuple(sorted((n, x)))] for x in chosen)
                return (pair_penalty, -remaining[n], n)
            pick = min(candidates, key=score)
        chosen.append(pick)
    for n in chosen:
        remaining[n] -= 1
    return tuple(sorted(chosen))


def random_pack(degrees: Counter, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    remaining = Counter(degrees)
    tickets = []
    for ticket_index in range(20):
        tickets_left = 20 - ticket_index
        tickets.append(choose_feasible_ticket(remaining, tickets_left, rng))
    return tickets


def broad_mesh_pack(degrees: Counter, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    remaining = Counter(degrees)
    pair_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    tickets = []
    for ticket_index in range(20):
        tickets_left = 20 - ticket_index
        ticket = choose_feasible_ticket(remaining, tickets_left, rng, pair_count)
        for i in range(7):
            for j in range(i + 1, 7):
                pair_count[(ticket[i], ticket[j])] += 1
        tickets.append(ticket)
    return tickets


def max_hits(tickets: list[tuple[int, ...]], actual: tuple[int, ...]) -> int:
    aset = set(actual)
    return max(len(aset & set(t)) for t in tickets)


def pair_score(tickets: list[tuple[int, ...]]) -> int:
    c = Counter()
    for t in tickets:
        for i in range(7):
            for j in range(i + 1, 7):
                c[(t[i], t[j])] += 1
    return sum(v * v for v in c.values())


def run(window: int = 100) -> pd.DataFrame:
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows = []
    for i in range(window, len(df)):
        history = df.iloc[i-window:i]
        row = df.iloc[i]
        rnd = int(row["round"])
        actual = actual_numbers(row)
        preds = model_predictions(history)
        degrees = make_degrees(preds)
        random_tickets = random_pack(degrees, seed=rnd * 1009 + 17)
        mesh_tickets = broad_mesh_pack(degrees, seed=rnd * 1009 + 17)
        rh = max_hits(random_tickets, actual)
        mh = max_hits(mesh_tickets, actual)
        rows.append({
            "round": rnd,
            "date": row["date"],
            "random_max_hits": rh,
            "mesh_max_hits": mh,
            "delta_mesh_minus_random": mh - rh,
            "random_pair_score": pair_score(random_tickets),
            "mesh_pair_score": pair_score(mesh_tickets),
            "candidate_nodes": len(degrees),
        })
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> None:
    wins = int((res.delta_mesh_minus_random > 0).sum())
    ties = int((res.delta_mesh_minus_random == 0).sum())
    losses = int((res.delta_mesh_minus_random < 0).sum())
    print("\n=== LOTO7 MESH COARSE OBSERVATION ===")
    print(f"n={len(res)} Mesh>Random={wins} Mesh=Random={ties} Mesh<Random={losses}")
    for name, col in (("Random", "random_max_hits"), ("Mesh", "mesh_max_hits")):
        g = res[col]
        print(
            f"{name}: mean_max={g.mean():.4f} "
            f"3+={(g>=3).mean():.4f} 4+={(g>=4).mean():.4f} "
            f"5+={(g>=5).mean():.4f} best={int(g.max())} "
            f"dist={g.value_counts().sort_index().to_dict()}"
        )
    print(
        f"pair_score_mean Random={res.random_pair_score.mean():.2f} "
        f"Mesh={res.mesh_pair_score.mean():.2f}"
    )
    print("WHY_NOT_ANALYZED=1")
    print("LOTO7_MESH_COARSE_OBSERVATION_COMPLETE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()
    res = run(args.window)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    summarize(res)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
