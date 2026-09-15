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
MODELS = ("lotocore", "x", "x_ungated", "x_spacing2", "x_actionpoint")


def actual_numbers(row):
    return tuple(sorted(int(row[f"n{i}"]) for i in range(1, 8)))


def model_predictions(history):
    return {
        "lotocore": lotocore.predict(history).numbers,
        "x": x_agent.predict(history, competition_gate=True).numbers,
        "x_ungated": x_agent.predict(history, competition_gate=False).numbers,
        "x_spacing2": x_agent.predict(history, competition_gate=True, spacing_v2=True).numbers,
        "x_actionpoint": x_agent.predict(history, competition_gate=True, spacing_v2=True, action_point=True).numbers,
    }


def make_degrees(preds, total_slots=140):
    base = Counter()
    for name in MODELS:
        base.update(preds[name])
    raw = {n: total_slots * c / sum(base.values()) for n, c in base.items()}
    deg = {n: min(20, int(v)) for n, v in raw.items()}
    used = sum(deg.values())
    order = sorted(base, key=lambda n: (raw[n] - int(raw[n]), base[n], -n), reverse=True)
    while used < total_slots:
        for n in order:
            if deg[n] < 20:
                deg[n] += 1; used += 1
                if used == total_slots: break
    while used > total_slots:
        for n in reversed(order):
            if deg[n] > 0:
                deg[n] -= 1; used -= 1
                if used == total_slots: break
    return Counter(deg)


def choose_ticket(remaining, tickets_left, rng, pair_count=None):
    chosen = [n for n, c in remaining.items() if c == tickets_left]
    if len(chosen) > 7: raise RuntimeError("infeasible degrees")
    while len(chosen) < 7:
        candidates = [n for n, c in remaining.items() if c > 0 and n not in chosen]
        if pair_count is None:
            weights = [remaining[n] for n in candidates]
            pick = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            rng.shuffle(candidates)
            pick = min(candidates, key=lambda n: (sum(pair_count[tuple(sorted((n,x)))] for x in chosen), -remaining[n], n))
        chosen.append(pick)
    for n in chosen: remaining[n] -= 1
    return tuple(sorted(chosen))


def random_pack(degrees, seed):
    rng = random.Random(seed); rem = Counter(degrees); out = []
    for i in range(20): out.append(choose_ticket(rem, 20-i, rng))
    return out


def broad_mesh_pack(degrees, seed):
    rng = random.Random(seed); rem = Counter(degrees); pc = defaultdict(int); out = []
    for i in range(20):
        t = choose_ticket(rem, 20-i, rng, pc)
        for a in range(7):
            for b in range(a+1,7): pc[(t[a],t[b])] += 1
        out.append(t)
    return out


def pair_score(tickets):
    c = Counter()
    for t in tickets:
        for i in range(7):
            for j in range(i+1,7): c[(t[i],t[j])] += 1
    return sum(v*v for v in c.values())


def loto6_bundle_pack(degrees, seed):
    """Random first; observe bundle; only a few degree-preserving swaps if they reduce overlap."""
    rng = random.Random(seed)
    tickets = [list(t) for t in random_pack(degrees, seed)]
    # Minimal correction: at most two accepted swaps across the whole 20-ticket bundle.
    accepted = 0
    for _ in range(200):
        if accepted >= 2: break
        i, j = rng.sample(range(20), 2)
        ai = [x for x in tickets[i] if x not in tickets[j]]
        bj = [x for x in tickets[j] if x not in tickets[i]]
        if not ai or not bj: continue
        a, b = rng.choice(ai), rng.choice(bj)
        before = pair_score(tickets)
        ni = sorted([b if x == a else x for x in tickets[i]])
        nj = sorted([a if x == b else x for x in tickets[j]])
        if len(set(ni)) < 7 or len(set(nj)) < 7: continue
        old_i, old_j = tickets[i], tickets[j]
        tickets[i], tickets[j] = ni, nj
        if pair_score(tickets) < before:
            accepted += 1
        else:
            tickets[i], tickets[j] = old_i, old_j
    return [tuple(t) for t in tickets]


def max_hits(tickets, actual):
    aset = set(actual)
    return max(len(aset & set(t)) for t in tickets)


def run(window=100):
    df = pd.read_csv(DATA).sort_values("round").reset_index(drop=True); rows=[]
    for i in range(window, len(df)):
        history=df.iloc[i-window:i]; row=df.iloc[i]; rnd=int(row["round"]); actual=actual_numbers(row)
        degrees=make_degrees(model_predictions(history)); seed=rnd*1009+17
        packs={"random":random_pack(degrees,seed), "loto6":loto6_bundle_pack(degrees,seed), "mesh":broad_mesh_pack(degrees,seed)}
        vals={k:max_hits(v,actual) for k,v in packs.items()}
        rows.append({"round":rnd,"date":row["date"],
            "random_max_hits":vals["random"],"loto6_max_hits":vals["loto6"],"mesh_max_hits":vals["mesh"],
            "loto6_minus_random":vals["loto6"]-vals["random"],"mesh_minus_loto6":vals["mesh"]-vals["loto6"],
            "random_pair_score":pair_score(packs["random"]),"loto6_pair_score":pair_score(packs["loto6"]),"mesh_pair_score":pair_score(packs["mesh"]),
            "candidate_nodes":len(degrees)})
    return pd.DataFrame(rows)


def summarize(res):
    print("\n=== LOTO7 THREE-WAY OUTER OBSERVATION ===")
    print(f"n={len(res)}")
    for name in ("random","loto6","mesh"):
        g=res[f"{name}_max_hits"]
        print(f"{name}: mean_max={g.mean():.4f} 3+={(g>=3).mean():.4f} 4+={(g>=4).mean():.4f} 5+={(g>=5).mean():.4f} best={int(g.max())} dist={g.value_counts().sort_index().to_dict()}")
    for a,b in (("loto6","random"),("mesh","loto6"),("mesh","random")):
        d=res[f"{a}_max_hits"]-res[f"{b}_max_hits"]
        print(f"{a}>{b}={(d>0).sum()} equal={(d==0).sum()} {a}<{b}={(d<0).sum()}")
    print(f"pair_score_mean Random={res.random_pair_score.mean():.2f} LOTO6={res.loto6_pair_score.mean():.2f} Mesh={res.mesh_pair_score.mean():.2f}")
    print("WHY_NOT_ANALYZED=1")
    print("LOTO7_THREE_WAY_OUTER_OBSERVATION_COMPLETE")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--window",type=int,default=100); args=p.parse_args()
    res=run(args.window); OUT.parent.mkdir(parents=True,exist_ok=True); res.to_csv(OUT,index=False); summarize(res); print(f"saved -> {OUT}")

if __name__ == "__main__": main()
