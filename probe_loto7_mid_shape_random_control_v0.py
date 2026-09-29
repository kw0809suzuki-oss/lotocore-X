from __future__ import annotations

from pathlib import Path
import math
import random
import statistics

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_mid_shape_random_control_v0.csv")
SIMS = 1000
SEED = 20260929


def draw(row):
    return sorted(int(row[f"n{i}"]) for i in range(1, 8))


def center(xs):
    return sum(xs) / len(xs)


def spread(xs):
    c = center(xs)
    return math.sqrt(sum((x-c)**2 for x in xs) / len(xs))


def mid_count(xs):
    return sum(13 <= n <= 24 for n in xs)


def transition(a,b):
    return {
        "mid_delta": mid_count(b)-mid_count(a),
        "spread_delta": spread(b)-spread(a),
        "range_delta": (max(b)-min(b))-(max(a)-min(a)),
    }


def metrics(draws):
    ts=[transition(draws[i],draws[i+1]) for i in range(len(draws)-1)]

    up_idx=[i for i,t in enumerate(ts[:-1]) if t["mid_delta"]>0]
    dn_idx=[i for i,t in enumerate(ts[:-1]) if t["mid_delta"]<0]

    def mean_next(idxs,key):
        return statistics.mean(ts[i+1][key] for i in idxs) if idxs else float("nan")

    def reversal_rate(idxs,sign):
        if not idxs:
            return float("nan")
        if sign=="UP":
            return sum(ts[i+1]["mid_delta"]<0 for i in idxs)/len(idxs)
        return sum(ts[i+1]["mid_delta"]>0 for i in idxs)/len(idxs)

    return {
        "up_reversal_rate": reversal_rate(up_idx,"UP"),
        "down_reversal_rate": reversal_rate(dn_idx,"DOWN"),
        "up_next_mid_mean": mean_next(up_idx,"mid_delta"),
        "down_next_mid_mean": mean_next(dn_idx,"mid_delta"),
        "up_next_spread_mean": mean_next(up_idx,"spread_delta"),
        "down_next_spread_mean": mean_next(dn_idx,"spread_delta"),
        "up_next_range_mean": mean_next(up_idx,"range_delta"),
        "down_next_range_mean": mean_next(dn_idx,"range_delta"),
    }


def percentile_rank(vals,x):
    return sum(v <= x for v in vals)/len(vals)


def two_sided_tail(vals,x):
    mu=statistics.mean(vals)
    dev=abs(x-mu)
    return sum(abs(v-mu)>=dev for v in vals)/len(vals)


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    actual=[draw(r) for _,r in df.iterrows()]
    obs=metrics(actual)

    rng=random.Random(SEED)
    sim_rows=[]
    for s in range(SIMS):
        sim=[sorted(rng.sample(range(1,38),7)) for _ in range(len(actual))]
        m=metrics(sim)
        m["sim"]=s
        sim_rows.append(m)

    sims=pd.DataFrame(sim_rows)

    print("=== LOTO7 MID-SHAPE RANDOM CONTROL v0 ===")
    print("Null: each draw is an independent uniform 7-of-37 sample.")
    print(f"draws={len(actual)} sims={SIMS} seed={SEED}")
    print("No fitted parameters or threshold search.")
    print()

    rows=[]
    for k,v in obs.items():
        vals=sims[k].tolist()
        mu=statistics.mean(vals)
        sd=statistics.pstdev(vals)
        z=(v-mu)/sd if sd else float("nan")
        pct=percentile_rank(vals,v)
        p2=two_sided_tail(vals,v)
        rows.append({
            "metric":k,
            "observed":v,
            "random_mean":mu,
            "random_sd":sd,
            "z":z,
            "percentile":pct,
            "two_sided_empirical_p":p2,
        })
        print(
            f"{k}: observed={v:+.6f} random_mean={mu:+.6f} "
            f"sd={sd:.6f} z={z:+.3f} percentile={pct:.3%} p2={p2:.3%}"
        )

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
