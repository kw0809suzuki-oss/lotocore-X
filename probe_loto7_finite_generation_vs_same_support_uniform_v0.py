from __future__ import annotations

from itertools import combinations
from pathlib import Path
import math
import pandas as pd

import lotocore

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_finite_generation_vs_same_support_uniform_v0.csv")
WINDOW=100
BOUNDARY_K=6


def actual_set(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}


def ranked_from_snapshot(snap):
    ranks={int(n):int(r) for n,r in snap["ranks"].items()}
    return sorted(ranks,key=lambda n:(ranks[n],n))


def weighted_state(scores):
    vals={int(n):max(0.0,float(v)) for n,v in scores.items()}
    total=sum(vals.values())
    p={n:vals.get(n,0.0)/total for n in range(1,38)}
    c=sum(n*p[n] for n in p)
    v=sum((n-c)**2*p[n] for n in p)
    return c,math.sqrt(v)


def set_state(nums):
    vals=list(nums)
    c=sum(vals)/len(vals)
    v=sum((n-c)**2 for n in vals)/len(vals)
    return c,math.sqrt(v)


def regenerate(support,snap):
    scores={int(n):float(v) for n,v in snap["scores"].items()}
    tc,ts=weighted_state(snap["scores"])
    best=None
    for combo in combinations(support,7):
        c,s=set_state(combo)
        err=(abs(c-tc)+abs(s-ts))/max(ts,1e-12)
        mass=sum(scores[n] for n in combo)
        key=(err,-mass,combo)
        if best is None or key<best[0]:
            best=(key,tuple(sorted(combo)))
    return best[1]


def nearest_mean_distance(actual,nums):
    vals=list(nums)
    return sum(min(abs(a-n) for n in vals) for a in actual)/7


def bands(actual,nums):
    vals=list(nums)
    ds=[min(abs(a-n) for n in vals) for a in actual]
    return {
        "exact":sum(d==0 for d in ds),
        "within1":sum(d<=1 for d in ds),
        "within2":sum(d<=2 for d in ds),
        "within3":sum(d<=3 for d in ds),
        "d4plus":sum(d>=4 for d in ds),
        "mean_distance":sum(ds)/7,
    }


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]

    for i in range(WINDOW,len(df)):
        history=df.iloc[i-WINDOW:i]
        target=df.iloc[i]
        actual=actual_set(target)

        core=tuple(sorted(lotocore.predict(history).numbers))
        snap=lotocore.score_snapshot(history)
        order=ranked_from_snapshot(snap)
        boundary=tuple(n for n in order if n not in set(core))[:BOUNDARY_K]
        support=tuple(sorted(set(core)|set(boundary)))
        regen=regenerate(support,snap)

        rb=bands(actual,regen)

        combos=list(combinations(support,7))
        metrics=[bands(actual,c) for c in combos]

        uniform={
            key:sum(m[key] for m in metrics)/len(metrics)
            for key in ("exact","within1","within2","within3","d4plus","mean_distance")
        }

        better_dist=sum(m["mean_distance"]>rb["mean_distance"] for m in metrics)/len(metrics)
        equal_dist=sum(abs(m["mean_distance"]-rb["mean_distance"])<1e-12 for m in metrics)/len(metrics)
        better_exact=sum(m["exact"]<rb["exact"] for m in metrics)/len(metrics)
        equal_exact=sum(m["exact"]==rb["exact"] for m in metrics)/len(metrics)

        rows.append({
            "target_round":int(target["round"]),
            "date":target.get("date",""),
            "support":"-".join(f"{n:02d}" for n in support),
            "regen7":"-".join(f"{n:02d}" for n in regen),
            "actual":"-".join(f"{n:02d}" for n in sorted(actual)),
            "regen_exact":rb["exact"],
            "uniform_exact":uniform["exact"],
            "delta_exact":rb["exact"]-uniform["exact"],
            "regen_within1":rb["within1"],
            "uniform_within1":uniform["within1"],
            "delta_within1":rb["within1"]-uniform["within1"],
            "regen_within2":rb["within2"],
            "uniform_within2":uniform["within2"],
            "delta_within2":rb["within2"]-uniform["within2"],
            "regen_within3":rb["within3"],
            "uniform_within3":uniform["within3"],
            "delta_within3":rb["within3"]-uniform["within3"],
            "regen_d4plus":rb["d4plus"],
            "uniform_d4plus":uniform["d4plus"],
            "delta_d4plus":rb["d4plus"]-uniform["d4plus"],
            "regen_mean_distance":rb["mean_distance"],
            "uniform_mean_distance":uniform["mean_distance"],
            "delta_mean_distance":rb["mean_distance"]-uniform["mean_distance"],
            "regen_distance_percentile_better_than":better_dist,
            "regen_distance_equal_fraction":equal_dist,
            "regen_exact_percentile_better_than":better_exact,
            "regen_exact_equal_fraction":equal_exact,
        })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 FINITE GENERATION vs SAME-SUPPORT UNIFORM v0 ===")
    print(f"targets={len(res)} support=13 choose7={math.comb(13,7)}")
    print("Same support13 in every comparison. Uniform baseline is exact average over all 1716 seven-number subsets.")
    print()
    for metric in ("exact","within1","within2","within3","d4plus","mean_distance"):
        r=res[f"regen_{metric}"].mean()
        u=res[f"uniform_{metric}"].mean()
        print(f"{metric}: regen={r:.6f} uniform={u:.6f} delta={r-u:+.6f}")
    print()
    print("ROUND-LEVEL COMPARISON")
    print(f"distance better/worse/same vs uniform mean = {(res.delta_mean_distance<0).mean():.6f} / {(res.delta_mean_distance>0).mean():.6f} / {(res.delta_mean_distance.abs()<1e-12).mean():.6f}")
    print(f"exact better/worse/same vs uniform mean = {(res.delta_exact>0).mean():.6f} / {(res.delta_exact<0).mean():.6f} / {(res.delta_exact.abs()<1e-12).mean():.6f}")
    print(f"mean fraction of same-support subsets with worse distance than regen={res.regen_distance_percentile_better_than.mean():.6f}")
    print()
    print("RECENT 20")
    cols=["target_round","regen_exact","uniform_exact","delta_exact","regen_within2","uniform_within2","delta_within2","regen_mean_distance","uniform_mean_distance","delta_mean_distance","support","regen7","actual"]
    print(res[cols].tail(20).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
