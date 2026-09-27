from __future__ import annotations

from itertools import combinations
from pathlib import Path
import math
import pandas as pd

import lotocore
import x_agent

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_regenerated7_distance_bands_v0.csv")
WINDOW=100
BOUNDARY_K=6
MODELS=("core","x")


def actual_set(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}


def model_core(history,model):
    if model=="core":
        return tuple(sorted(lotocore.predict(history).numbers))
    return tuple(sorted(x_agent.predict(history,competition_gate=True).numbers))


def snapshot(history,model):
    if model=="core":
        return lotocore.score_snapshot(history)
    return x_agent.score_snapshot(history,competition_gate=True)


def ranked(snap):
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


def regenerate(core,boundary,snap):
    support=tuple(sorted(set(core)|set(boundary)))
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


def distances(actual,nums):
    vals=list(nums)
    return [min(abs(a-n) for n in vals) for a in actual]


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]
    for i in range(WINDOW,len(df)):
        history=df.iloc[i-WINDOW:i]
        actual=actual_set(df.iloc[i])
        for model in MODELS:
            base=model_core(history,model)
            snap=snapshot(history,model)
            order=ranked(snap)
            boundary=tuple(n for n in order if n not in set(base))[:BOUNDARY_K]
            regen=regenerate(base,boundary,snap)

            for variant,nums in (("base",base),("regen",regen)):
                ds=distances(actual,nums)
                rows.append({
                    "target_round":int(df.iloc[i]["round"]),
                    "model":model,
                    "variant":variant,
                    "mean_distance":sum(ds)/7,
                    "exact_count":sum(d==0 for d in ds),
                    "within1_count":sum(d<=1 for d in ds),
                    "within2_count":sum(d<=2 for d in ds),
                    "within3_count":sum(d<=3 for d in ds),
                    "d0":sum(d==0 for d in ds),
                    "d1":sum(d==1 for d in ds),
                    "d2":sum(d==2 for d in ds),
                    "d3":sum(d==3 for d in ds),
                    "d4plus":sum(d>=4 for d in ds),
                })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 REGENERATED7 DISTANCE BANDS v0 ===")
    print(f"targets={len(df)-WINDOW} boundary={BOUNDARY_K}")
    print("Each actual number is measured to the nearest number in the 7-number output.")
    print()

    for model in MODELS:
        print(f"--- {model.upper()} ---")
        for variant in ("base","regen"):
            p=res[(res.model==model)&(res.variant==variant)]
            print(
                f"{variant}: mean_distance={p.mean_distance.mean():.6f} "
                f"exact={p.exact_count.mean():.6f}/7 "
                f"within1={p.within1_count.mean():.6f}/7 "
                f"within2={p.within2_count.mean():.6f}/7 "
                f"within3={p.within3_count.mean():.6f}/7 "
                f"d4plus={p.d4plus.mean():.6f}/7"
            )
        b=res[(res.model==model)&(res.variant=="base")].set_index("target_round")
        g=res[(res.model==model)&(res.variant=="regen")].set_index("target_round")
        print(
            f"delta regen-base: "
            f"within1={(g.within1_count-b.within1_count).mean():+.6f} "
            f"within2={(g.within2_count-b.within2_count).mean():+.6f} "
            f"within3={(g.within3_count-b.within3_count).mean():+.6f} "
            f"d4plus={(g.d4plus-b.d4plus).mean():+.6f}"
        )
        print()

    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
