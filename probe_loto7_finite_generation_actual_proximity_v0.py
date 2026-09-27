from __future__ import annotations

from pathlib import Path
from collections import Counter
import pandas as pd

import lotocore
import x_agent

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_finite_generation_actual_proximity_v0.csv")
SUMMARY=Path("results/loto7_finite_generation_actual_proximity_summary_v0.csv")
WINDOW=100
MAX_K=10
MODELS=("core","x")


def actual_set(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}


def predict_core(history,model):
    if model=="core":
        return tuple(sorted(lotocore.predict(history).numbers))
    if model=="x":
        return tuple(sorted(x_agent.predict(history,competition_gate=True).numbers))
    raise ValueError(model)


def snapshot(history,model):
    if model=="core":
        return lotocore.score_snapshot(history)
    if model=="x":
        return x_agent.score_snapshot(history,competition_gate=True)
    raise ValueError(model)


def ranked(snapshot):
    ranks={int(n):int(r) for n,r in snapshot["ranks"].items()}
    return sorted(ranks,key=lambda n:(ranks[n],n))


def nearest_mean_distance(actual,support):
    support=list(support)
    return sum(min(abs(a-s) for s in support) for a in actual)/len(actual)


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]

    for i in range(WINDOW,len(df)):
        history=df.iloc[i-WINDOW:i]
        target=df.iloc[i]
        actual=actual_set(target)

        for model in MODELS:
            core=set(predict_core(history,model))
            snap=snapshot(history,model)
            order=ranked(snap)
            noncore=[n for n in order if n not in core]

            for k in range(MAX_K+1):
                boundary=set(noncore[:k])
                support=core|boundary
                core_hits=len(core&actual)
                boundary_hits=len(boundary&actual)
                support_hits=len(support&actual)
                outside_hits=7-support_hits
                rows.append({
                    "target_round":int(target["round"]),
                    "date":target.get("date",""),
                    "model":model,
                    "boundary_k":k,
                    "support_size":len(support),
                    "core_hits":core_hits,
                    "boundary_hits":boundary_hits,
                    "support_hits":support_hits,
                    "outside_hits":outside_hits,
                    "nearest_mean_distance":nearest_mean_distance(actual,support),
                    "core":"-".join(f"{n:02d}" for n in sorted(core)),
                    "boundary":"-".join(f"{n:02d}" for n in noncore[:k]),
                    "actual":"-".join(f"{n:02d}" for n in sorted(actual)),
                })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    summary=[]
    n_targets=len(df)-WINDOW
    for model in MODELS:
        part=res[res.model==model]
        for k in range(MAX_K+1):
            p=part[part.boundary_k==k]
            size=7+k
            expected=7*size/37
            mean_hits=p.support_hits.mean()
            per_slot=p.support_hits.sum()/(len(p)*size)
            summary.append({
                "model":model,
                "boundary_k":k,
                "support_size":size,
                "targets":len(p),
                "mean_core_hits":p.core_hits.mean(),
                "mean_boundary_hits":p.boundary_hits.mean(),
                "mean_support_hits":mean_hits,
                "random_expected_mean_hits":expected,
                "mean_hit_lift_vs_random":mean_hits-expected,
                "per_slot_hit_rate":per_slot,
                "capture_2plus_rate":(p.support_hits>=2).mean(),
                "capture_3plus_rate":(p.support_hits>=3).mean(),
                "capture_4plus_rate":(p.support_hits>=4).mean(),
                "mean_nearest_distance":p.nearest_mean_distance.mean(),
                "median_nearest_distance":p.nearest_mean_distance.median(),
                "full_7_capture_rate":(p.support_hits==7).mean(),
            })

    sm=pd.DataFrame(summary)
    sm.to_csv(SUMMARY,index=False)

    print("=== LOTO7 FINITE GENERATION -> ACTUAL PROXIMITY v0 ===")
    print(f"targets={n_targets} rounds={int(df.iloc[WINDOW]['round'])}..{int(df.iloc[-1]['round'])} window={WINDOW}")
    print("Historical walk-forward: representation is built before each target draw.")
    print()
    for model in MODELS:
        print(f"--- {model.upper()} ---")
        p=sm[sm.model==model]
        for _,r in p.iterrows():
            print(
                f"k={int(r.boundary_k):2d} size={int(r.support_size):2d} "
                f"mean_hits={r.mean_support_hits:.4f} "
                f"random_exp={r.random_expected_mean_hits:.4f} "
                f"lift={r.mean_hit_lift_vs_random:+.4f} "
                f"3+={r.capture_3plus_rate:.4f} "
                f"4+={r.capture_4plus_rate:.4f} "
                f"near_dist={r.mean_nearest_distance:.4f}"
            )
        print()

    print("=== WHERE ACTUAL NUMBERS LAND ===")
    for model in MODELS:
        p=res[(res.model==model)&(res.boundary_k==6)]
        print(
            f"{model.upper()} k6: core_hits_mean={p.core_hits.mean():.4f} "
            f"boundary_hits_mean={p.boundary_hits.mean():.4f} "
            f"outside_mean={p.outside_hits.mean():.4f}"
        )
    print()
    print("=== RECENT 20 TARGETS, k=6 ===")
    cols=["target_round","model","core_hits","boundary_hits","support_hits","outside_hits","nearest_mean_distance","core","boundary","actual"]
    print(res[res.boundary_k==6][cols].tail(40).to_string(index=False))
    print()
    print(f"saved -> {OUT}")
    print(f"saved -> {SUMMARY}")


if __name__=="__main__":
    main()
