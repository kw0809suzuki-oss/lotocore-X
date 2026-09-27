from __future__ import annotations

from itertools import combinations
from pathlib import Path
import math
import pandas as pd

import lotocore
import x_agent

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_finite_generation_regenerated7_v0.csv")
SUMMARY=Path("results/loto7_finite_generation_regenerated7_summary_v0.csv")
WINDOW=100
BOUNDARY_K=6
MODELS=("core","x")


def actual_set(row):
    return {int(row[f"n{i}"]) for i in range(1,8)}


def model_core(history,model):
    if model=="core":
        return tuple(sorted(lotocore.predict(history).numbers))
    if model=="x":
        return tuple(sorted(x_agent.predict(history,competition_gate=True).numbers))
    raise ValueError(model)


def model_snapshot(history,model):
    if model=="core":
        return lotocore.score_snapshot(history)
    if model=="x":
        return x_agent.score_snapshot(history,competition_gate=True)
    raise ValueError(model)


def ranked_from_snapshot(snap):
    ranks={int(n):int(r) for n,r in snap["ranks"].items()}
    return sorted(ranks,key=lambda n:(ranks[n],n))


def weighted_state(scores):
    vals={int(n):max(0.0,float(v)) for n,v in scores.items()}
    total=sum(vals.values())
    if total<=0:
        p={n:1/37 for n in range(1,38)}
    else:
        p={n:vals.get(n,0.0)/total for n in range(1,38)}
    center=sum(n*p[n] for n in p)
    variance=sum((n-center)**2*p[n] for n in p)
    return center,math.sqrt(variance)


def set_state(nums):
    vals=list(nums)
    center=sum(vals)/len(vals)
    variance=sum((n-center)**2 for n in vals)/len(vals)
    return center,math.sqrt(variance)


def regenerate7(core,boundary,snap):
    support=tuple(sorted(set(core)|set(boundary)))
    score={int(n):float(v) for n,v in snap["scores"].items()}
    target_center,target_spread=weighted_state(snap["scores"])

    support_scores=[score[n] for n in support]
    score_min=min(support_scores)
    score_max=max(support_scores)
    score_span=max(score_max-score_min,1e-12)

    candidates=[]
    for combo in combinations(support,7):
        c,s=set_state(combo)
        ce=abs(c-target_center)
        se=abs(s-target_spread)

        # Primary objective: preserve current A-state shape.
        # Normalize both errors by target spread so center/spread are comparable.
        shape_error=(ce+se)/max(target_spread,1e-12)

        # Secondary objective only: among equally shaped sets, retain model score mass.
        mass=sum(score[n] for n in combo)
        candidates.append((shape_error,-mass,combo,ce,se,mass,c,s))

    best=min(candidates,key=lambda x:(x[0],x[1],x[2]))
    _,_,combo,ce,se,mass,c,s=best
    return tuple(sorted(combo)),{
        "target_center":target_center,
        "target_spread":target_spread,
        "regen_center":c,
        "regen_spread":s,
        "center_error":ce,
        "spread_error":se,
        "score_mass":mass,
        "support_size":len(support),
    }


def mean_nearest_distance(actual,nums):
    vals=list(nums)
    return sum(min(abs(a-n) for n in vals) for a in actual)/len(actual)


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]

    for i in range(WINDOW,len(df)):
        history=df.iloc[i-WINDOW:i]
        target=df.iloc[i]
        actual=actual_set(target)

        for model in MODELS:
            base=tuple(sorted(model_core(history,model)))
            snap=model_snapshot(history,model)
            ranked=ranked_from_snapshot(snap)
            boundary=tuple(n for n in ranked if n not in set(base))[:BOUNDARY_K]
            regen,meta=regenerate7(base,boundary,snap)

            base_hits=len(set(base)&actual)
            regen_hits=len(set(regen)&actual)

            rows.append({
                "target_round":int(target["round"]),
                "date":target.get("date",""),
                "model":model,
                "base_hits":base_hits,
                "regen_hits":regen_hits,
                "hit_delta":regen_hits-base_hits,
                "base_nearest_distance":mean_nearest_distance(actual,base),
                "regen_nearest_distance":mean_nearest_distance(actual,regen),
                "distance_delta":mean_nearest_distance(actual,regen)-mean_nearest_distance(actual,base),
                "overlap_base_regen":len(set(base)&set(regen)),
                "base":"-".join(f"{n:02d}" for n in base),
                "boundary6":"-".join(f"{n:02d}" for n in boundary),
                "regen7":"-".join(f"{n:02d}" for n in regen),
                "actual":"-".join(f"{n:02d}" for n in sorted(actual)),
                **meta,
            })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    summary=[]
    random_expected=49/37
    for model in MODELS:
        p=res[res.model==model]
        summary.append({
            "model":model,
            "targets":len(p),
            "base_mean_hits":p.base_hits.mean(),
            "regen_mean_hits":p.regen_hits.mean(),
            "mean_hit_delta":p.hit_delta.mean(),
            "base_3plus_rate":(p.base_hits>=3).mean(),
            "regen_3plus_rate":(p.regen_hits>=3).mean(),
            "base_4plus_rate":(p.base_hits>=4).mean(),
            "regen_4plus_rate":(p.regen_hits>=4).mean(),
            "improved_rounds":int((p.hit_delta>0).sum()),
            "worsened_rounds":int((p.hit_delta<0).sum()),
            "same_rounds":int((p.hit_delta==0).sum()),
            "base_mean_nearest_distance":p.base_nearest_distance.mean(),
            "regen_mean_nearest_distance":p.regen_nearest_distance.mean(),
            "mean_overlap_base_regen":p.overlap_base_regen.mean(),
            "regen_lift_vs_uniform7":p.regen_hits.mean()-random_expected,
        })

    sm=pd.DataFrame(summary)
    sm.to_csv(SUMMARY,index=False)

    print("=== LOTO7 FINITE GENERATION -> REGENERATED 7 v0 ===")
    print(f"targets={len(df)-WINDOW} rounds={int(df.iloc[WINDOW]['round'])}..{int(df.iloc[-1]['round'])} boundary={BOUNDARY_K}")
    print("Historical walk-forward. Actual target numbers are NOT used to select regenerated7.")
    print("Regenerator: support13 -> choose 7 minimizing current score-state center+spread error; score mass is tie-break only.")
    print()

    for _,r in sm.iterrows():
        print(f"--- {r['model'].upper()} ---")
        print(f"base mean hits={r.base_mean_hits:.6f}")
        print(f"regen mean hits={r.regen_mean_hits:.6f}")
        print(f"delta={r.mean_hit_delta:+.6f}")
        print(f"uniform7 expected={random_expected:.6f} regen lift={r.regen_lift_vs_uniform7:+.6f}")
        print(f"3+: base={r.base_3plus_rate:.6f} regen={r.regen_3plus_rate:.6f}")
        print(f"4+: base={r.base_4plus_rate:.6f} regen={r.regen_4plus_rate:.6f}")
        print(f"rounds improve/worse/same={int(r.improved_rounds)}/{int(r.worsened_rounds)}/{int(r.same_rounds)}")
        print(f"nearest distance: base={r.base_mean_nearest_distance:.6f} regen={r.regen_mean_nearest_distance:.6f}")
        print(f"mean base-regenerated overlap={r.mean_overlap_base_regen:.6f}/7")
        print()

    print("=== RECENT 20 TARGETS ===")
    cols=["target_round","model","base_hits","regen_hits","hit_delta","base_nearest_distance","regen_nearest_distance","base","boundary6","regen7","actual"]
    print(res[cols].tail(40).to_string(index=False))
    print()
    print(f"saved -> {OUT}")
    print(f"saved -> {SUMMARY}")


if __name__=="__main__":
    main()
