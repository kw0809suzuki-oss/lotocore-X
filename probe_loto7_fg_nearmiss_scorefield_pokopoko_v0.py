from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import math
import pandas as pd
import lotocore

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_fg_nearmiss_scorefield_pokopoko_v0.csv")
WINDOW=100
BOUNDARY_K=6
NUMBERS=range(1,38)


def draws(history):
    cols=[f"n{i}" for i in range(1,8)]
    return [sorted(map(int,row)) for row in history[cols].to_numpy()]


def components(history):
    ds=draws(history)
    long=ds[-100:]
    recent=ds[-20:]
    fl=Counter(n for d in long for n in d)
    fr=Counter(n for d in recent for n in d)
    gap={n:len(ds) for n in NUMBERS}
    for g,d in enumerate(reversed(ds)):
        for n in d:
            if gap[n]==len(ds):
                gap[n]=g
    out={}
    for n in NUMBERS:
        c_long=0.60*fl[n]/max(1,len(long))
        c_recent=0.25*fr[n]/max(1,len(recent))
        c_gap=0.15*min(gap[n],12)/12.0
        out[n]={
            "long":c_long,
            "recent":c_recent,
            "gap":c_gap,
            "score":c_long+c_recent+c_gap,
            "freq_long":fl[n],
            "freq_recent":fr[n],
            "raw_gap":gap[n],
        }
    return out


def ranked(snapshot):
    rr={int(n):int(r) for n,r in snapshot["ranks"].items()}
    return sorted(rr,key=lambda n:(rr[n],n)),rr


def weighted_state(scores):
    vals={int(n):max(0.0,float(v)) for n,v in scores.items()}
    total=sum(vals.values())
    p={n:vals.get(n,0.0)/total for n in NUMBERS}
    c=sum(n*p[n] for n in p)
    v=sum((n-c)**2*p[n] for n in p)
    return c,math.sqrt(v)


def set_state(nums):
    vals=list(nums)
    c=sum(vals)/len(vals)
    v=sum((n-c)**2 for n in vals)/len(vals)
    return c,math.sqrt(v)


def regen7(core,boundary,snap):
    support=tuple(sorted(set(core)|set(boundary)))
    scores={int(n):float(v) for n,v in snap["scores"].items()}
    tc,ts=weighted_state(snap["scores"])
    best=None
    for combo in combinations(support,7):
        combo=tuple(sorted(combo))
        c,s=set_state(combo)
        err=(abs(c-tc)+abs(s-ts))/max(ts,1e-12)
        mass=sum(scores[n] for n in combo)
        key=(err,-mass,combo)
        if best is None or key<best[0]:
            best=(key,combo)
    return best[1],support


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    rows=[]

    for i in range(WINDOW,len(df)):
        history=df.iloc[i-WINDOW:i]
        actual={int(df.iloc[i][f"n{j}"]) for j in range(1,8)}

        core=tuple(sorted(lotocore.predict(history).numbers))
        snap=lotocore.score_snapshot(history)
        order,ranks=ranked(snap)
        boundary=tuple(n for n in order if n not in set(core))[:BOUNDARY_K]
        regen,support=regen7(core,boundary,snap)
        comp=components(history)

        for a in sorted(actual):
            if a in regen:
                continue
            ds=[(abs(a-r),r) for r in regen]
            min_d=min(d for d,_ in ds)
            if min_d not in (1,2):
                continue
            tied=[r for d,r in ds if d==min_d]
            chosen=max(tied,key=lambda n:(comp[n]["score"],-n))

            rows.append({
                "target_round":int(df.iloc[i]["round"]),
                "actual":a,
                "nearest_selected":chosen,
                "distance":min_d,
                "direction":"actual_higher" if a>chosen else "actual_lower",
                "actual_in_support13":int(a in support),
                "actual_in_core7":int(a in core),
                "actual_in_boundary6":int(a in boundary),
                "selected_in_core7":int(chosen in core),
                "selected_in_boundary6":int(chosen in boundary),
                "actual_score":comp[a]["score"],
                "selected_score":comp[chosen]["score"],
                "score_delta_actual_minus_selected":comp[a]["score"]-comp[chosen]["score"],
                "actual_rank":ranks[a],
                "selected_rank":ranks[chosen],
                "rank_delta_actual_minus_selected":ranks[a]-ranks[chosen],
                "actual_long":comp[a]["long"],
                "selected_long":comp[chosen]["long"],
                "long_delta":comp[a]["long"]-comp[chosen]["long"],
                "actual_recent":comp[a]["recent"],
                "selected_recent":comp[chosen]["recent"],
                "recent_delta":comp[a]["recent"]-comp[chosen]["recent"],
                "actual_gap_component":comp[a]["gap"],
                "selected_gap_component":comp[chosen]["gap"],
                "gap_delta":comp[a]["gap"]-comp[chosen]["gap"],
                "actual_raw_gap":comp[a]["raw_gap"],
                "selected_raw_gap":comp[chosen]["raw_gap"],
                "core":"-".join(f"{n:02d}" for n in core),
                "boundary6":"-".join(f"{n:02d}" for n in boundary),
                "regen7":"-".join(f"{n:02d}" for n in regen),
            })

    r=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    r.to_csv(OUT,index=False)

    print("=== FG NEAR-MISS SCORE FIELD POKOPOKO v0 ===")
    print(f"near_miss_pairs={len(r)}")
    print(f"distance1={(r.distance==1).sum()} distance2={(r.distance==2).sum()}")
    print()
    print("WHERE WAS THE ACTUAL NUMBER?")
    print(f"in support13={r.actual_in_support13.mean():.6f}")
    print(f"in core7={r.actual_in_core7.mean():.6f}")
    print(f"in boundary6={r.actual_in_boundary6.mean():.6f}")
    print()
    print("TOTAL SCORE")
    print(f"actual score > selected={ (r.score_delta_actual_minus_selected>0).mean():.6f}")
    print(f"actual score = selected={ (r.score_delta_actual_minus_selected.abs()<1e-12).mean():.6f}")
    print(f"actual score < selected={ (r.score_delta_actual_minus_selected<0).mean():.6f}")
    print(f"mean score delta actual-selected={r.score_delta_actual_minus_selected.mean():+.6f}")
    print(f"median rank delta actual-selected={r.rank_delta_actual_minus_selected.median():+.3f}  mean={r.rank_delta_actual_minus_selected.mean():+.3f}")
    print()
    print("COMPONENT WIN RATE: actual > selected")
    for col in ("long_delta","recent_delta","gap_delta"):
        print(f"{col}={(r[col]>0).mean():.6f} equal={(r[col].abs()<1e-12).mean():.6f} mean_delta={r[col].mean():+.6f}")
    print()
    print("BY DISTANCE")
    for d in (1,2):
        p=r[r.distance==d]
        print(f"d={d} n={len(p)} support={p.actual_in_support13.mean():.6f} score_win={(p.score_delta_actual_minus_selected>0).mean():.6f} recent_win={(p.recent_delta>0).mean():.6f} gap_win={(p.gap_delta>0).mean():.6f}")
    print()
    print("RECENT 40")
    cols=["target_round","actual","nearest_selected","distance","actual_in_support13","actual_rank","selected_rank","score_delta_actual_minus_selected","long_delta","recent_delta","gap_delta","regen7"]
    print(r[cols].tail(40).to_string(index=False))
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
