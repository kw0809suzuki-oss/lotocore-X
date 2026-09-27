from __future__ import annotations
# Reuse the v0 probe logic, adding subgroup summaries only.
from probe_loto7_fg_nearmiss_scorefield_pokopoko_v0 import *
import pandas as pd

def run_rows():
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
            if a in regen: continue
            ds=[(abs(a-r),r) for r in regen]
            md=min(d for d,_ in ds)
            if md not in (1,2): continue
            tied=[r for d,r in ds if d==md]
            chosen=max(tied,key=lambda n:(comp[n]["score"],-n))
            rows.append({
                "round":int(df.iloc[i]["round"]),
                "actual":a,"chosen":chosen,"distance":md,
                "in_support":int(a in support),
                "in_core":int(a in core),
                "in_boundary":int(a in boundary),
                "actual_rank":ranks[a],"chosen_rank":ranks[chosen],
                "score_delta":comp[a]["score"]-comp[chosen]["score"],
                "long_delta":comp[a]["long"]-comp[chosen]["long"],
                "recent_delta":comp[a]["recent"]-comp[chosen]["recent"],
                "gap_delta":comp[a]["gap"]-comp[chosen]["gap"],
            })
    return pd.DataFrame(rows)

def main():
    r=run_rows()
    print("=== FG NEAR-MISS POKOPOKO SUBGROUP v1 ===")
    for label,p in [("INSIDE_SUPPORT13",r[r.in_support==1]),("OUTSIDE_SUPPORT13",r[r.in_support==0])]:
        print()
        print(f"--- {label} ---")
        print(f"n={len(p)} share={len(p)/len(r):.6f}")
        print(f"actual rank median={p.actual_rank.median():.3f} mean={p.actual_rank.mean():.3f}")
        print(f"chosen rank median={p.chosen_rank.median():.3f} mean={p.chosen_rank.mean():.3f}")
        print(f"actual score > chosen={(p.score_delta>0).mean():.6f} <={(p.score_delta<=0).mean():.6f} mean_delta={p.score_delta.mean():+.6f}")
        print(f"long actual win={(p.long_delta>0).mean():.6f} recent actual win={(p.recent_delta>0).mean():.6f} gap actual win={(p.gap_delta>0).mean():.6f}")
        print(f"rank buckets actual: 1-7={(p.actual_rank<=7).mean():.6f} 8-13={((p.actual_rank>=8)&(p.actual_rank<=13)).mean():.6f} 14-20={((p.actual_rank>=14)&(p.actual_rank<=20)).mean():.6f} 21-30={((p.actual_rank>=21)&(p.actual_rank<=30)).mean():.6f} 31-37={(p.actual_rank>=31).mean():.6f}")
        print(f"distance1={(p.distance==1).mean():.6f} distance2={(p.distance==2).mean():.6f}")

    inside=r[r.in_support==1]
    print()
    print("INSIDE SUPPORT: WHAT GOT OMITTED?")
    print(f"actual was Core7={inside.in_core.mean():.6f}")
    print(f"actual was Boundary6={inside.in_boundary.mean():.6f}")
    print(f"shape chose neighbor despite actual having higher total score={(inside.score_delta>0).mean():.6f}")
    print(f"shape chose neighbor despite actual having higher recent={(inside.recent_delta>0).mean():.6f}")
    print()
    outside=r[r.in_support==0]
    print("OUTSIDE SUPPORT: HOW FAR DOWN THE SCORE FIELD?")
    for cutoff in (14,16,18,20,25,30):
        print(f"actual rank <= {cutoff}: {(outside.actual_rank<=cutoff).mean():.6f}")
    print()
    print("DIRECTION")
    print(f"actual numerically above chosen={(r.actual>r.chosen).mean():.6f} below={(r.actual<r.chosen).mean():.6f}")

if __name__=="__main__":
    main()
