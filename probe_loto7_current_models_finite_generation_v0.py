from __future__ import annotations

from pathlib import Path
import pandas as pd

import lotocore
import x_agent

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_current_models_finite_generation_v0.csv")
LATEST=Path("results/loto7_current_models_finite_generation_latest_v0.csv")
WINDOW=100
MAX_K=10
MODELS=("core","x")


def model_core(history, model):
    if model=="core":
        return tuple(sorted(lotocore.predict(history).numbers))
    if model=="x":
        return tuple(sorted(x_agent.predict(history, competition_gate=True).numbers))
    raise ValueError(model)


def model_snapshot(history, model):
    if model=="core":
        return lotocore.score_snapshot(history)
    if model=="x":
        return x_agent.score_snapshot(history, competition_gate=True)
    raise ValueError(model)


def ranked_from_snapshot(snapshot):
    ranks={int(n):int(r) for n,r in snapshot["ranks"].items()}
    return sorted(ranks, key=lambda n:(ranks[n],n))


def boundary_for(history, model, core, k):
    snap=model_snapshot(history,model)
    ranked=ranked_from_snapshot(snap)
    return tuple(n for n in ranked if n not in set(core))[:k]


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)

    states={m:[] for m in MODELS}
    for i in range(WINDOW,len(df)+1):
        history=df.iloc[i-WINDOW:i]
        for m in MODELS:
            core=model_core(history,m)
            snap=model_snapshot(history,m)
            ranked=ranked_from_snapshot(snap)
            noncore=tuple(n for n in ranked if n not in set(core))
            states[m].append({
                "history_end_round":int(df.iloc[i-1]["round"]),
                "core":core,
                "noncore":noncore,
                "snapshot":snap,
            })

    rows=[]
    for m in MODELS:
        s=states[m]
        for k in range(MAX_K+1):
            full=0
            total_covered=0
            total_outside=0
            total_entering=0
            entering_in_boundary=0
            changing=0
            transition_count=len(s)-1

            for j in range(transition_count):
                cur=s[j]; nxt=s[j+1]
                core=set(cur["core"])
                next_core=set(nxt["core"])
                boundary=set(cur["noncore"][:k])
                support=core|boundary

                covered=len(support & next_core)
                outside=7-covered
                entering=next_core-core

                full += int(covered==7)
                total_covered += covered
                total_outside += outside
                if entering:
                    changing += 1
                    total_entering += len(entering)
                    entering_in_boundary += len(entering & boundary)

            rows.append({
                "model":m,
                "boundary_k":k,
                "support_size":7+k,
                "transitions":transition_count,
                "full_reentry_rate":full/transition_count,
                "mean_next_core_covered":total_covered/transition_count,
                "mean_next_core_outside":total_outside/transition_count,
                "changing_transitions":changing,
                "changing_transition_rate":changing/transition_count,
                "total_entering":total_entering,
                "entering_captured_by_boundary":entering_in_boundary,
                "entering_capture_rate":entering_in_boundary/total_entering if total_entering else 0.0,
            })

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    latest_rows=[]
    for m in MODELS:
        latest=states[m][-1]
        core=latest["core"]
        snap=latest["snapshot"]
        scores={int(n):float(v) for n,v in snap["scores"].items()}
        ranks={int(n):int(v) for n,v in snap["ranks"].items()}
        boundary10=latest["noncore"][:10]
        for role, nums in (("core",core),("boundary",boundary10)):
            for n in nums:
                latest_rows.append({
                    "model":m,
                    "history_end_round":latest["history_end_round"],
                    "role":role,
                    "number":n,
                    "rank":ranks[n],
                    "score":scores[n],
                })
    latest_df=pd.DataFrame(latest_rows).sort_values(["model","role","rank"])
    latest_df.to_csv(LATEST,index=False)

    print("=== LOTO7 CURRENT MODELS FINITE GENERATION v0 ===")
    print(f"history_window={WINDOW} transitions={len(states['core'])-1} latest_history_end={states['core'][-1]['history_end_round']}")
    print("No lottery target outcome is used. Re-entry is model-State(t) -> model-State(t+1).")
    print()

    for m in MODELS:
        part=res[res.model==m]
        print(f"--- {m.upper()} ---")
        for _,r in part.iterrows():
            print(
                f"k={int(r.boundary_k):2d} support={int(r.support_size):2d} "
                f"full={r.full_reentry_rate:.6f} "
                f"covered={r.mean_next_core_covered:.4f}/7 "
                f"entering_capture={r.entering_capture_rate:.6f}"
            )
        print()

    print("=== LATEST FINITE REPRESENTATION ===")
    for m in MODELS:
        latest=states[m][-1]
        core=latest["core"]
        b5=latest["noncore"][:5]
        b6=latest["noncore"][:6]
        b10=latest["noncore"][:10]
        print(f"{m.upper()} history_end={latest['history_end_round']}")
        print("core7="+" ".join(f"{n:02d}" for n in core))
        print("boundary5="+" ".join(f"{n:02d}" for n in b5))
        print("boundary6="+" ".join(f"{n:02d}" for n in b6))
        print("boundary10="+" ".join(f"{n:02d}" for n in b10))
        if m=="x":
            st=latest["snapshot"].get("state",{})
            keys=("field_center_delta","field_spread_delta","shape_delta","gap_delta","spacing_strength","w_persist","w_reverse","w_gap","w_shape","w_geom")
            print("state="+" ".join(f"{k}={st.get(k)}" for k in keys))
        else:
            print("state="+str(latest["snapshot"].get("state",{})))
        print()

    print(f"saved -> {OUT}")
    print(f"saved -> {LATEST}")


if __name__=="__main__":
    main()
