from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from probe_loto7_a_world_shape_aware_box_v0 import (
    DATA, LONG_WINDOW, RECENT_WINDOW, ALPHA,
    draws_from_df, frequency_distribution, mix_distribution,
    distribution_state, box_state, rank_numbers, choose_shape_aware,
)

OUT=Path("results/loto7_a_world_shape_turnover_v0.csv")


def replacement_count(a,b):
    return len(set(a)-set(b))


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=draws_from_df(df)
    states=[]

    for i in range(LONG_WINDOW,len(draws)):
        lp=frequency_distribution(draws[i-LONG_WINDOW:i])
        rp=frequency_distribution(draws[i-RECENT_WINDOW:i])
        p=mix_distribution(lp,rp)
        ranked=rank_numbers(p)
        top7=tuple(sorted(ranked[:7]))
        ac,asp=distribution_state(p)
        shape,swapped,_,_,_,_,sc,ss=choose_shape_aware(ranked,p,ac,asp)
        tc,ts=box_state(top7)
        states.append({
            "round":int(df.iloc[i]["round"]),
            "top7":top7,"shape":shape,
            "a_center":ac,"a_spread":asp,
            "top7_center":tc,"top7_spread":ts,
            "shape_center":sc,"shape_spread":ss,
            "swapped":swapped,
        })

    rows=[]
    for j in range(1,len(states)):
        p=states[j-1]; c=states[j]
        rec={
            "from_round":p["round"],"to_round":c["round"],
            "top7_replacements":replacement_count(p["top7"],c["top7"]),
            "shape_replacements":replacement_count(p["shape"],c["shape"]),
        }
        for axis in ("center","spread"):
            da=c[f"a_{axis}"]-p[f"a_{axis}"]
            dt=c[f"top7_{axis}"]-p[f"top7_{axis}"]
            ds=c[f"shape_{axis}"]-p[f"shape_{axis}"]
            rec[f"a_d_{axis}"]=da
            rec[f"top7_d_{axis}"]=dt
            rec[f"shape_d_{axis}"]=ds
            rec[f"top7_amp_{axis}"]=abs(dt)/abs(da) if abs(da)>1e-12 else None
            rec[f"shape_amp_{axis}"]=abs(ds)/abs(da) if abs(da)>1e-12 else None
        rows.append(rec)

    res=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 A-WORLD SHAPE TURNOVER v0 ===")
    print(f"transitions={len(res)} alpha={ALPHA:.2f}")
    print("No future target outcome is used.")
    print()
    print("BOX TURNOVER")
    for name in ("top7","shape"):
        r=res[f"{name}_replacements"]
        print(
            f"{name}: changed={(r>0).mean():.6f} "
            f"same={(r==0).mean():.6f} "
            f"mean_replacements={r.mean():.6f} "
            f"median={r.median():.6f} "
            f"dist={r.value_counts().sort_index().to_dict()}"
        )
    print()
    print("MOTION AMPLIFICATION |box delta| / |A delta|")
    for axis in ("center","spread"):
        for name in ("top7","shape"):
            v=res[f"{name}_amp_{axis}"].dropna()
            print(
                f"{axis} {name}: median={v.median():.6f} "
                f"mean={v.mean():.6f} p90={v.quantile(.90):.6f}"
            )
    print()
    print("ABSOLUTE BOX STEP SIZE")
    for axis in ("center","spread"):
        print(
            f"{axis}: A={res[f'a_d_{axis}'].abs().mean():.6f} "
            f"top7={res[f'top7_d_{axis}'].abs().mean():.6f} "
            f"shape={res[f'shape_d_{axis}'].abs().mean():.6f}"
        )
    print()
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
