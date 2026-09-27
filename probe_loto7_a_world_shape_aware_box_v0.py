from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import pandas as pd

DATA = Path("data/loto7.csv")
OUT = Path("results/loto7_a_world_shape_aware_box_v0.csv")
NUMBERS = list(range(1, 38))
LONG_WINDOW = 100
RECENT_WINDOW = 20
ALPHA = 0.72
BOX_SIZE = 7
POOL_SIZE = 10
AXES = ("center","spread")


def draws_from_df(df):
    return [[int(row[f"n{i}"]) for i in range(1,8)] for _,row in df.iterrows()]


def frequency_distribution(draws):
    c=Counter(n for d in draws for n in d)
    total=sum(c.values())
    return {n:c[n]/total for n in NUMBERS}


def mix_distribution(lp,rp):
    raw={n:ALPHA*lp[n]+(1-ALPHA)*rp[n] for n in NUMBERS}
    total=sum(raw.values())
    return {n:raw[n]/total for n in NUMBERS}


def distribution_state(p):
    center=sum(n*p[n] for n in NUMBERS)
    var=sum((n-center)**2*p[n] for n in NUMBERS)
    return center,math.sqrt(var)


def box_state(box):
    vals=list(box)
    center=sum(vals)/len(vals)
    var=sum((n-center)**2 for n in vals)/len(vals)
    return center,math.sqrt(var)


def box_mass(box,p):
    return sum(p[n] for n in box)


def rank_numbers(p):
    return sorted(NUMBERS,key=lambda n:(-p[n],n))


def shape_error(box,a_center,a_spread):
    c,s=box_state(box)
    return abs(c-a_center),abs(s-a_spread),c,s


def choose_shape_aware(ranked,p,a_center,a_spread):
    top7=tuple(sorted(ranked[:7]))
    bce,bse,_,_=shape_error(top7,a_center,a_spread)
    incoming=ranked[7:10]
    choices=[]
    for out_num in ranked[:7]:
        for in_num in incoming:
            cand=tuple(sorted((set(top7)-{out_num})|{in_num}))
            ce,se,c,s=shape_error(cand,a_center,a_spread)
            if ce < bce-1e-12 and se < bse-1e-12:
                choices.append((ce+se, box_mass(top7,p)-box_mass(cand,p), out_num,in_num,cand,ce,se,c,s))
    if not choices:
        ce,se,c,s=shape_error(top7,a_center,a_spread)
        return top7,False,None,None,ce,se,c,s
    best=min(choices,key=lambda x:(x[0],x[1],x[2],x[3]))
    _,_,out_num,in_num,cand,ce,se,c,s=best
    return cand,True,out_num,in_num,ce,se,c,s


def sign(x,eps=1e-12):
    return 1 if x>eps else (-1 if x<-eps else 0)


def main():
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    draws=draws_from_df(df)
    rows=[]

    for i in range(LONG_WINDOW,len(draws)):
        lp=frequency_distribution(draws[i-LONG_WINDOW:i])
        rp=frequency_distribution(draws[i-RECENT_WINDOW:i])
        p=mix_distribution(lp,rp)
        ranked=rank_numbers(p)
        top7=tuple(sorted(ranked[:7]))
        ac,asp=distribution_state(p)

        tce,tse,tc,ts=shape_error(top7,ac,asp)
        cand,swapped,out_num,in_num,sce,sse,sc,ss=choose_shape_aware(ranked,p,ac,asp)

        tm=box_mass(top7,p)
        sm=box_mass(cand,p)
        rows.append({
            "round":int(df.iloc[i]["round"]),
            "date":df.iloc[i].get("date",""),
            "a_center":ac,"a_spread":asp,
            "top7":"-".join(f"{n:02d}" for n in top7),
            "shape_box":"-".join(f"{n:02d}" for n in cand),
            "swapped":swapped,"out":out_num,"in":in_num,
            "top7_center":tc,"top7_spread":ts,
            "shape_center":sc,"shape_spread":ss,
            "top7_center_error":tce,"top7_spread_error":tse,
            "shape_center_error":sce,"shape_spread_error":sse,
            "top7_mass":tm,"shape_mass":sm,
            "mass_loss_ratio":(tm-sm)/tm if tm else 0.0,
        })

    res=pd.DataFrame(rows)

    # Motion comparison across consecutive A-world states.
    motion=[]
    for j in range(1,len(res)):
        prev=res.iloc[j-1]; cur=res.iloc[j]
        rec={"to_round":int(cur["round"])}
        for axis in AXES:
            da=float(cur[f"a_{axis}"]-prev[f"a_{axis}"])
            dt=float(cur[f"top7_{axis}"]-prev[f"top7_{axis}"])
            ds=float(cur[f"shape_{axis}"]-prev[f"shape_{axis}"])
            sa,st,ss=sign(da),sign(dt),sign(ds)
            rec[f"top7_match_{axis}"]=sa==st
            rec[f"shape_match_{axis}"]=sa==ss
            rec[f"a_d_{axis}"]=da
            rec[f"top7_d_{axis}"]=dt
            rec[f"shape_d_{axis}"]=ds
        motion.append(rec)
    mot=pd.DataFrame(motion)

    OUT.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT,index=False)

    print("=== LOTO7 A-WORLD SHAPE-AWARE BOX v0 ===")
    print(f"states={len(res)} transitions={len(mot)} alpha={ALPHA:.2f} pool=top{POOL_SIZE}")
    print("No future target outcome is used.")
    print()
    print("STATIC STATE ERROR")
    print(f"center mean abs error: top7={res['top7_center_error'].mean():.6f} shape={res['shape_center_error'].mean():.6f}")
    print(f"spread mean abs error: top7={res['top7_spread_error'].mean():.6f} shape={res['shape_spread_error'].mean():.6f}")
    print(f"center median abs error: top7={res['top7_center_error'].median():.6f} shape={res['shape_center_error'].median():.6f}")
    print(f"spread median abs error: top7={res['top7_spread_error'].median():.6f} shape={res['shape_spread_error'].median():.6f}")
    print()
    print("COMPRESSION MASS")
    print(f"top7 mean mass={res['top7_mass'].mean():.6f}")
    print(f"shape mean mass={res['shape_mass'].mean():.6f}")
    print(f"mean mass loss ratio={res['mass_loss_ratio'].mean():.6f}")
    print(f"median mass loss ratio={res['mass_loss_ratio'].median():.6f}")
    print(f"swapped states={int(res['swapped'].sum())}/{len(res)} = {res['swapped'].mean():.6f}")
    print()
    print("MOTION DIRECTION AGREEMENT")
    for axis in AXES:
        t=mot[f"top7_match_{axis}"].mean()
        s=mot[f"shape_match_{axis}"].mean()
        print(f"{axis}: top7={t:.6f} shape={s:.6f} delta={s-t:+.6f}")
    top_both=(mot["top7_match_center"] & mot["top7_match_spread"]).mean()
    shape_both=(mot["shape_match_center"] & mot["shape_match_spread"]).mean()
    print(f"both center+spread motion match: top7={top_both:.6f} shape={shape_both:.6f} delta={shape_both-top_both:+.6f}")
    print()
    print("DELTA CORRELATION")
    for axis in AXES:
        top_p=mot[f"a_d_{axis}"].corr(mot[f"top7_d_{axis}"])
        shp_p=mot[f"a_d_{axis}"].corr(mot[f"shape_d_{axis}"])
        print(f"{axis}: top7={top_p:.6f} shape={shp_p:.6f} delta={shp_p-top_p:+.6f}")
    print()
    print("RECENT 15")
    cols=["round","top7","shape_box","swapped","out","in","mass_loss_ratio","top7_center_error","shape_center_error","top7_spread_error","shape_spread_error"]
    print(res[cols].tail(15).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__=="__main__":
    main()
