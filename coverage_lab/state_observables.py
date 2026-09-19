#!/usr/bin/env python3
from __future__ import annotations

import argparse, math, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lotocore
import x_agent

ROLL=100
KS=(10,12,15,18)

def entropy(vals):
    xs=[max(float(v),0.0) for v in vals]
    s=sum(xs)
    if s<=0: return 0.0
    ps=[x/s for x in xs if x>0]
    return -sum(p*math.log(p) for p in ps)

def core_obs(snap):
    scores={int(k):float(v) for k,v in snap["scores"].items()}
    ranked=sorted(scores,key=lambda n:(-scores[n],n))
    vals=[scores[n] for n in ranked]
    total=sum(vals)
    top7=sum(vals[:7])
    top12=sum(vals[:12])
    return {
        "score_top1": vals[0],
        "score_margin_1_7": vals[0]-vals[6],
        "score_margin_7_8": vals[6]-vals[7],
        "top7_mass": top7/total if total else 0.0,
        "top12_mass": top12/total if total else 0.0,
        "score_entropy": entropy(vals),
    }

def x_obs(snap):
    st=snap["state"]
    return {
        "competition_margin": float(st["competition_margin"]),
        "spacing_strength": float(st["spacing_strength"]),
        "action_signal": float(st["action_signal"]),
        "field_center_abs": abs(float(st["field_center_delta"])),
        "field_spread_abs": abs(float(st["field_spread_delta"])),
        "gap_abs": abs(float(st["gap_delta"])),
    }

def make_observables(data_path:Path, window:int=100):
    df=pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    rows=[]
    for i in range(window,len(df)):
        hist=df.iloc[i-window:i]
        rnd=int(df.iloc[i]["round"])
        for model,snap,func in (
            ("lotocore",lotocore.score_snapshot(hist),core_obs),
            ("x",x_agent.score_snapshot(hist,competition_gate=True),x_obs),
        ):
            rec={"round":rnd,"model":model}
            rec.update(func(snap))
            rows.append(rec)
    return pd.DataFrame(rows)

def rolling_bin(series: pd.Series, prior: pd.Series):
    if len(prior)<ROLL: return None
    q1=prior.quantile(1/3)
    q2=prior.quantile(2/3)
    v=series
    if v<=q1: return "low"
    if v>=q2: return "high"
    return "mid"

def attach_bins(obs:pd.DataFrame):
    out=[]
    for model,m in obs.groupby("model"):
        m=m.sort_values("round").reset_index(drop=True)
        metric_cols=[c for c in m.columns if c not in ("round","model")]
        for metric in metric_cols:
            vals=m[metric]
            bins=[]
            for i,v in enumerate(vals):
                prior=vals.iloc[max(0,i-ROLL):i]
                bins.append(rolling_bin(v,prior))
            x=m[["round","model"]].copy()
            x["metric"]=metric
            x["metric_value"]=vals
            x["state_bin"]=bins
            out.append(x)
    return pd.concat(out,ignore_index=True)

def summarize(merged:pd.DataFrame):
    use=merged.dropna(subset=["state_bin"])
    print("=== MODEL-SPECIFIC PRE-TARGET STATE x COMPRESSION ===")
    print(f"rolling_reference={ROLL} prior rounds; bins=prior-tertiles")
    for model in sorted(use.model.unique()):
        print(f"MODEL {model}")
        mm=use[use.model==model]
        for metric in sorted(mm.metric.unique()):
            g=mm[mm.metric==metric]
            print(f" METRIC {metric}")
            for b in ("low","mid","high"):
                z=g[g.state_bin==b]
                if z.empty: continue
                ks=[]
                for k in KS:
                    q=z[z.k==k]
                    ks.append(f"K{k}={q.hit_lift.mean():.3f}")
                print(f"  {b:4s} rounds={z['round'].nunique():3d} " + " ".join(ks))
    print("BOUNDARY: descriptive pre-target state stratification. No metric or threshold is promoted from this run alone.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/loto7.csv"))
    p.add_argument("--curve",type=Path,default=Path("results/loto7_candidate_compression_fixed_k.csv"))
    p.add_argument("--out",type=Path,default=Path("results/loto7_candidate_compression_state_observables.csv"))
    p.add_argument("--window",type=int,default=100)
    a=p.parse_args()
    obs=make_observables(a.data,a.window)
    bins=attach_bins(obs)
    curve=pd.read_csv(a.curve)
    merged=curve.merge(bins,on=["round","model"],how="inner")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    merged.to_csv(a.out,index=False)
    summarize(merged)
    print(f"saved -> {a.out}")

if __name__=="__main__":
    main()
