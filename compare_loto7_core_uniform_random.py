"""Walk-forward comparison: existing LOTO7 model family vs ordinary uniform random.
Same target rounds, same 100-draw history window, no target leakage.
"""
from __future__ import annotations
import argparse, random
from collections import Counter
from pathlib import Path
import pandas as pd
import lotocore, x_agent

DATA=Path("data/loto7.csv")
OUT=Path("results/loto7_core_x_vs_uniform_random.csv")
MODELS=("core","x","x_ungated","x_spacing2","x_actionpoint")
RANDOM_REPS=1000

def actual_numbers(row): return set(int(row[f"n{i}"]) for i in range(1,8))
def hits(ticket,actual): return len(set(ticket)&actual)
def predictions(history):
    return {
        "core": lotocore.predict(history).numbers,
        "x": x_agent.predict(history,competition_gate=True).numbers,
        "x_ungated": x_agent.predict(history,competition_gate=False).numbers,
        "x_spacing2": x_agent.predict(history,competition_gate=True,spacing_v2=True).numbers,
        "x_actionpoint": x_agent.predict(history,competition_gate=True,spacing_v2=True,action_point=True).numbers,
    }
def rate(hist,k):
    n=sum(hist.values()); return sum(v for h,v in hist.items() if h>=k)/n if n else 0
def mean(hist):
    n=sum(hist.values()); return sum(h*v for h,v in hist.items())/n if n else 0

def run(window=100,reps=RANDOM_REPS):
    df=pd.read_csv(DATA).sort_values("round").reset_index(drop=True)
    hist={m:Counter() for m in MODELS}; rh=Counter(); rows=[]
    for i in range(window,len(df)):
        history=df.iloc[i-window:i]; row=df.iloc[i]; rnd=int(row["round"]); actual=actual_numbers(row)
        preds=predictions(history); rec={"round":rnd,"date":row.get("date","")}
        for m,ticket in preds.items():
            h=hits(ticket,actual); hist[m][h]+=1; rec[f"{m}_hits"]=h
        rng=random.Random(rnd*1000003+20260917); rs=[]
        for _ in range(reps):
            h=hits(rng.sample(range(1,38),7),actual); rh[h]+=1; rs.append(h)
        rec["random_mean_hits"]=sum(rs)/len(rs); rows.append(rec)
    return pd.DataFrame(rows),hist,rh

def show(name,h):
    print(f"{name:13s} mean={mean(h):.4f} 3+={rate(h,3):.4f} 4+={rate(h,4):.4f} 5+={rate(h,5):.4f} 6+={rate(h,6):.4f} 7={rate(h,7):.4f} dist={dict(sorted(h.items()))}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--window",type=int,default=100); p.add_argument("--random-reps",type=int,default=RANDOM_REPS); a=p.parse_args()
    res,hists,rh=run(a.window,a.random_reps); OUT.parent.mkdir(parents=True,exist_ok=True); res.to_csv(OUT,index=False)
    print("LOTO7 CORE / X FAMILY vs UNIFORM RANDOM — WALK-FORWARD")
    print(f"target_rounds={len(res)} history_window={a.window} random_reps_per_round={a.random_reps}")
    for m in MODELS: show(m.upper(),hists[m])
    show("RANDOM",rh)
    print("NOTE: historical walk-forward observation; not evidence of future lottery advantage by itself.")
    print(f"saved -> {OUT}")
if __name__=="__main__": main()
