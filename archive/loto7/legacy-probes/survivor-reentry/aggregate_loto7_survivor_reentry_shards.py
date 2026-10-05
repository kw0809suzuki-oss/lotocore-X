from pathlib import Path
import pandas as pd
from probe_loto7_survivor_reentry_shard import load_real, search

files=sorted(Path("results").glob("loto7_survivor_reentry_shard_*.csv"))
if len(files)!=2:
    raise RuntimeError(f"expected 2 shard files, got {len(files)}")
n=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
if len(n)!=1000:
    raise RuntimeError(f"expected 1000 null worlds, got {len(n)}")

real=load_real()
_,_,_,ra,rm=search(real)
print("REAL best alpha",ra,"mean",rm["mean"],"p3",rm["p3"],"range",rm["range"])
print("REAL quarters",rm["quarters"].tolist())
print("REAL survivor-hit",rm["survivor_hit"],"reentry-hit",rm["reentry_hit"])
print("NULL worlds",len(n),"from 2 x 500 shards")
print("P(null best mean >= real)",float((n["mean"]>=rm["mean"]).mean()))
print("P(null best mean>=real AND range<=real)",float(((n["mean"]>=rm["mean"])&(n["range"]<=rm["range"])).mean()))
print("P(null survivor-hit>=real)",float((n["survivor_hit"]>=rm["survivor_hit"]).mean()))
print("P(null reentry-hit>=real)",float((n["reentry_hit"]>=rm["reentry_hit"]).mean()))
n.to_csv("results/loto7_survivor_reentry_2x500.csv",index=False)
