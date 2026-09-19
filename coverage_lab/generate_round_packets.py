#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

import lotocore
import x_agent

def actual_numbers(row):
    return [int(row[f"n{i}"]) for i in range(1,8)]

def packet(round_row, history, model_name, snapshot, model_version, relations):
    return {
        "round": int(round_row["round"]),
        "date": str(round_row.get("date","")) or None,
        "history_end_round": int(history.iloc[-1]["round"]),
        "model": model_name,
        "model_version": model_version,
        "pre_draw": True,
        "scores": snapshot["scores"],
        "ranks": snapshot["ranks"],
        "relations": relations,
        "state": snapshot["state"],
        "actual": None,
    }

def run(data_path: Path, out_dir: Path, window: int, last_n: int, version: str):
    df=pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    start=max(window, len(df)-last_n)
    out_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]

    for i in range(start,len(df)):
        history=df.iloc[i-window:i]
        target=df.iloc[i]
        core_pred=set(lotocore.predict(history).numbers)
        x_pred=set(x_agent.predict(history,competition_gate=True).numbers)
        relations={
            "core_x_both":sorted(core_pred & x_pred),
            "core_only":sorted(core_pred - x_pred),
            "x_only":sorted(x_pred - core_pred),
        }
        snaps={
            "lotocore":lotocore.score_snapshot(history),
            "x":x_agent.score_snapshot(history,competition_gate=True),
        }
        for model_name,snap in snaps.items():
            p=packet(target,history,model_name,snap,version,relations)
            path=out_dir/f"round_{int(target['round'])}_{model_name}.json"
            path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            manifest.append({
                "round":int(target["round"]),
                "model":model_name,
                "path":str(path),
                "history_end_round":int(history.iloc[-1]["round"]),
            })

    (out_dir/"manifest.json").write_text(
        json.dumps({"packets":manifest,"actual_attached":False},ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({"packet_count":len(manifest),"round_count":len(manifest)//2,"out_dir":str(out_dir)},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/loto7.csv"))
    p.add_argument("--out",type=Path,default=Path("coverage_lab/packets"))
    p.add_argument("--window",type=int,default=100)
    p.add_argument("--last-n",type=int,default=20)
    p.add_argument("--version",default="main-score-snapshot-v0.1")
    a=p.parse_args()
    run(a.data,a.out,a.window,a.last_n,a.version)

if __name__=="__main__":
    main()
