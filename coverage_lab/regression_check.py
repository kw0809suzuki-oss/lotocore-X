#!/usr/bin/env python3
"""Regression guard for observation-only score snapshots.

Requires the same historical CSV used by existing walk_forward.py.
The guard verifies that calling score_snapshot() does not alter predict()
and that CORE's score ranking reproduces its existing selected 7 numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import lotocore
import x_agent

def check(data_path: Path, window: int = 100, last_n: int = 20) -> None:
    df = pd.read_csv(data_path).sort_values("round").reset_index(drop=True)
    start = max(window, len(df) - last_n)
    checked = 0
    for i in range(start, len(df)):
        history = df.iloc[i-window:i]

        core_before = lotocore.predict(history)
        _ = lotocore.score_snapshot(history)
        core_after = lotocore.predict(history)
        assert core_before.numbers == core_after.numbers
        core_snap = lotocore.score_snapshot(history)
        core_ranked = sorted(range(1,38), key=lambda n: core_snap["ranks"][str(n)])
        assert tuple(sorted(core_ranked[:7])) == core_before.numbers

        x_before = x_agent.predict(history, competition_gate=True)
        _ = x_agent.score_snapshot(history, competition_gate=True)
        x_after = x_agent.predict(history, competition_gate=True)
        assert x_before.numbers == x_after.numbers

        checked += 1

    print({"checked_rounds": checked, "predict_unchanged": True})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/loto7.csv"))
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--last-n", type=int, default=20)
    a=p.parse_args()
    check(a.data,a.window,a.last_n)

if __name__=="__main__":
    main()
