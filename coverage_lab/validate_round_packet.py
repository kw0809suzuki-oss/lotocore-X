#!/usr/bin/env python3
import json, sys
from pathlib import Path

def main(path):
    d=json.loads(Path(path).read_text(encoding="utf-8"))
    errs=[]
    if d.get("pre_draw") is not True: errs.append("pre_draw must be true")
    scores=d.get("scores",{})
    ranks=d.get("ranks",{})
    expected={str(i) for i in range(1,38)}
    if set(scores)!=expected: errs.append("scores must contain keys 1..37 exactly")
    if set(ranks)!=expected: errs.append("ranks must contain keys 1..37 exactly")
    rv=list(ranks.values())
    if sorted(rv)!=list(range(1,38)): errs.append("ranks must be a permutation of 1..37")
    if d.get("history_end_round",10**9) >= d.get("round",-1):
        errs.append("history_end_round must be < round")
    print(json.dumps({"valid":not errs,"errors":errs},ensure_ascii=False,indent=2))
    return 0 if not errs else 1

if __name__=="__main__":
    raise SystemExit(main(sys.argv[1]))
