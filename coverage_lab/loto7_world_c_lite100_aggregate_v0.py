#!/usr/bin/env python3
import json
from pathlib import Path

files=sorted(Path("artifacts").glob("**/world_c_lite_shard_*.json"))
worlds=[]
for f in files:
    worlds.extend(json.loads(f.read_text(encoding="utf-8"))["worlds"])
worlds.sort(key=lambda x:x["world_id"])
if len(worlds)!=100 or [w["world_id"] for w in worlds] != list(range(1,101)):
    raise RuntimeError(f"expected exact worlds 1..100, got {len(worlds)}")
hits=[w["world_id"] for w in worlds if w["fixed_condition"]]
out={
    "experiment":"loto7-world-c-lite100-v0",
    "pre_fixed_condition":"N_plus >= 4 AND delta_min >= 0",
    "world_count":100,
    "condition_hits":len(hits),
    "condition_rate":len(hits)/100,
    "hit_world_ids":hits
}
Path("results/world_c_lite100").mkdir(parents=True,exist_ok=True)
Path("results/world_c_lite100/summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2))
