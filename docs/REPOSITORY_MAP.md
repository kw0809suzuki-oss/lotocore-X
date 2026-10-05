# REPOSITORY MAP｜2026-10-05

## Active / 現役
Model / runtime:
- lotocore.py
- x_agent.py
- fetch_loto7.py
- walk_forward.py
- random_kun.py

LOTO7 evaluation / Phase10:
- coverage_lab/
- compare_loto7_core_uniform_random.py
- compare_loto7_mesh.py
- observe_loto7_core_x_relation.py
- .github/workflows/candidate-compression-phase10.yml
- .github/workflows/loto7-phase10-next-snapshot.yml
- .github/workflows/loto7-core-vs-random.yml

LOTO6 parallel line:
- loto6_*.py / check_loto6_*.py / run_loto6_x.py
- corresponding .github/workflows/loto6-*.yml

## Archive / 過去探索
- archive/loto7/legacy-probes/72-28/
- archive/loto7/legacy-probes/slow-fast/
- archive/loto7/legacy-probes/survivor-reentry/
- archive/loto7/legacy-probes/residual-play/
- archive/workflows/

Archive は削除ではない。再現・履歴確認用。現在Authorityではない。

## Frozen naming warning
1. 旧 slow/fast probe: trainで選んだ alpha を固定した holdout。
2. coverage_lab frozen Round Packet: 結果開示前に packet を固定。
3. 現在プロジェクトの Frozen v0.1: 4回窓 / 30特徴 / past-only Z / K30 / 9 states / Branch3。

1 と 2 は 3 ではない。
