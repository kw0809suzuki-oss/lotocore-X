# lotocore-X

LOTO6 / LOTO7 の実験リポジトリ。

## Authority boundary / 正本境界

この repository の現行コード系は主に Loto Core / Dynamic Structure X / Random / Candidate Compression Phase 10。
現在プロジェクトで扱っている Branch10 と Frozen v0.1（4回窓・30特徴・K30・9 states・Branch3）の正本は、この repository には存在しない。

特に注意:
- archive 内の旧 slow/fast "frozen" は固定済み alpha の意味で、Frozen v0.1 ではない。
- coverage_lab 内の "frozen Round Packet" は結果開示前に固定した packet の意味で、Frozen v0.1 ではない。

## Active surface / 現役面
- lotocore.py
- x_agent.py
- fetch_loto7.py
- walk_forward.py
- coverage_lab/
- .github/workflows/candidate-compression-phase10.yml
- .github/workflows/loto7-phase10-next-snapshot.yml
- .github/workflows/loto7-core-vs-random.yml

LOTO6 は並行保存し、LOTO7 の現在研究と混同しない。

詳しい仕分けは docs/REPOSITORY_MAP.md を参照。
過去 Probe / 旧 workflow は archive/ に保存し、現在Authorityとして読まない。
