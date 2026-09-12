# LOTO6 CORE / X / Random 比較｜2026-09-12

## 条件
- 取得履歴：第2037回〜第2136回、100回
- 入力窓：直前60回
- 評価：残り40回を順番に未読ウォークフォワード
- 同一条件：CORE / Dynamic X / Random
- Random：抽選回番号をseedにして再現可能
- Run：GitHub Actions `LOTO6 Core X Random Compare` #2
- Artifact：`loto6-core-x-random-results` / 10296252451

## 一次結果
- CORE：n=40 / mean_hits=0.7750 / hit3+=0.0000 / best=2 / dist={0:14,1:21,2:5}
- X：n=40 / mean_hits=1.0500 / hit3+=0.0750 / best=4 / dist={0:13,1:16,2:8,3:2,4:1}
- Random：n=40 / mean_hits=0.7750 / hit3+=0.0000 / best=2 / dist={0:17,1:15,2:8}

## ペア比較
- X > CORE：11
- X = CORE：25
- X < CORE：4
- X > Random：18
- X = Random：13
- X < Random：9
- CORE > Random：13
- CORE = Random：15
- CORE < Random：12

## 観測
この40回ではXの平均一致数と最大一致数がCORE / Randomを上回った。COREとRandomは平均一致数が同じだった。

## 境界
- 40回だけでXの一般的優位や予測能力を確定しない。
- LOTO6 Coreは今回追加した保守的baselineであり、会話で作った15口COREと同一物ではない。
- Xの閾値・Weight・60回窓は暫定。
- 次は結果を細分化しすぎず、Xが勝った11回 / 負けた4回でField regimeとの関係があるかを観測候補とする。
