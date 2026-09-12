# Decisions｜lotocore-X

## Championと比較条件

- Loto Coreを固定寄りのChampionとして保持する。
- Dynamic Structure XとRandomを同じウォークフォワード条件で比較する。
- 結果開示後の後付け最適化を避け、介入不能Fieldを保つ。

## 局所Flow

- 観測：初回比較でLoto Coreは平均hit 1.3500、hit3+ 12.0%。
- 観測：Run #21の直近300回比較では局所Flowが5勝0敗。
- 観測：spread拡大とspacing不均一度上昇が同時に現れる候補が抽出された。
- 判断：固定された流儀ではなく、Field依存の一時的流儀として保持する。
- 未確定：5勝0敗が持続効果か一時的偏りか。

## Field間接続

- Loto：介入不能FieldでXの特徴を観測する。
- Kaggriculture：資源境界・経済ループ・相手介入へ接続する。
- 判断：両者を混ぜず、Field依存とX固有を比較する。

## 2026-09-06｜Re-entry方式

- Cut：Flowの継続を担う独立Diff Server。呼び出し動作が増えるため目的と不一致。
- 試行：`AGENTS.md`と`.flow/current.json`を作業量の多いリポジトリへ置く。
- 境界：リポジトリが作業場として選ばれている場合に限り、自動読込と復帰を期待できる。
- 判定保留：新規スレッドでの実地Re-entryまで有効性を確定しない。

## 2026-09-12｜LOTO6 Dynamic X

- 既存LOTO7実験は保持し、LOTO6は別ファイル・別workflowとして追加する。
- LOTO6では細かい配置特徴を主役にせず、`quiet / one_side_stay / turbulent` の粗いFieldからRelationを立てる。
- LOTO6版は旧LOTO7 `x_agent.py` の完全移植とは呼ばない。Dynamic Xの上位構造を保った最小アダプタとして扱う。
- Run #1 (`34687301529`) は成功。第2037回〜第2136回の100回から15口を生成し、artifact `10296052667` を保存した。
- Run #1の15口は次回結果を見る前の固定出力として保持する。
- CORE15口・会話上の仮X15口・GitHub X15口を混同せず、生成経路ごとに分離して観測する。
- 閾値・Weight・analog類似度は暫定。1回の結果から予測能力や優劣を確定しない。
