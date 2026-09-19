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

## 2026-09-19｜LOTO7 Phase 10 実購入シミュレーション

- Phase 9 strict-forwardの次Probeとして、候補層liftではなく10口の券面結果へ接続する。
- Phase 10の主候補は、Phase 9でsame-K random差が最も大きかった X / last1 / K10 に固定する。これはPhase 10用の比較候補であり、Champion昇格ではない。
- 各回10口、1口300円（1回3,000円）を実購入単位として扱う。
- 当せん判定は本数字7個・ボーナス数字2個を使い、ロト7の1〜6等条件にそのまま通す。
- 比較枝は adaptive X last1 K10 / fixed X K10 / fixed X K37 / same-K random / uniform random tickets。
- 券面allocatorは既存 compare_loto7_mesh.py のdegree-preserving / pair-dispersion思想を再利用し、結果開示後に券を組み替えない。
- 賞金額は回ごとに変動するため、一次判定は等級発生数・本数字最大一致・無当せん率を主指標とする。金額換算を出す場合は公式理論値として別扱いする。
