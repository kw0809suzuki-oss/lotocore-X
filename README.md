# lotocore-X

ロト7を使って、固定寄りの観測モデル **Loto Core** と、現在Fieldから構造を組み直す **X** を同じ条件で比較する実験場。

## 目的

- 「構造が薄く、こちらからFieldへ介入できない環境」でXが何をするかを見る。
- Xがランダムな揺れを構造として過学習していないか確認する。
- Loto Core / X / Random baseline を同じウォークフォワード条件で比較する。

## 基本ループ

過去履歴 → 予測 → 次回結果を開示 → 評価 → State更新 → 次回予測

## 評価

- 本数字一致数
- 平均一致数
- 3個以上一致率
- 予測集合の重心差
- 予測集合の分散差
- Xの構造更新回数

> この実験は予測可能性の検証であり、当選を保証するものではありません。

## 実行

```bash
pip install -r requirements.txt
python fetch_loto7.py --limit 180
python walk_forward.py --window 100
```

結果は `results/latest.csv` とコンソール集計に出ます。
