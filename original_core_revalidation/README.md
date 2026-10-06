# LOTO7 Original CORE Revalidation v0

親問い:

> center × variance で現在地を置き、過去近傍20地点の次状態から作った固定IQR領域は、同幅Random領域より直近100回の実状態を多く含むか。

## 固定仕様

- 評価: 最新100回
- state: center, variance
- K=20
- 距離: past-only z-score Euclidean
- 次領域: neighbor successors の 25–75 percentile rectangle
- Random: 同じwidth、centerは過去stateから無作為抽出
- 1000 repeats
- ticket生成なし
- 他モデル比較なし
- feature追加なし
- parameter探索なし

実行:

```bash
python original_core_revalidation/run.py
```

出力:

- `results/original_core_revalidation/summary.json`
- `results/original_core_revalidation/targets.csv`
- `results/original_core_revalidation/random_hit_counts.json`

結果が保存されるまでは解釈を増やさない。
