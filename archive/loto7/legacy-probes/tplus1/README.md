# LOTO7 T+1 時間想像モデル｜Playground

このフォルダは **独立した遊び枝**。

既存の LOTO CORE / X / Random、本線の `.flow/current.json`、および他の遊び（Master vs Flow 等）には触れない。

## 発想

次の数字を直接予測しない。

1回の抽選を `State_T` として見る。

- 重心 `mu = sum(numbers) / 7`
- 広がり `width = max(numbers) - min(numbers)`
- 密集 `cluster` は最初は粗く扱う

そのうえで、

`State_T -> State_(T+1)*`

という **次の状態の想像** を先に置き、その状態から数字へ Re-entry する。

## 最初の検証

当選数ではなく、まず State だけを見る。

- 実測 `State_(T+1)`
- 想像 `State_(T+1)*`
- Random な次State

を比較し、T+1想像が Random より近いかだけを見る。

### 距離の最小形

```text
distance =
  |mu_pred - mu_actual| / mu_scale
+ |width_pred - width_actual| / width_scale
```

cluster は後付けせず、必要なら第2段階で追加する。

## Walk-forward

各時点 T で使えるのは T までの情報だけ。

```text
history <= T
   ↓
State_T
   ↓
T+1 State を想像
   ↓
実際の T+1 を開封
   ↓
距離比較
```

未来を見てから規則を変えない。

## G1

まずは2次元だけ。

`State = (重心, 広がり)`

T+1の方向を次の4象限で表す。

- 高側へ + 開く
- 高側へ + 閉じる
- 低側へ + 開く
- 低側へ + 閉じる

現在の変化から次方向を1つ仮置きし、その象限内で粗い次Stateを作る。

## 評価

最初に見るのはこれだけ。

1. T+1想像の平均State距離
2. Random State の平均距離
3. T+1想像が Random より近かった回数
4. 方向一致率（高/低 × 開/閉）

ここで差がなければ Closure。

差があった場合だけ、

`State -> 7数字生成 -> 一致数`

へ Re-entry する。

## 禁止

- 既存 CORE / X の閾値やコードを変更しない
- `.flow/current.json` をこの遊びのために更新しない
- Master vs Flow 側のファイルを変更しない
- 実結果を見たあとでルールを最適化しない
- 単発の良結果を予測能力と呼ばない

## 一行

> 数字を予測するのではなく、数字空間が時間を1つ進めたときの「次の状態」を想像し、その想像に情報があるかを見る。
