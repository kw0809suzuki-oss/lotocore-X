# LOTO7｜局所観測ノート v0

第1回〜695回などの短い連続区間を眺め、**何が変わり、何が残り、どこで関係が崩れたか**を元の数字へ戻れる形で残す、小さなStreamlit観測道具です。

これは予測機、自動分類器、類似度ランキングではありません。

## 起動

リポジトリ直下で:

```bash
python fetch_loto7.py --limit 695
pip install -r local_observation_note/requirements.txt
streamlit run local_observation_note/app.py
```

`data/loto7.csv` が無い場合は、画面からCSVをアップロードできます。

## 入力

最低限:

```text
round,n1,n2,n3,n4,n5,n6,n7
```

既存 `fetch_loto7.py` の `date,b1,b2` 列があっても構いません。**ボーナス数字は観測に混ぜません。**

入力時に以下を表示します。

- 欠回
- 回番号重複
- 本数字の重複
- 1〜37範囲外
- 数値として読めない必須値

表示用に回番号順・数字順へ整えても、アップロードした元bytesは `local_observation_note/workspace/source.csv` にそのまま保存します。

## v0でできること

- 横軸1〜37、縦軸抽選回の点配置を見る
- 任意区間を伸縮する
- ObservationとInterpretationを分けて保存する
- 注目数字・注目範囲を任意で残す
- 保存した区間へ戻る
- 二つの区間を左右に並べる
- 観測JSONをダウンロードする
- source SHA-256で観測元を確認する

点同士は自動で結びません。

## 保存

実行時の作業物:

- `local_observation_note/workspace/source.csv`
- `local_observation_note/workspace/observations.json`

各観測JSONには、その区間の回番号と本数字7個もスナップショットとして保持します。

## v0でやらないこと

- 予測
- 自動分類
- 現象名の強制
- 類似度順位
- Null比較
- 数字間の自動対応
- 「近くに出た」ことを「移動した」と解釈すること

まず触って、**記述で取りこぼしたものが出たところだけ直す**ことを優先します。
