# lotocore-X｜作業開始規約

このリポジトリでは、予測や変更より先に現在Stateを再取得する。

## 開始時

1. `.flow/current.json`を読む。
2. `route`に書かれた参照先を必要な順に読む。
3. 保存済み結果・実装・会話由来の観測を混同せず、根拠の所在を示す。
4. 観測事実 → 解釈・推論 → 判断を分けて現在地を出す。

## 作業中

- この実験は予測可能性の観測であり、当選を保証しない。
- 介入不能Fieldであることを保ち、結果を見た後の後付け最適化を避ける。
- Loto Core / X / Randomを同じウォークフォワード条件で比較する。
- 不在を否定として扱わず、未保存の結果を推測で復元しない。
- 新しい作用はChampionとの差分とProbeなしとの差を残す。

## 終了時

- 意味のあるState変化があった場合だけ`.flow/current.json`を更新する。
- 採用・Cut・比較条件の変更は`.flow/decisions.md`へ追記する。
- 一次結果を伴う実験だけ`.flow/runs/`へ記録する。
- 次のスレッドが復帰するための最小情報だけを残す。

## 2026-10-05｜Authority境界
- `archive/` は過去Probe・旧workflowの保存域。現在Stateとして自動昇格させない。
- repository 内の旧 `frozen` 名称を、現在プロジェクトの Frozen v0.1 と同一視しない。
- Branch10 / Frozen v0.1 は、この repository に正本が存在すると確認できるまで外部Authorityとして扱う。
- `.flow/current.json` の Phase10 は repository 内の保存Stateであり、プロジェクト全体の最新研究座標とは限らない。
