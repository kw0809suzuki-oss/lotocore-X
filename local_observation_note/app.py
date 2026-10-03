from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from observer_core import (
    NUMBER_COLUMNS,
    interval_frame,
    load_observations,
    make_observation,
    parse_csv,
    point_frame,
    save_observations,
    source_sha256,
    validate_and_normalize,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE = HERE / "workspace"
SOURCE_COPY = WORKSPACE / "source.csv"
OBSERVATIONS = WORKSPACE / "observations.json"
DEFAULT_SOURCE = REPO_ROOT / "data" / "loto7.csv"

st.set_page_config(page_title="LOTO7 局所観測ノート v0", layout="wide")
st.title("LOTO7｜局所観測ノート v0")
st.caption("法則を当てる機械ではなく、見えた関係を潰さず持ち帰るための観測面。")


def load_source() -> tuple[bytes | None, str | None]:
    source_bytes: bytes | None = None
    source_name: str | None = None

    if DEFAULT_SOURCE.exists():
        if st.checkbox("既存 data/loto7.csv を使う", value=True):
            source_bytes = DEFAULT_SOURCE.read_bytes()
            source_name = str(DEFAULT_SOURCE.relative_to(REPO_ROOT))

    uploaded = st.file_uploader("別のCSVを使う場合はここへ", type=["csv"])
    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        source_name = uploaded.name

    return source_bytes, source_name


raw, source_name = load_source()
if raw is None or source_name is None:
    st.info("CSVを用意してください。既存の fetch_loto7.py なら data/loto7.csv が作られます。")
    st.stop()

WORKSPACE.mkdir(parents=True, exist_ok=True)
SOURCE_COPY.write_bytes(raw)
source_hash = source_sha256(raw)

try:
    raw_df = parse_csv(raw)
except Exception as exc:
    st.error(f"CSVを読めませんでした: {exc}")
    st.stop()

df, errors = validate_and_normalize(raw_df)

st.subheader("入力確認")
c1, c2, c3 = st.columns(3)
c1.metric("表示可能な回数", len(df))
c2.metric("最初の回", int(df["round"].min()) if not df.empty else "-")
c3.metric("最後の回", int(df["round"].max()) if not df.empty else "-")
st.caption(f"source: {source_name} / sha256: {source_hash[:16]}…")

if errors:
    st.warning("入力に確認事項があります。元CSVは変更せず workspace/source.csv に保持しています。")
    for msg in errors:
        st.write(f"- {msg}")
else:
    st.success("本数字7個・回番号について、欠回・重複・範囲外の問題は見つかりませんでした。")

if df.empty:
    st.stop()

min_round = int(df["round"].min())
max_round = int(df["round"].max())

st.divider()
st.subheader("観測する区間")

if "window_start" not in st.session_state:
    st.session_state.window_start = max(min_round, max_round - 8)
if "window_end" not in st.session_state:
    st.session_state.window_end = max_round

window = st.slider(
    "表示区間",
    min_value=min_round,
    max_value=max_round,
    value=(int(st.session_state.window_start), int(st.session_state.window_end)),
)
st.session_state.window_start, st.session_state.window_end = window
view = interval_frame(df, *window)


def draw_chart(frame: pd.DataFrame, key: str) -> None:
    points = point_frame(frame)
    if points.empty:
        st.info("この区間に表示できるデータがありません。")
        return
    spec = {
        "mark": {"type": "point", "filled": True, "size": 80},
        "encoding": {
            "x": {
                "field": "number",
                "type": "quantitative",
                "scale": {"domain": [1, 37]},
                "axis": {"title": "数字", "tickMinStep": 1, "values": list(range(1, 38))},
            },
            "y": {
                "field": "round",
                "type": "ordinal",
                "sort": "ascending",
                "axis": {"title": "抽選回"},
            },
            "tooltip": [
                {"field": "round", "type": "ordinal", "title": "回"},
                {"field": "number", "type": "quantitative", "title": "数字"},
            ],
        },
    }
    st.vega_lite_chart(points, spec, use_container_width=True, key=key)


draw_chart(view, "main_chart")
with st.expander("数字列をそのまま見る"):
    st.dataframe(view, hide_index=True, use_container_width=True)

st.divider()
st.subheader("この区間を記録")

with st.form("observation_form"):
    focus_numbers = st.multiselect("注目した数字（任意）", options=list(range(1, 38)))
    use_focus_range = st.checkbox("注目範囲を残す")
    focus_range = None
    if use_focus_range:
        focus_range = st.slider("注目範囲", 1, 37, (1, 37))
    observation = st.text_area(
        "Observation｜何が残ったか・何が変わったか・どこで崩れたか",
        height=150,
    )
    interpretation = st.text_area(
        "Interpretation｜必要な場合だけ。Observationとは分ける",
        height=100,
    )
    submitted = st.form_submit_button("観測を保存")

if submitted:
    if not observation.strip():
        st.error("Observationだけは空にせず残してください。")
    else:
        records = load_observations(OBSERVATIONS)
        record = make_observation(
            source_name=source_name,
            source_hash=source_hash,
            interval=view,
            focus_numbers=focus_numbers,
            focus_range=focus_range,
            observation=observation,
            interpretation=interpretation,
        )
        records.append(record)
        save_observations(OBSERVATIONS, records)
        st.success(
            f"第{record['start_round']}回〜第{record['end_round']}回の観測を保存しました。"
        )

records = load_observations(OBSERVATIONS)

if records:
    st.divider()
    st.subheader("保存した観測へ戻る")
    labels = [
        f"{i+1}. 第{r.get('start_round')}〜{r.get('end_round')}回｜{r.get('observation','')[:36]}"
        for i, r in enumerate(records)
    ]
    selected_index = st.selectbox(
        "観測記録",
        options=list(range(len(records))),
        format_func=lambda i: labels[i],
    )
    selected = records[selected_index]
    if selected.get("source_sha256") != source_hash:
        st.warning("この記録は現在開いているCSVとは異なるsource hashです。")
    st.write("**Observation**")
    st.write(selected.get("observation", ""))
    if selected.get("interpretation"):
        st.write("**Interpretation**")
        st.write(selected.get("interpretation", ""))
    st.caption(
        f"第{selected.get('start_round')}〜第{selected.get('end_round')}回 / "
        f"source {str(selected.get('source_sha256',''))[:16]}…"
    )
    if st.button("この区間をメイン表示へ戻す"):
        st.session_state.window_start = int(selected["start_round"])
        st.session_state.window_end = int(selected["end_round"])
        st.rerun()

    st.download_button(
        "観測JSONをダウンロード",
        data=json.dumps(records, ensure_ascii=False, indent=2),
        file_name="loto7_local_observations.json",
        mime="application/json",
    )

st.divider()
st.subheader("二つの区間を並べる")
left, right = st.columns(2)

with left:
    st.markdown("**区間A**")
    a = st.slider(
        "A",
        min_value=min_round,
        max_value=max_round,
        value=(max(min_round, max_round - 8), max_round),
        key="compare_a",
    )
    frame_a = interval_frame(df, *a)
    draw_chart(frame_a, "compare_a_chart")
    st.dataframe(frame_a, hide_index=True, use_container_width=True)

with right:
    st.markdown("**区間B**")
    default_b_end = max(min_round, max_round - 9)
    default_b_start = max(min_round, default_b_end - 8)
    b = st.slider(
        "B",
        min_value=min_round,
        max_value=max_round,
        value=(default_b_start, default_b_end),
        key="compare_b",
    )
    frame_b = interval_frame(df, *b)
    draw_chart(frame_b, "compare_b_chart")
    st.dataframe(frame_b, hide_index=True, use_container_width=True)

st.caption(
    "点同士は自動で結びません。近い位置に現れたことを、同じ対象の移動とは扱いません。"
)
