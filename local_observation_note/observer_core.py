from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

NUMBER_COLUMNS = [f"n{i}" for i in range(1, 8)]
REQUIRED_COLUMNS = ["round", *NUMBER_COLUMNS]


def source_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_csv(raw: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(raw))


def validate_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), [
            "必須列がありません: " + ", ".join(missing)
        ]

    work = df.copy()
    for col in REQUIRED_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    invalid_numeric = work[REQUIRED_COLUMNS].isna().any(axis=1)
    if invalid_numeric.any():
        rows = (work.index[invalid_numeric] + 2).tolist()
        errors.append(f"数値として読めない必須値があります: CSV行 {rows}")

    usable = work.loc[~invalid_numeric, REQUIRED_COLUMNS].copy()
    if usable.empty:
        return usable, errors

    usable["round"] = usable["round"].astype(int)
    for col in NUMBER_COLUMNS:
        usable[col] = usable[col].astype(int)

    duplicate_rounds = sorted(
        usable.loc[usable["round"].duplicated(keep=False), "round"].unique().tolist()
    )
    if duplicate_rounds:
        errors.append(f"回番号が重複しています: {duplicate_rounds}")

    for _, row in usable.iterrows():
        r = int(row["round"])
        nums = [int(row[c]) for c in NUMBER_COLUMNS]
        if any(n < 1 or n > 37 for n in nums):
            errors.append(f"第{r}回: 1〜37の範囲外の本数字があります: {nums}")
        if len(set(nums)) != 7:
            errors.append(f"第{r}回: 本数字7個の中に重複があります: {nums}")

    rounds = sorted(usable["round"].unique().tolist())
    if len(rounds) >= 2:
        missing_rounds: list[int] = []
        present = set(rounds)
        for r in range(rounds[0], rounds[-1] + 1):
            if r not in present:
                missing_rounds.append(r)
        if missing_rounds:
            errors.append(f"欠回があります: {missing_rounds}")

    # 表示用だけ回番号順・本数字昇順にする。元CSV bytesは別に保持する。
    rows: list[dict[str, int]] = []
    for _, row in usable.sort_values("round").iterrows():
        nums = sorted(int(row[c]) for c in NUMBER_COLUMNS)
        item: dict[str, int] = {"round": int(row["round"])}
        item.update({f"n{i+1}": n for i, n in enumerate(nums)})
        rows.append(item)
    normalized = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return normalized, errors


def interval_frame(df: pd.DataFrame, start_round: int, end_round: int) -> pd.DataFrame:
    lo, hi = sorted((int(start_round), int(end_round)))
    return df[(df["round"] >= lo) & (df["round"] <= hi)].copy()


def point_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for _, row in df.iterrows():
        r = int(row["round"])
        for col in NUMBER_COLUMNS:
            rows.append({"round": r, "number": int(row[col])})
    return pd.DataFrame(rows, columns=["round", "number"])


def make_observation(
    *,
    source_name: str,
    source_hash: str,
    interval: pd.DataFrame,
    focus_numbers: list[int],
    focus_range: tuple[int, int] | None,
    observation: str,
    interpretation: str,
) -> dict[str, Any]:
    draws = []
    for _, row in interval.sort_values("round").iterrows():
        draws.append(
            {
                "round": int(row["round"]),
                "numbers": [int(row[c]) for c in NUMBER_COLUMNS],
            }
        )
    start_round = draws[0]["round"] if draws else None
    end_round = draws[-1]["round"] if draws else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "source_sha256": source_hash,
        "start_round": start_round,
        "end_round": end_round,
        "focus_numbers": sorted(set(int(n) for n in focus_numbers)),
        "focus_range": list(focus_range) if focus_range is not None else None,
        "observation": observation.strip(),
        "interpretation": interpretation.strip(),
        "draws": draws,
    }


def load_observations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_observations(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
