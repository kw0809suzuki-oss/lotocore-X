from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

URLS = [
    "https://www.luckydayloto.com/loto6/datatable/winningnumber-table.html",
    "https://www.luckydayloto.com/loto6/datatable/winningnumber-table-page2.html",
    "https://www.luckydayloto.com/loto6/datatable/winningnumber-table-page3.html",
    "https://www.luckydayloto.com/loto6/datatable/winningnumber-table-page4.html",
]
OUT = Path("data/loto6.csv")


def _clean_table(df: pd.DataFrame) -> pd.DataFrame | None:
    # The source table is displayed as:
    # round | date | n1..n6 | bonus
    if df.shape[1] < 9:
        return None
    out = df.iloc[:, :9].copy()
    out.columns = ["round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "b1"]
    out["round"] = (
        out["round"]
        .astype(str)
        .str.replace("第", "", regex=False)
        .str.replace("回", "", regex=False)
    )
    numeric = ["round", "n1", "n2", "n3", "n4", "n5", "n6", "b1"]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["round", "n1", "n2", "n3", "n4", "n5", "n6"])
    if out.empty:
        return None
    out["round"] = out["round"].astype(int)
    for c in ["n1", "n2", "n3", "n4", "n5", "n6", "b1"]:
        out[c] = out[c].astype("Int64")
    return out


def _fetch_page(url: str) -> list[pd.DataFrame]:
    headers = {"User-Agent": "lotocore-X research experiment"}
    candidates_url = [url]
    alt = url.replace("www.luckydayloto.com", "luckydayloto.com")
    if alt != url:
        candidates_url.append(alt)

    last_error: Exception | None = None
    for candidate_url in candidates_url:
        for attempt in range(3):
            try:
                r = requests.get(candidate_url, headers=headers, timeout=20)
                r.raise_for_status()
                tables = pd.read_html(StringIO(r.text))
                candidates: list[pd.DataFrame] = []
                for table in tables:
                    cleaned = _clean_table(table)
                    if cleaned is not None and len(cleaned) >= 20:
                        candidates.append(cleaned)
                return candidates
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

    if last_error is not None:
        raise last_error
    return []


def fetch(limit: int = 100) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for url in URLS:
        candidates.extend(_fetch_page(url))
        merged = pd.concat(candidates, ignore_index=True).drop_duplicates("round") if candidates else pd.DataFrame()
        if len(merged) >= limit:
            break

    if not candidates:
        raise RuntimeError("LOTO6 history table not found")

    data = pd.concat(candidates, ignore_index=True).drop_duplicates("round")
    data = data.sort_values("round").tail(limit).reset_index(drop=True)
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()

    df = fetch(args.limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(
        f"saved {len(df)} LOTO6 draws: round {df['round'].min()}..{df['round'].max()} -> {OUT}"
    )


if __name__ == "__main__":
    main()
