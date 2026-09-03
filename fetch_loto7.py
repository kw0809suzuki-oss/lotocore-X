from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

START_URL = "https://www.takarakuji-loto7.jp/LotoNumber/HistoryPartOutPut"
OUT = Path("data/loto7.csv")


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:, :11].copy()
    df.columns = [
        "round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "b1", "b2"
    ]
    df["round"] = (
        df["round"].astype(str).str.replace("第", "", regex=False).str.replace("回", "", regex=False)
    )
    for c in ["round", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "b1", "b2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["round", "n1", "n2", "n3", "n4", "n5", "n6", "n7"])
    df["round"] = df["round"].astype(int)
    for c in ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "b1", "b2"]:
        df[c] = df[c].astype("Int64")
    return df


def fetch(limit: int = 180) -> pd.DataFrame:
    url = START_URL
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    headers = {"User-Agent": "lotocore-X research experiment"}

    while url and url not in seen:
        seen.add(url)
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        if not tables:
            raise RuntimeError(f"history table not found: {url}")
        frames.append(_clean_table(tables[0]))

        combined = pd.concat(frames, ignore_index=True).drop_duplicates("round")
        if len(combined) >= limit:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        nxt = None
        for a in soup.find_all("a", href=True):
            text = "".join(a.stripped_strings)
            if "次のページ" in text:
                nxt = urljoin(url, a["href"])
                break
        url = nxt

    data = pd.concat(frames, ignore_index=True).drop_duplicates("round")
    data = data.sort_values("round").tail(limit).reset_index(drop=True)
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=180)
    args = p.parse_args()

    df = fetch(args.limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} draws: round {df['round'].min()}..{df['round'].max()} -> {OUT}")


if __name__ == "__main__":
    main()
