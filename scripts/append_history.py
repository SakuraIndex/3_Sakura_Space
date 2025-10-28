#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTRA4: docs/outputs/astra4_history.csv に「その日の終値レベル」を 1 行だけ追加する。
優先度:
  1) docs/outputs/astra4_intraday.csv があれば、その日の「最新時刻の値」
  2) なければ現状維持（何もせず終了）
重複（日付かぶり）は自動スキップ。
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

INDEX_KEY = os.environ.get("INDEX_KEY", "astra4")
OUT_DIR = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_CSV = OUT_DIR / f"{INDEX_KEY}_history.csv"
INTRA_CSV = OUT_DIR / f"{INDEX_KEY}_intraday.csv"

def load_intraday_last_value() -> tuple[str, float] | None:
    """intraday.csv から最新日の最終値（ローカル時刻扱い）を取り出す。"""
    if not INTRA_CSV.exists():
        print(f"[append] not found: {INTRA_CSV}")
        return None
    df = pd.read_csv(INTRA_CSV)
    if df.shape[1] < 2:
        print(f"[append] invalid intraday shape: {INTRA_CSV}")
        return None
    ts_col, val_col = df.columns[:2]
    df = df.rename(columns={ts_col: "ts", val_col: "val"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts","val"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        print("[append] intraday is empty")
        return None
    last_row = df.iloc[-1]
    date_str = last_row["ts"].date().isoformat()
    value = float(last_row["val"])
    return date_str, value

def append_history(date_str: str, value: float) -> bool:
    """history.csv に (date,value) を追記。重複日はスキップ。追記したら True。"""
    if HIST_CSV.exists():
        hist = pd.read_csv(HIST_CSV)
        if hist.shape[1] >= 2:
            dcol, vcol = hist.columns[:2]
            hist = hist.rename(columns={dcol: "date", vcol: "value"})
        else:
            # 壊れていたら作り直し
            hist = pd.DataFrame(columns=["date","value"])
    else:
        hist = pd.DataFrame(columns=["date","value"])

    if (hist["date"] == date_str).any():
        print(f"[append] already exists: {date_str} -> skip")
        return False

    new = pd.DataFrame([{"date": date_str, "value": value}])
    out = pd.concat([hist, new], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(HIST_CSV, index=False)
    print(f"[append] appended: {date_str},{value}")
    return True

def main():
    rv = load_intraday_last_value()
    if rv is None:
        print("[append] nothing to do.")
        return
    date_str, value = rv
    append_history(date_str, value)

if __name__ == "__main__":
    main()
