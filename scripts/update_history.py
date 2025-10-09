#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTRA4などの指数の intraday CSV から、当日分の終値（または最新値）を
docs/outputs/<index>_history.csv（date,value）へ追記／更新するスクリプト。

- 環境変数 INDEX_KEY（例: "astra4"）があればそれを使用。無ければ "astra4"。
- intraday が無ければ何もせず終了（0終了）。
- 既に当日分の行があれば上書き（最新値で更新）。
"""

from __future__ import annotations

import os
import sys
import pandas as pd
from pathlib import Path

JP_TZ = "Asia/Tokyo"

def load_intraday_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[INFO] intraday CSV not found: {path}")
        return pd.DataFrame(columns=["time", "value"])
    df = pd.read_csv(path)
    # カラム名を小文字正規化
    df.columns = [str(c).strip().lower() for c in df.columns]
    # time の推定
    tcol = None
    for k in ("time", "timestamp", "date", "datetime"):
        if k in df.columns:
            tcol = k
            break
    if tcol is None:
        # Unnamed: 0 などを rescue
        for c in df.columns:
            if c.startswith("unnamed"):
                tcol = c
                break
    if tcol is None:
        print("[WARN] no time-like column in intraday CSV.")
        return pd.DataFrame(columns=["time","value"])

    # value の推定
    vcol = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("value", "index", "score") or ("value" in lc):
            vcol = c
            break
    if vcol is None:
        # time以外の数値列の平均をとる
        numeric_cols = []
        for c in df.columns:
            if c == tcol: 
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                numeric_cols.append(c)
        if not numeric_cols:
            print("[WARN] no numeric columns in intraday CSV.")
            return pd.DataFrame(columns=["time","value"])
        vals = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        v = vals.mean(axis=1)
    else:
        v = pd.to_numeric(df[vcol], errors="coerce")

    t = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(JP_TZ)
    out = pd.DataFrame({"time": t, "value": v})
    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def main():
    index_key = os.environ.get("INDEX_KEY", "astra4").strip().lower()
    outputs = Path("docs") / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    intraday_path = outputs / f"{index_key}_intraday.csv"
    history_path  = outputs / f"{index_key}_history.csv"

    intraday = load_intraday_csv(intraday_path)
    if intraday.empty:
        print("[INFO] intraday is empty. Skip updating history.")
        return 0

    # 当日(JST)の最新値を採用
    latest_row = intraday.iloc[-1]
    latest_time = latest_row["time"].tz_convert(JP_TZ)
    latest_date = latest_time.date()  # 当日の date
    latest_value = float(latest_row["value"])
    print(f"[INFO] latest {index_key}: {latest_date} -> {latest_value}")

    # 既存historyを読み込み
    if history_path.exists():
        hist = pd.read_csv(history_path)
        hist.columns = [str(c).strip().lower() for c in hist.columns]
        if not {"date","value"}.issubset(set(hist.columns)):
            hist = pd.DataFrame(columns=["date","value"])
    else:
        hist = pd.DataFrame(columns=["date","value"])

    # dateをdate型に寄せる
    if "date" in hist.columns:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date

    # 当日行があれば更新、無ければ追記
    if (hist["date"] == latest_date).any():
        hist.loc[hist["date"] == latest_date, "value"] = latest_value
        print("[INFO] updated today row in history.")
    else:
        hist = pd.concat(
            [hist, pd.DataFrame([{"date": latest_date, "value": latest_value}])],
            ignore_index=True
        )
        print("[INFO] appended today row to history.")

    # 並べ替えて保存
    hist = hist.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    hist.to_csv(history_path, index=False)
    print(f"[OK] wrote {history_path}")

    # 実行時刻メモ
    (outputs / "_last_run.txt").write_text(pd.Timestamp.now(tz=JP_TZ).isoformat())

    return 0

if __name__ == "__main__":
    sys.exit(main())
