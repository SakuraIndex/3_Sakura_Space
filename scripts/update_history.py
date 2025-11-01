#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTRA4などの指数の intraday CSV から、当日分の終値（または最新値）を
docs/outputs/<index>_history.csv（date,value）へ追記／更新するスクリプト。

- 環境変数 INDEX_KEY（例: "astra4"）があればそれを使用。無ければ "astra4"。
- intraday が無ければ何もせず終了（0終了）。
- 既に当日分の行があれば上書き（最新値で更新）。
- 値列の自動判定を強化（外部指数列は無視）。%系が来た場合は前日レベルから復元。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd

JP_TZ = "Asia/Tokyo"
MAX_ABS_DAILY_MOVE = 0.25  # 25%
ROUND = 6

BLOCK_COL_PATTERNS = ("^dji", "^ixic", "^n225", "eurjpy", "usd", "spx", "sp500")

def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def pick_time_column(df: pd.DataFrame) -> str | None:
    for k in ("time", "timestamp", "datetime", "date"):
        if k in df.columns:
            return k
    for c in df.columns:
        if str(c).lower().startswith("unnamed"):
            return c
    return None

def pick_value_series(df: pd.DataFrame, index_key: str) -> pd.Series | None:
    cols = list(df.columns)
    keyn = norm(index_key)
    # 1) index名に近い列を優先
    for c in cols:
        if norm(str(c)) in (keyn, f"{keyn}value", f"{keyn}level"):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    # 2) value/level
    for c in cols:
        cn = str(c).lower()
        if cn in ("value","level","index","close") or "value" in cn:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    # 3) それ以外の数値列。ただし外部指数っぽい列は除外
    candidates = []
    for c in cols:
        cn = str(c).lower()
        if any(pat in cn for pat in BLOCK_COL_PATTERNS):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            candidates.append((c, s))
    if not candidates:
        return None
    # 分散が最大の列を取る（定数列などを避けるため）
    candidates.sort(key=lambda x: float(pd.Series(x[1]).var(skipna=True)), reverse=True)
    return candidates[0][1]

def load_intraday_csv(path: Path, index_key: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[INFO] intraday CSV not found: {path}")
        return pd.DataFrame(columns=["time","value"])

    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    tcol = pick_time_column(raw)
    if tcol is None:
        print("[WARN] no time-like column in intraday CSV.")
        return pd.DataFrame(columns=["time","value"])

    vser = pick_value_series(raw, index_key)
    if vser is None:
        print("[WARN] no value-like column in intraday CSV.")
        return pd.DataFrame(columns=["time","value"])

    t = pd.to_datetime(raw[tcol], errors="coerce", utc=True).dt.tz_convert(JP_TZ)
    v = pd.to_numeric(vser, errors="coerce")

    out = pd.DataFrame({"time": t, "value": v})
    out = out.dropna(subset=["time","value"]).sort_values("time").reset_index(drop=True)
    return out

def read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date","value"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"date","value"}.issubset(df.columns):
        return pd.DataFrame(columns=["date","value"])
    df = df[["date","value"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    return df

def write_history(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out = out.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    out["value"] = out["value"].round(ROUND)
    out.to_csv(path, index=False)

def main() -> int:
    index_key = os.environ.get("INDEX_KEY", "astra4").strip().lower()
    outputs = Path("docs") / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    intraday_path = outputs / f"{index_key}_intraday.csv"
    history_path  = outputs / f"{index_key}_history.csv"

    intraday = load_intraday_csv(intraday_path, index_key)
    if intraday.empty:
        print("[INFO] intraday is empty. Skip updating history.")
        return 0

    hist = read_history(history_path)
    prev_level = float(hist.iloc[-1]["value"]) if not hist.empty else None
    last_date  = hist.iloc[-1]["date"] if not hist.empty else None

    latest_row  = intraday.iloc[-1]
    latest_time = latest_row["time"]
    latest_date = latest_time.tz_convert(JP_TZ).date()
    raw_value   = float(latest_row["value"])

    if not math.isfinite(raw_value):
        print("[WARN] latest value is not finite. skip.")
        return 0

    # 値の解釈：前日レベルがあり、raw が小さすぎる（%っぽい）なら pct とみなして復元
    if prev_level is not None and abs(raw_value) <= 5.0 and prev_level > 10.0:
        pct = raw_value / 100.0
        new_level = prev_level * (1.0 + pct)
        print(f"[INFO] interpret as percent: prev={prev_level:.6f}, pct={raw_value:.4f}% -> {new_level:.6f}")
    else:
        new_level = raw_value

    if prev_level is not None:
        move = (new_level - prev_level) / prev_level
        if abs(move) > MAX_ABS_DAILY_MOVE:
            print(f"[WARN] abnormal daily move {move*100:.2f}% > {MAX_ABS_DAILY_MOVE*100:.0f}%. skip.")
            return 0

    # 未来日防止
    if last_date is not None and latest_date < last_date:
        print("[WARN] latest_date < last_date. skip.")
        return 0

    # 更新 or 追記
    if not hist.empty and (hist["date"] == latest_date).any():
        hist.loc[hist["date"] == latest_date, "value"] = new_level
        print(f"[INFO] updated {latest_date}: {new_level:.6f}")
    else:
        hist = pd.concat([hist, pd.DataFrame([{"date": latest_date, "value": new_level}])],
                         ignore_index=True)
        print(f"[INFO] appended {latest_date}: {new_level:.6f}")

    write_history(hist, history_path)
    (outputs / "_last_run.txt").write_text(pd.Timestamp.now(tz=JP_TZ).isoformat())
    print(f"[OK] wrote {history_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
