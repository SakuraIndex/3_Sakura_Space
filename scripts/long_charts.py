#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long-term charts (1d / 7d / 1m / 1y) for an index (e.g., astra4)

- 1d:
  * 当日(JST) 09:00–15:30 のみを抽出（タイムゾーンは環境変数に依存、デフォルトJST）
  * 単一列(value) 形式 / 銘柄横並び(板) 形式 / index列(Unnamed: 0) など何でも受ける
  * 等加重平均で value 算出にも対応
- 7d/1m/1y:
  * docs/outputs/<index>_history.csv（date,value）から描画
  * データ不足時は注記

出力:
  docs/outputs/<index>_1d.png
  docs/outputs/<index>_7d.png
  docs/outputs/<index>_1m.png
  docs/outputs/<index>_1y.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ================== 設定 ==================
# タイムゾーン（環境変数が無ければJST）
RAW_TZ = os.environ.get("RAW_TZ_INTRADAY", "Asia/Tokyo")
DISP_TZ = os.environ.get("DISPLAY_TZ", "Asia/Tokyo")

# 取引時間（JSTを想定、RAW_TZ/DISP_TZがJSTで無い場合も表示側は DISP_TZ に合わせます）
SESSION_START = os.environ.get("SESSION_START", "09:00")
SESSION_END   = os.environ.get("SESSION_END",   "15:30")

# ダークテーマ
BG = "#0E1117"
FG = "#E6E6E6"
ACCENT = "#3bd6c6"
TITLE  = "#f2b6c6"

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "text.color": FG,
    "grid.color": FG,
    "savefig.facecolor": BG,
})

# ================== ユーティリティ ==================
def lower_cols(df: pd.DataFrame) -> List[str]:
    cols = [str(c) if c is not None else "" for c in df.columns]
    df.columns = [c.strip() for c in cols]
    low = [c.lower() for c in df.columns]
    return low

def find_time_col(cols_lower: List[str]) -> Optional[str]:
    # 典型パターンを優先
    for k in ("time", "timestamp", "datetime", "date"):
        if k in cols_lower:
            return cols_lower[cols_lower.index(k)]

    # あいまい一致
    for i, c in enumerate(cols_lower):
        if ("time" in c) or ("date" in c):
            return cols_lower[i]

    # Unnamed: 0 / インデックス列
    for i, c in enumerate(cols_lower):
        if c.startswith("unnamed") and (": 0" in c or c.endswith("0")):
            return cols_lower[i]

    # 列名が空 or 先頭列が時間列のことがある
    if len(cols_lower) > 0:
        return cols_lower[0]

    return None

def to_disp_tz(ts: pd.Series) -> pd.Series:
    # 何が来ても表示TZに
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    return dt.dt.tz_convert(DISP_TZ)

def parse_time_any(series: pd.Series) -> pd.Series:
    """
    tz-naive/aware混在・フォーマット混在でもOKにする共通化。
    RAW_TZ を基準にして DISP_TZ へ。
    """
    # まず datetime へ（UTC扱い）
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    # tz情報が無い（NaTで落ちた）部分は RAW_TZ でローカライズしてUTCへ
    # ここは series を直接再解釈したほうが精度が良いので個別処理
    mask_nat = dt.isna()
    if mask_nat.any():
        dt2 = pd.to_datetime(series[mask_nat], errors="coerce")
        # tz-naiveは RAW_TZ でローカライズ → UTC → DISP_TZ
        dt2 = dt2.dt.tz_localize(RAW_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        dt[mask_nat] = dt2

    return dt.dt.tz_convert(DISP_TZ)

def read_any_intraday(path: Path) -> pd.DataFrame:
    """
    intraday CSV を読み込み、必ず ["time","value","volume"] で返す。
    time列は find_time_col で厳密/あいまい/Unnamed/先頭列 まで探す。
    value が無ければ板形式とみなし、数値列の等加重平均。
    """
    if not path.exists():
        return pd.DataFrame(columns=["time", "value", "volume"])

    df = pd.read_csv(path, dtype=str)
    # コメント列(#...) の除外
    drop = [c for c in df.columns if str(c).strip().startswith("#")]
    if drop:
        df = df.drop(columns=drop)

    cols_lower = lower_cols(df)
    tcol_name = find_time_col(cols_lower)
    if tcol_name is None:
        print(f"[WARN] time-like column not found in {path.name}. columns={list(df.columns)}")
        return pd.DataFrame(columns=["time", "value", "volume"])

    # 実列名（大小区別あり）を拾い直す
    real_tcol = df.columns[cols_lower.index(tcol_name)]

    # time 変換
    out = pd.DataFrame()
    out["time"] = parse_time_any(df[real_tcol])

    # 単一 value/volume の検出
    vcol = None
    volcol = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("value", "index", "score") or ("value" in lc):
            vcol = c
        if lc == "volume" or ("volume" in lc):
            volcol = c

    if vcol is not None:
        out["value"] = pd.to_numeric(df[vcol], errors="coerce")
        out["volume"] = pd.to_numeric(df[volcol], errors="coerce") if (volcol and volcol in df.columns) else 0
    else:
        # 板形式 → time以外の数値列を平均
        num_cols: List[str] = []
        for c in df.columns:
            if c == real_tcol:
                continue
            as_num = pd.to_numeric(df[c], errors="coerce")
            if as_num.notna().sum() > 0:
                num_cols.append(c)
        if not num_cols:
            return pd.DataFrame(columns=["time", "value", "volume"])
        vals = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out["value"] = vals.mean(axis=1)
        out["volume"] = 0

    out = out.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    return out

def clamp_session_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["time"].dt.tz_convert(DISP_TZ)
    today = pd.Timestamp.now(tz=DISP_TZ).normalize()
    start = pd.to_datetime(f"{today.date()} {SESSION_START}", utc=False).tz_localize(DISP_TZ)
    end   = pd.to_datetime(f"{today.date()} {SESSION_END}",   utc=False).tz_localize(DISP_TZ)
    m = (t >= start) & (t <= end)
    return df.loc[m].reset_index(drop=True)

def resample_minutes(df: pd.DataFrame, rule="1min") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.set_index("time").sort_index()
    out = tmp[["value"]].resample(rule).mean()
    out["value"] = out["value"].interpolate(limit_direction="both")
    out["volume"] = 0
    return out.reset_index()

def decorate(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TITLE, fontsize=20, pad=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    for sp in ax.spines.values():
        sp.set_color(FG)

def save_png(fig: plt.Figure, out_path: Path):
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

def read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

def plot_history(ax, hist: pd.DataFrame, title: str):
    decorate(ax, title, "Date", "Index Value")
    if len(hist) >= 2:
        ax.plot(hist["date"], hist["value"], linewidth=2.2, color=ACCENT)
    elif len(hist) == 1:
        ax.plot(hist["date"], hist["value"], marker="o", linewidth=0, color=ACCENT)
        y = hist["value"].iloc[0]
        ax.set_ylim(y - 0.1, y + 0.1)
        ax.text(0.5, 0.5, "Only 1 point (need ≥ 2)", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)

def main():
    index_key = os.environ.get("INDEX_KEY", "astra4").strip().lower()
    index_name = index_key.upper().replace("_", "")
    outputs = Path("docs/outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    intraday_csv = outputs / f"{index_key}_intraday.csv"
    history_csv  = outputs / f"{index_key}_history.csv"

    # ---- 1d ----
    try:
        intraday = read_any_intraday(intraday_csv)
        intraday = clamp_session_today(intraday)
        intraday = resample_minutes(intraday, "1min")
    except Exception as e:
        print(f"[WARN] intraday load failed: {e}")
        intraday = pd.DataFrame(columns=["time", "value", "volume"])

    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    decorate(ax, f"{index_name} (1d)", "Time", "Index Value")
    if not intraday.empty:
        ax.plot(intraday["time"], intraday["value"], linewidth=2.4, color=ACCENT)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
    save_png(fig, outputs / f"{index_key}_1d.png")

    # ---- 7d / 1m / 1y ----
    hist = read_history(history_csv)

    # 7d
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history(ax, hist.tail(7), f"{index_name} (7d)")
    save_png(fig, outputs / f"{index_key}_7d.png")

    # 1m（30日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history(ax, hist.tail(30), f"{index_name} (1m)")
    save_png(fig, outputs / f"{index_key}_1m.png")

    # 1y（365日）
    fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
    plot_history(ax, hist.tail(365), f"{index_name} (1y)")
    save_png(fig, outputs / f"{index_key}_1y.png")

    # 実行記録
    (outputs / "_last_run.txt").write_text(pd.Timestamp.now(tz=DISP_TZ).isoformat())
    print("[OK] charts generated.")

if __name__ == "__main__":
    main()
