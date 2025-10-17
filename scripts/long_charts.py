#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats
 - 縦軸は「level」をそのまま描画
 - 1dの注記は Δ(level) と Δ%(signed) を併記
 - Δ%(signed) は「基準の絶対値」を分母にして符号はΔ(level)の符号に合わせる
 - 0/欠損/符号またぎ時は Δ% を N/A にフォールバック
 - ダークテーマ、線色は上げ:緑 / 下げ:赤 / 横ばい:グレー
"""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# constants / paths
# ------------------------
INDEX_KEY = "astra4"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

PNG_1D = OUTDIR / f"{INDEX_KEY}_1d.png"
PNG_7D = OUTDIR / f"{INDEX_KEY}_7d.png"
PNG_1M = OUTDIR / f"{INDEX_KEY}_1m.png"
PNG_1Y = OUTDIR / f"{INDEX_KEY}_1y.png"

MARKER_TXT = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
STATS_JSON = OUTDIR / f"{INDEX_KEY}_stats.json"

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _style(ax: plt.Axes, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)

    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.8)

    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)

    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(first: float, last: float) -> str:
    if np.isnan(first) or np.isnan(last):
        return FLAT
    if last > first:
        return GREEN
    if last < first:
        return RED
    return FLAT

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    ASTRA4 の列名を推定。候補が無ければ最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {
        "astra4", "astra4mean", "astra4index", "spaceindex", "sakura4"
    }
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex に、数値列へ強制変換して NA 行を除去。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("ASTRA4: neither intraday nor history csv found.")

    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    return df

# ------------------------
# math helpers
# ------------------------
def _first_last_valid(s: pd.Series) -> Tuple[Optional[pd.Timestamp], float, Optional[pd.Timestamp], float]:
    """系列の有効な最初と最後の (時刻, 値) を返す。無ければ (None, nan, None, nan)。"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None, np.nan, None, np.nan
    return s.index[0], float(s.iloc[0]), s.index[-1], float(s.iloc[-1])

def _signed_pct_change_from_level(first: float, last: float, eps: float = 1e-12) -> Optional[float]:
    """
    level→％への変換（符号一貫）
    - 分母は |first|
    - 符号は Δ(level) の符号
    - |first| ≦ eps のときや、first/last が欠損のときは None
    """
    if any(map(lambda x: np.isnan(x), (first, last))):
        return None
    if abs(first) <= eps:
        return None

    delta_level = last - first
    mag_change = abs(last) - abs(first)
    # 分母は |first|
    pct = (mag_change / abs(first)) * 100.0
    if delta_level < 0:
        pct = -abs(pct)
    elif delta_level > 0:
        pct = +abs(pct)
    else:
        pct = 0.0
    return float(pct)

# ------------------------
# chart generation
# ------------------------
def _save_line(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        # 空でも枠だけ出しておく
        fig, ax = plt.subplots()
        _style(ax, title)
        fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    first_t, first_v, last_t, last_v = _first_last_valid(s)
    color = _trend_color(first_v, last_v)

    fig, ax = plt.subplots()
    _style(ax, title)
    ax.plot(s.index, s.values, color=color, linewidth=1.7)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save_line(tail_1d, col, PNG_1D, f"{INDEX_KEY.upper()} (1d level)")
    _save_line(tail_7d, col, PNG_7D, f"{INDEX_KEY.upper()} (7d level)")
    _save_line(df,      col, PNG_1M, f"{INDEX_KEY.upper()} (1m level)")
    _save_line(df,      col, PNG_1Y, f"{INDEX_KEY.upper()} (1y level)")

# ------------------------
# stats + marker (level & %)
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    - Δ(level) は last - first（当日 intraday の有効最初→最後）
    - Δ%(signed) は |first| を分母、符号は Δ(level) に一致
    - 分母ゼロ/欠損は Δ% を N/A（stats.json の pct_1d は null）
    - scale は "level"
    """
    df = _load_df()
    col = _pick_index_column(df)

    # intraday の有効最初・最後
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    first_t, first_v, last_t, last_v = _first_last_valid(series)

    delta_level = float("nan")
    pct_signed: Optional[float] = None
    if first_t is not None and last_t is not None:
        delta_level = last_v - first_v
        pct_signed = _signed_pct_change_from_level(first_v, last_v)

    # JSON（サイト側が読む）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_signed is None else round(float(pct_signed), 6),
        "delta_level": None if np.isnan(delta_level) else round(float(delta_level), 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    STATS_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # 投稿用テキスト
    if first_t is None or last_t is None:
        MARKER_TXT.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A\n", encoding="utf-8")
    else:
        delta_str = f"{delta_level:+.6f}"
        if pct_signed is None:
            pct_str = "N/A"
        else:
            pct_str = f"{pct_signed:+.2f}%"

        basis = f"(basis first-row valid={first_t.isoformat()}->{last_t.isoformat()})"
        MARKER_TXT.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_str} (level)  Δ%={pct_str} {basis}\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
