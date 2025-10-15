#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats  (fraction→pct 統一, dark theme, dynamic line color)
"""
from pathlib import Path
import json
from datetime import datetime, timezone
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

# ------------------------
# plotting style (dark)
# ------------------------
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"  # ゼロ近傍・判定不能のとき

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index / Value", color=FG_TEXT, fontsize=10)

def _trend_color(series: pd.Series, mode: str) -> str:
    """
    線色の判定:
      - mode="intraday": 末尾の値の符号で (+)緑 / (-)赤 / 0はFLAT
      - mode="window"  : 期間の純変化 (last - first) で判定
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT

    if mode == "intraday":
        last = s.iloc[-1]
        if last > 0:
            return GREEN
        if last < 0:
            return RED
        return FLAT

    # window（7d/1m/1y）は first/last の有効値で純変化を見る
    first = s.iloc[0]
    last  = s.iloc[-1]
    delta = last - first
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str, mode: str) -> None:
    fig, ax = plt.subplots()
    _apply(ax, title)
    color = _trend_color(df[col], mode=mode)
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    ASTRA4 本体列を推定。既知の候補が無ければ最後の列を使う。
    例: 'Astra4_mean', 'astra4', 'ASTRA4'
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
    先頭列を DatetimeIndex に、数値列へ強制変換して NA を除去。
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
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    # 1d は「その時点の fraction（= 当日リターン）」→ 最後の符号で色判定
    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d)", mode="intraday")

    # 7d/1m/1y は純変化で色判定（上げ → 緑、下げ → 赤）
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", mode="window")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)", mode="window")

# ------------------------
# stats (pct) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    仕様:
      - ASTRA4 の intraday 値は “fraction”（例: -0.054 ≒ -5.4%）
      - サイト側は百分率を期待 → pct_1d は (last * 100) を出力
      - scale="pct" に統一
    """
    df = _load_df()
    col = _pick_index_column(df)

    pct = None
    if len(df.index) > 0:
        last = pd.to_numeric(df[col], errors="coerce").dropna()
        if not last.empty:
            pct = float(last.iloc[-1]) * 100.0

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct is None else round(pct, 6),
        "scale": "pct",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if pct is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A\n", encoding="utf-8")
    else:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: {pct:+.2f}%\n", encoding="utf-8")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
