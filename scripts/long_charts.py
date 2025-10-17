#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats (level 指数)
- ダークテーマ
- 1d/7d/1m/1y を出力
- 線色は「期間の純変化 (last-first)」で判定（上昇=緑/下落=赤/フラット=グレー）
- マーカー/JSONは level と % を併記。ただし始値が小さすぎる場合は % を N/A にする
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# constants / paths
# =========================
INDEX_KEY = "astra4"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# =========================
# plotting (dark)
# =========================
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#222632"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _apply(ax: plt.Axes, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.35, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=13, pad=8)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(first: float | None, last: float | None) -> str:
    if first is None or last is None:
        return FLAT
    delta = last - first
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

# =========================
# data
# =========================
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    ASTRA4 本体列を推定。既知候補が無ければ最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {"astra4", "astra4mean", "astra4index", "spaceindex", "sakura4"}
    for c in df.columns:
        if norm(c) in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("ASTRA4: neither intraday nor history csv found.")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

def _first_last_valid(series: pd.Series) -> tuple[datetime|None, datetime|None, float|None, float|None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None, None, None
    first_ts = s.index[0].to_pydatetime()
    last_ts  = s.index[-1].to_pydatetime()
    return first_ts, last_ts, float(s.iloc[0]), float(s.iloc[-1])

def _delta_level_and_pct(first_v: float, last_v: float) -> tuple[float, float | None]:
    """
    level 型の 1d 変化量:
      - delta_level = last - first
      - delta_pct   = (last/first - 1) * 100      （ただし |first| が小さすぎる時は None）
    """
    delta_level = last_v - first_v
    if abs(first_v) < 0.5:               # 0 近傍で%が暴れるのを抑制
        delta_pct = None
    else:
        delta_pct = (last_v / first_v - 1.0) * 100.0
    return delta_level, delta_pct

# =========================
# charts
# =========================
def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots()
    _apply(ax, title)
    first_ts, last_ts, first_v, last_v = _first_last_valid(s)
    ax.plot(s.index, s.values, color=_trend_color(first_v, last_v), linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 可読性重視でシンプルに tail で近傍を表示
    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)")

# =========================
# stats + marker
# =========================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    first_ts, last_ts, first_v, last_v = _first_last_valid(s)

    delta_level: float | None = None
    delta_pct:   float | None = None

    if first_ts is not None:
        delta_level, delta_pct = _delta_level_and_pct(first_v, last_v)

    # JSON
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if delta_pct is None else round(float(delta_pct), 6),
        "delta_level": None if delta_level is None else round(float(delta_level), 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # Marker（レベルと％併記。％は N/A の場合あり）
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if first_ts is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A (level)\n", encoding="utf-8")
    else:
        pct_str = "N/A" if delta_pct is None else f"{delta_pct:+.2f}%"
        marker.write_text(
            (
                f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={pct_str} "
                f"(basis first-row valid={first_ts.isoformat()}->{last_ts.isoformat()})\n"
            ),
            encoding="utf-8",
        )

# =========================
# main
# =========================
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
