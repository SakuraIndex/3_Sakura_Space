#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats
  - y-axis: level (raw index)
  - 1d color: intraday delta sign
  - percent change: from first valid value of the day (guarded; else N/A)
  - dark theme (subtle grid)
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
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
# plotting (dark theme)
# =========================
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#242833"   # 控えめ
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _apply_dark(ax, title: str, y_label: str = "Index (level)") -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)

    ax.set_facecolor(DARK_AX)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel(y_label, color=FG_TEXT, fontsize=10)
    ax.tick_params(colors=FG_TEXT, labelsize=10)

    # スパインと薄いグリッド
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.6)

    # 科学表記は避ける
    ax.yaxis.get_major_formatter().set_scientific(False)

def _trend_color_from_delta(delta: float | None) -> str:
    if delta is None:
        return FLAT
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

# =========================
# data helpers
# =========================
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    ASTRA4 本体列を推定。既知候補が無い場合は最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {
        "astra4", "astra4index", "astra4mean", "spaceindex", "sakura4"
    }
    for c in df.columns:
        if norm(c) in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば優先、無ければ history。先頭列を DatetimeIndex に。
    """
    if INTRADAY_CSV.exists():
        df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    elif HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=[0], index_col=0)
    else:
        raise FileNotFoundError("ASTRA4: neither intraday nor history csv found.")

    # 強制 numeric + NA 落とし（全列 NaN 行は除外）
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df

@dataclass
class IntradayBasis:
    first_valid: float | None
    last_value: float | None
    start_iso: str | None
    end_iso: str | None

def _today_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    データ末尾の '日付' を当日と見なし、同日の行だけ抽出。
    """
    if df.empty:
        return df
    last_ts = df.index[-1]
    mask = (df.index.date == last_ts.date())
    return df.loc[mask]

def _intraday_basis(series: pd.Series) -> IntradayBasis:
    s = pd.to_numeric(series, errors="coerce")
    day = _today_slice(s.to_frame("v"))["v"].dropna()

    if day.empty:
        return IntradayBasis(None, None, None, None)

    first_valid = float(day.iloc[0])
    last_value  = float(day.iloc[-1])

    start_iso = day.index[0].astimezone(timezone.utc).isoformat(timespec="minutes")
    end_iso   = day.index[-1].astimezone(timezone.utc).isoformat(timespec="minutes")
    return IntradayBasis(first_valid, last_value, start_iso, end_iso)

# =========================
# charts
# =========================
def _save_line(df: pd.DataFrame, col: str, out_png: Path, title: str,
               color: str | None = None) -> None:
    fig, ax = plt.subplots()
    _apply_dark(ax, title, y_label="Index (level)")
    ax.plot(df.index, df[col], linewidth=1.6, color=color or FG_TEXT)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 1d 用に今日のスライス（なければ直近 ~1000 点）
    df_today = _today_slice(df)
    if df_today.empty:
        df_1d = df.tail(1000)
    else:
        df_1d = df_today

    # 1d の色は当日 Δlevel で決定
    basis = _intraday_basis(df[col])
    delta_level = None
    if basis.first_valid is not None and basis.last_value is not None:
        delta_level = basis.last_value - basis.first_valid
    color_1d = _trend_color_from_delta(delta_level)
    _save_line(df_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", color_1d)

    # 7d/1m/1y は区間の first→last の純変化で色決定
    def color_window(dfx: pd.DataFrame) -> str:
        s = pd.to_numeric(dfx[col], errors="coerce").dropna()
        if s.empty:
            return FLAT
        d = float(s.iloc[-1] - s.iloc[0])
        return _trend_color_from_delta(d)

    tail_7d = df.tail(7 * 1000)
    _save_line(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)", color_window(tail_7d))
    _save_line(df,       col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)", color_window(df))
    _save_line(df,       col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)", color_window(df))

# =========================
# stats (level + optional pct)
# =========================
SAFE_EPS = 1e-9

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _safe_pct(first_val: float | None, last_val: float | None) -> float | None:
    """
    (last / first - 1) * 100 をガード付きで返す。
    基準値が 0 近傍、NaN、どちらか欠損のときは None を返す。
    """
    if first_val is None or last_val is None:
        return None
    if abs(first_val) < SAFE_EPS:
        return None
    try:
        return (last_val / first_val - 1.0) * 100.0
    except Exception:
        return None

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    basis = _intraday_basis(df[col])

    delta_level = None
    pct_1d = None
    if basis.first_valid is not None and basis.last_value is not None:
        delta_level = basis.last_value - basis.first_valid
        pct_1d = _safe_pct(basis.first_valid, basis.last_value)

    # JSON
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if pct_1d is None else round(float(pct_1d), 6),
        "delta_level": None if delta_level is None else round(float(delta_level), 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    # Marker text (併記)
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    basis_note = "basis first-row invalid" if basis.first_valid is None else f"basis first-row valid={basis.start_iso}->{basis.end_iso}"
    if delta_level is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A ({basis_note})\n", encoding="utf-8")
    else:
        pct_str = "N/A" if pct_1d is None else f"{pct_1d:+.2f}%"
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={pct_str} ({basis_note})\n",
            encoding="utf-8",
        )

# =========================
# main
# =========================
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
