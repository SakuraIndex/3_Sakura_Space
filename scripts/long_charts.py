#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats
- y軸: level（元データの値）
- 1d注記: level差(Δ) と パーセンテージポイント(Δ% = Δ*100) を併記
- ダークテーマ + 上下で線色を自動切替
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt

# ====== const / paths ======
INDEX_KEY = "astra4"
OUTDIR = Path("docs/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV  = OUTDIR / f"{INDEX_KEY}_history.csv"
INTRADAY_CSV = OUTDIR / f"{INDEX_KEY}_intraday.csv"

# ====== styling (dark) ======
DARK_BG = "#0e0f13"
DARK_AX = "#0b0c10"
FG_TEXT = "#e7ecf1"
GRID    = "#2a2e3a"
RED     = "#ff6b6b"
GREEN   = "#28e07c"
FLAT    = "#9aa3af"

def _apply(ax, title: str) -> None:
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
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(first: float, last: float) -> str:
    if pd.isna(first) or pd.isna(last):
        return FLAT
    if last > first:
        return GREEN
    if last < first:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str) -> None:
    # 線色は期間純変化で判定
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    first = s.iloc[0] if len(s) else float("nan")
    last  = s.iloc[-1] if len(s) else float("nan")
    color = _trend_color(first, last)

    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ====== data helpers ======
def _pick_index_column(df: pd.DataFrame) -> str:
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

# ====== charts ======
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)")
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m level)")
    _save(df,      col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y level)")

# ====== stats (level + percent points) ======
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _delta_level_and_pp(s: pd.Series):
    """levelの純変化と、そのパーセンテージポイント(Δ% = Δ*100)を返す。"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None, None, None, None
    first = float(s.iloc[0])
    last  = float(s.iloc[-1])
    delta = last - first
    delta_pp = delta * 100.0  # percentage points
    return first, last, delta, delta_pp

def write_stats_and_marker() -> None:
    df = _load_df()
    col = _pick_index_column(df)
    s = pd.to_numeric(df[col], errors="coerce").dropna()

    first, last, delta_level, delta_pp = _delta_level_and_pp(s)
    # JSON は level と Δ%（pp）を両方持つ。サイト側は scale="level" を見て解釈。
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None if delta_pp is None else round(delta_pp, 6),  # Δ%（pp）
        "delta_level": None if delta_level is None else round(delta_level, 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # ポスト用の一行メモ（level と Δ%（pp）を併記）
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta_level is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: Δ=N/A (level)  Δ%=N/A\n", encoding="utf-8")
    else:
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta_level:+.6f} (level)  Δ%={delta_pp:+.2f}% "
            f"(basis first-row valid={s.index[0].isoformat()}->{s.index[-1].isoformat()})\n",
            encoding="utf-8",
        )

# ====== main ======
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
