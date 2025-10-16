#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats
  - dark theme
  - 1d/7d/1m/1y を level で描画
  - 1d の増減は percent ではなく「delta_level」を算出
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

def _trend_color(series: pd.Series, mode: str) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT
    if mode == "intraday":
        # 直近の値の増減（終値－始値）で判定
        delta = s.iloc[-1] - s.iloc[0]
        return GREEN if delta > 0 else RED if delta < 0 else FLAT
    # window: first/last の純増減で判定
    delta = s.iloc[-1] - s.iloc[0]
    return GREEN if delta > 0 else RED if delta < 0 else FLAT

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
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    tail_1d = df.tail(1000)
    tail_7d = df.tail(7 * 1000)

    # 1d：level（縦軸は指数の値）
    _save(tail_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png",
          f"{INDEX_KEY.upper()} (1d level)", mode="intraday")

    # 7d/1m/1y：level
    _save(tail_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png",
          f"{INDEX_KEY.upper()} (7d)", mode="window")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1m.png",
          f"{INDEX_KEY.upper()} (1m)", mode="window")
    _save(df, col, OUTDIR / f"{INDEX_KEY}_1y.png",
          f"{INDEX_KEY.upper()} (1y)", mode="window")

# ------------------------
# stats (level) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    仕様（ASTRA4）:
      - スケールは「level」
      - 1d の増減は percent ではなく absolute な delta_level (last - first)
      - 先頭/末尾は NaN を除いた「最初の有効値 / 最後の有効値」
    """
    df = _load_df()
    col = _pick_index_column(df)

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    marker_path = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"

    if s.empty:
        payload = {
            "index_key": INDEX_KEY, "pct_1d": None, "delta_level": None,
            "scale": "level", "updated_at": _now_utc_iso()
        }
        (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        marker_path.write_text(f"{INDEX_KEY.upper()} 1d: N/A (basis invalid)\n", encoding="utf-8")
        return

    first_ts = s.index[0]
    last_ts  = s.index[-1]
    first_v  = float(s.iloc[0])
    last_v   = float(s.iloc[-1])
    delta_lv = last_v - first_v

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,                  # percent は出さない
        "delta_level": round(delta_lv, 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker_path.write_text(
        f"{INDEX_KEY.upper()} 1d: Δ={delta_lv:+.4g} (level) "
        f"(basis first-row valid={first_ts.isoformat()}→{last_ts.isoformat()})\n",
        encoding="utf-8",
    )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
