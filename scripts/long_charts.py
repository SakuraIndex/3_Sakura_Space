#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats
- 1日の最初の有効値で正規化して "level" を作成（基準=1.0）
- ダークテーマ
- 線色は上昇=GREEN / 下降=RED / フラット=FLAT を自動判定
- 出力:
    docs/outputs/astra4_1d.png, _7d.png, _1m.png, _1y.png
    docs/outputs/astra4_stats.json   {"index_key","pct_1d":null,"delta_level", "scale":"level","updated_at"}
    docs/outputs/astra4_post_intraday.txt  "ASTRA4 1d: Δ=+x.xx% (level) (basis first-row valid=...->...)"
"""
from __future__ import annotations

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
GRID    = "#262a36"
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
    ax.set_title(title, color=FG_TEXT, fontsize=12, pad=10)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(series: pd.Series, mode: str) -> str:
    """
    線色の判定:
      - mode="intraday": 基準=1.0 として最後の level の符号（last-1）
      - mode="window"  : 期間の純変化 (last - first) で判定
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return FLAT

    if mode == "intraday":
        last = s.iloc[-1]
        delta = last - 1.0
        if delta > 0:
            return GREEN
        if delta < 0:
            return RED
        return FLAT

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
    ax.plot(df.index, df[col], color=color, linewidth=1.8)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

# ------------------------
# data loading helpers
# ------------------------
def _pick_index_column(df: pd.DataFrame) -> str:
    """
    ASTRA4 本体列を推定。既知候補が無ければ最後の列。
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {
        "astra4", "astra4mean", "astra4index", "spaceindex", "sakura4"
    }
    for c in df.columns:
        if norm(c) in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """
    intraday があれば intraday 優先、無ければ history。
    先頭列を DatetimeIndex に、数値列へ強制変換して NA 行は落とす。
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
    # index は昇順に保証
    df = df.sort_index()
    return df

# ------------------------
# level series (normalize by first valid of the day)
# ------------------------
def _build_level_series(df: pd.DataFrame, col: str) -> tuple[pd.Series, datetime, datetime]:
    """
    1日の intraday がある場合は「そのファイル内の最初の有効値」を基準（=1.0）に正規化。
    無ければ df 全体の最後のカレンダーデイの最初の有効値を基準にする。
    戻り値: (level_series, first_ts, last_ts)
    """
    s_all = pd.to_numeric(df[col], errors="coerce")

    # 使用する日付範囲を決める
    if INTRADAY_CSV.exists():
        s = s_all.dropna()
    else:
        # history のときは最後のカレンダーデイだけを抽出
        s_nonan = s_all.dropna()
        if s_nonan.empty:
            return s_nonan, None, None
        last_day = s_nonan.index.max().date()
        s = s_nonan[s_nonan.index.date == last_day]

    s = s.dropna()
    if s.empty:
        return s, None, None

    first_val = float(s.iloc[0])
    # 0 divide を避ける
    if first_val == 0.0:
        # 何も描けないが、落ちないようフラットを返す
        level = pd.Series([1.0] * len(s.index), index=s.index)
    else:
        level = s / first_val

    return level, s.index[0].to_pydatetime(), s.index[-1].to_pydatetime()

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    # 1d 用 level を作る（当日の範囲）
    level_1d, first_ts, last_ts = _build_level_series(df, col)

    # 7d/1m/1y 用：history / intraday をまとめて level 化する
    # ルール: 全期間で「各日の最初の有効値」での正規化は行わず、
    #         直列全体で最初の有効値を基準に 1 本の level にする
    s_all = pd.to_numeric(df[col], errors="coerce").dropna()
    level_all = None
    if not s_all.empty:
        base = float(s_all.iloc[0])
        level_all = (s_all / base) if base != 0.0 else pd.Series([1.0] * len(s_all), index=s_all.index)

    # --- 保存 ---
    if not level_1d.empty:
        _save(level_1d.to_frame("level"), "level",
              OUTDIR / f"{INDEX_KEY}_1d.png",
              f"{INDEX_KEY.upper()} (1d level)", mode="intraday")

    if level_all is not None and not level_all.empty:
        tail_7d = level_all.tail(7 * 1000)
        _save(tail_7d.to_frame("level"), "level",
              OUTDIR / f"{INDEX_KEY}_7d.png",
              f"{INDEX_KEY.upper()} (7d)", mode="window")

        _save(level_all.to_frame("level"), "level",
              OUTDIR / f"{INDEX_KEY}_1m.png",
              f"{INDEX_KEY.upper()} (1m)", mode="window")

        _save(level_all.to_frame("level"), "level",
              OUTDIR / f"{INDEX_KEY}_1y.png",
              f"{INDEX_KEY.upper()} (1y)", mode="window")

# ------------------------
# stats (level) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def write_stats_and_marker() -> None:
    """
    出力ポリシー（R-BANK9 と同一）:
      - pct_1d は使わない（null 固定）
      - delta_level = 最終 level の値（= close/open の倍率, 例 1.0723）
      - ポスト用テキストは Δ=(delta_level-1)*100 を小数2桁の % で表示
      - basis として first/last の有効タイムスタンプを併記
    """
    df = _load_df()
    col = _pick_index_column(df)

    level_1d, first_ts, last_ts = _build_level_series(df, col)

    delta_level = None
    if not level_1d.empty:
        delta_level = float(level_1d.iloc[-1])

    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,                              # 仕様どおり null
        "delta_level": None if delta_level is None else round(delta_level, 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta_level is None or first_ts is None or last_ts is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A (level) (basis invalid)\n", encoding="utf-8")
    else:
        pct = (delta_level - 1.0) * 100.0
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={pct:+.2f}% (level) "
            f"(basis first-row valid={first_ts.isoformat()}->{last_ts.isoformat()})\n",
            encoding="utf-8"
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
