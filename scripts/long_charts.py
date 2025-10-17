#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats
- 1d も 7d/1m/1y も「指数レベル（level）」をそのまま描画
- 騰落表示は level の“差分”で統一（%にしない）
- ダークテーマ、動的ライン色
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
FLAT    = "#9aa3af"  # ゼロ近傍・判定不能

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, alpha=0.4, linewidth=0.8)
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)

def _trend_color(series: pd.Series, mode: str) -> str:
    """
    線色の判定:
      - mode="intraday": 最後と最初の“差分”（last-first）で判定
      - mode="window"  : 期間の純変化（last-first）
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return FLAT
    delta = float(s.iloc[-1]) - float(s.iloc[0])
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

def _save(df: pd.DataFrame, col: str, out_png: Path, title: str, mode: str) -> None:
    if df.empty:
        return
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
    ASTRA4 本体列を推定。既知が無ければ最後の列。
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

def _load_df_any() -> pd.DataFrame:
    """
    intraday があれば intraday 優先。無ければ history。
    先頭列を DatetimeIndex に、数値列へ変換して NA 行を落とす。
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

def _load_df_intraday() -> pd.DataFrame | None:
    if not INTRADAY_CSV.exists():
        return None
    df = pd.read_csv(INTRADAY_CSV, parse_dates=[0], index_col=0)
    for c in list(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")

# ------------------------
# chart generation
# ------------------------
def gen_pngs() -> None:
    df_all = _load_df_any()
    col = _pick_index_column(df_all)

    # 1d: intraday があればそれを、無ければ直近 1d 相当の tail
    df_1d = _load_df_intraday()
    if df_1d is None:
        df_1d = df_all.tail(1000)

    # 7d は tail で簡易に（実データ密度は指数に依存）
    df_7d = df_all.tail(7 * 1000)

    _save(df_1d, col, OUTDIR / f"{INDEX_KEY}_1d.png", f"{INDEX_KEY.upper()} (1d level)", mode="intraday")
    _save(df_7d, col, OUTDIR / f"{INDEX_KEY}_7d.png", f"{INDEX_KEY.upper()} (7d)", mode="window")
    _save(df_all, col, OUTDIR / f"{INDEX_KEY}_1m.png", f"{INDEX_KEY.upper()} (1m)", mode="window")
    _save(df_all, col, OUTDIR / f"{INDEX_KEY}_1y.png", f"{INDEX_KEY.upper()} (1y)", mode="window")

# ------------------------
# stats (level) + marker
# ------------------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _first_valid_and_last(s: pd.Series) -> tuple[pd.Timestamp|None, float|None, pd.Timestamp|None, float|None]:
    s_num = pd.to_numeric(s, errors="coerce")
    valid = s_num.dropna()
    if valid.empty:
        return None, None, None, None
    first_ts = valid.index[0]
    last_ts  = valid.index[-1]
    return first_ts, float(valid.iloc[0]), last_ts, float(valid.iloc[-1])

def write_stats_and_marker() -> None:
    """
    1d の変化量は「intraday の最初の有効値 → 最後の有効値」の level 差分。
    intraday が無ければ history の tail で代替。
    JSON: {"pct_1d": null, "delta_level": <float>, "scale": "level"}
    TXT : "ASTRA4 1d: Δ=+0.001234 (level) (basis first-row valid=...->...)"
    """
    # まず intraday を優先
    df_intra = _load_df_intraday()
    df_any   = _load_df_any()
    col      = _pick_index_column(df_any)

    basis_from, basis_val, last_ts, last_val = (None, None, None, None)

    if df_intra is not None and not df_intra.empty:
        fts, fv, lts, lv = _first_valid_and_last(df_intra[col])
        basis_from, basis_val, last_ts, last_val = fts, fv, lts, lv

    # intraday が不十分なら全体 df で代替
    if basis_val is None or last_val is None:
        fts, fv, lts, lv = _first_valid_and_last(df_any[col])
        basis_from, basis_val, last_ts, last_val = fts, fv, lts, lv

    delta = None
    if basis_val is not None and last_val is not None:
        delta = last_val - basis_val

    # JSON（% は使わない）
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,                      # ％は出さない
        "delta_level": None if delta is None else round(delta, 6),
        "scale": "level",
        "updated_at": _now_utc_iso(),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # TXT マーカー
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    if delta is None or basis_from is None or last_ts is None:
        marker.write_text(f"{INDEX_KEY.upper()} 1d: N/A (level)\n", encoding="utf-8")
    else:
        marker.write_text(
            f"{INDEX_KEY.upper()} 1d: Δ={delta:+.6f} (level) "
            f"(basis first-row valid={basis_from.isoformat()}->{last_ts.isoformat()})\n",
            encoding="utf-8",
        )

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs()
    write_stats_and_marker()
