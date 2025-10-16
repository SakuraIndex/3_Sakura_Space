#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA4 charts + stats  (level-scale version, dark theme, dynamic color)
  - 統一仕様: "Index (level)" 軸 / R-BANK9形式に準拠
  - 基準: intraday の最初の有効値
  - 色分け: 終値 > 始値 → 緑, 終値 < 始値 → 赤
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
GREEN   = "#22c55e"  # 上昇
RED     = "#ef4444"  # 下落
FLAT    = "#9aa3af"  # ゼロ・不明

def _apply(ax, title: str) -> None:
    fig = ax.figure
    fig.set_size_inches(12, 7)
    fig.set_dpi(160)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    ax.grid(False)  # グリッド線は非表示で視認性アップ（R-BANK9と同様）
    ax.tick_params(colors=FG_TEXT, labelsize=10)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title(title, color=FG_TEXT, fontsize=12)
    ax.set_xlabel("Time", color=FG_TEXT, fontsize=10)
    ax.set_ylabel("Index (level)", color=FG_TEXT, fontsize=10)
    for sp in ax.spines.values():
        sp.set_color(GRID)

def _pick_index_column(df: pd.DataFrame) -> str:
    """ASTRA4 本体列を自動特定"""
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    candidates = {"astra4", "astra4mean", "astra4index", "spaceindex", "sakura4"}
    ncols = {c: norm(c) for c in df.columns}
    for c, nc in ncols.items():
        if nc in candidates:
            return c
    return df.columns[-1]

def _load_df() -> pd.DataFrame:
    """intraday優先でロードし、DatetimeIndexに整形"""
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
# core logic (level)
# ------------------------
def first_valid_basis(series):
    """最初の有効値とその時刻を返す"""
    s = series.dropna()
    if s.empty:
        return None, None
    return s.index[0], float(s.iloc[0])

def to_level(series, basis_val: float):
    """値を基準値で正規化（level化）"""
    if basis_val is None or basis_val == 0.0:
        return pd.Series(index=series.index, dtype=float)
    return series / basis_val

def pick_color(open_lv: float, close_lv: float) -> str:
    """陽線=緑, 陰線=赤, 同値=FLAT"""
    if open_lv is None or close_lv is None:
        return FLAT
    delta = close_lv - open_lv
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return FLAT

# ------------------------
# plot & stats
# ------------------------
def gen_pngs_and_stats() -> None:
    df = _load_df()
    col = _pick_index_column(df)

    basis_ts, basis_val = first_valid_basis(df[col])
    if basis_val is None:
        print("⚠️ ASTRA4: 有効な基準値が見つかりません。")
        return

    df["level"] = to_level(df[col], basis_val)
    first_lv = df["level"].dropna().iloc[0]
    last_lv  = df["level"].dropna().iloc[-1]
    delta_level = last_lv - first_lv
    pct_1d = (last_lv / first_lv - 1.0) * 100.0

    color = pick_color(first_lv, last_lv)
    title = f"{INDEX_KEY.upper()} (1d level)"

    # 描画
    fig, ax = plt.subplots()
    _apply(ax, title)
    ax.plot(df.index, df["level"], color=color, linewidth=1.8)
    fig.savefig(OUTDIR / f"{INDEX_KEY}_1d.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 統計
    payload = {
        "index_key": INDEX_KEY,
        "pct_1d": None,  # level統一なのでnull
        "delta_level": round(delta_level, 6),
        "scale": "level",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (OUTDIR / f"{INDEX_KEY}_stats.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # テキストマーカー
    marker = OUTDIR / f"{INDEX_KEY}_post_intraday.txt"
    marker.write_text(
        f"{INDEX_KEY.upper()} 1d: Δ={pct_1d:+.2f}% "
        f"(basis first-row valid={basis_ts.isoformat()}->{df.index[-1].isoformat()})\n",
        encoding="utf-8"
    )

    print(f"✅ ASTRA4 updated: Δ={pct_1d:+.2f}%, level={last_lv:.3f}")

# ------------------------
# main
# ------------------------
if __name__ == "__main__":
    gen_pngs_and_stats()
