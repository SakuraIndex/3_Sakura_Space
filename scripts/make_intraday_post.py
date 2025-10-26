#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

JST = "Asia/Tokyo"

# -------------------- helpers (datetime) --------------------

def _try_parse_col_as_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(JST)
    else:
        dt = dt.dt.tz_convert(JST)
    return dt

def _autodetect_dt_col(raw: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in raw.columns:
            if pd.to_datetime(raw[c], errors="coerce").notna().mean() >= 0.8:
                return c
    best_col, best_valid = None, -1.0
    for c in raw.columns:
        v = pd.to_datetime(raw[c], errors="coerce").notna().mean()
        if v > best_valid:
            best_col, best_valid = c, v
    return best_col if best_valid >= 0.8 else None

def to_jst_index(raw: pd.DataFrame, dt_col_opt: Optional[str]) -> pd.DataFrame:
    candidates = ["Datetime","datetime","Timestamp","timestamp","Date","date","Time","time"]
    dt_col = dt_col_opt if (dt_col_opt and dt_col_opt in raw.columns) else _autodetect_dt_col(raw, candidates)
    if dt_col is None:
        raise ValueError(f"CSV内の日時列を自動検出できませんでした。--dt-col で明示してください。列={list(raw.columns)}")
    dt = _try_parse_col_as_datetime(raw[dt_col])
    out = raw.copy()
    out.index = dt
    out = out.drop(columns=[dt_col])
    return out.sort_index()

def filter_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df_jst.empty:
        return df_jst
    day = df_jst.index[-1].date()
    s = pd.Timestamp(f"{day} {start}", tz=JST)
    e = pd.Timestamp(f"{day} {end}", tz=JST)
    return df_jst.loc[(df_jst.index >= s) & (df_jst.index <= e)]

# -------------------- helpers (value column) --------------------

def _variations(key: str) -> List[str]:
    cand = [key, key.lower(), key.upper(), key.capitalize()]
    cand += [f"{x}_mean" for x in cand]
    return list(dict.fromkeys(cand))  # unique, keep order

def find_value_column(df: pd.DataFrame, index_key: str) -> str:
    cols = list(df.columns)
    # 1) 厳密／大小文字違い／_mean 付きの総当たり
    for name in _variations(index_key):
        if name in cols:
            return name
    # 2) "_mean" を含む列がちょうど1つならそれを採用
    mean_like = [c for c in cols if "_mean" in c.lower()]
    if len(mean_like) == 1:
        return mean_like[0]
    # 3) 数値列が1つだけならそれを採用（日時列は既に index 化済み）
    numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(f"CSV に対象列 '{index_key}' がありません。列={cols}")

def _detect_unit_is_ratio(s: pd.Series) -> bool:
    """95%点の絶対値が 0.5 未満なら ratio（= 0.5% 未満が多い）とみなす"""
    arr = pd.to_numeric(s, errors="coerce").to_numpy()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return False
    return float(np.quantile(np.abs(arr), 0.95)) < 0.5

def to_percent_series(series: pd.Series, value_type: str) -> pd.Series:
    """value_type: auto | ratio | percent"""
    s = pd.to_numeric(series, errors="coerce")
    vt = value_type.lower()
    if vt == "auto":
        vt = "ratio" if _detect_unit_is_ratio(s) else "percent"
    if vt == "ratio":
        s = s * 100.0
    return s

def make_title_label(index_key: str) -> str:
    return index_key.upper()

# -------------------- CLI --------------------

@dataclass
class Args:
    index_key: str
    csv: str
    out_json: str
    out_text: str
    snapshot_png: str
    session_start: str
    session_end: str
    day_anchor: str
    basis: str
    label: Optional[str]
    dt_col: Optional[str]
    value_type: str  # auto | percent | ratio

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-text", required=True)
    p.add_argument("--snapshot-png", required=True)
    p.add_argument("--session-start", required=True)
    p.add_argument("--session-end", required=True)
    p.add_argument("--day-anchor", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--dt-col", default=None)
    p.add_argument("--value-type", choices=["auto","ratio","percent"], default="auto")
    a = p.parse_args()
    return Args(
        index_key=a.index_key, csv=a.csv, out_json=a.out_json, out_text=a.out_text,
        snapshot_png=a.snapshot_png, session_start=a.session_start, session_end=a.session_end,
        day_anchor=a.day_anchor, basis=a.basis, label=a.label, dt_col=a.dt_col, value_type=a.value_type
    )

# -------------------- Core --------------------

def summarize_and_plot(
    df_sess: pd.DataFrame,
    value_col: str,
    title_label: str,
    basis_label: str,
    snapshot_png: str,
) -> Tuple[float, pd.Timestamp]:

    s = pd.to_numeric(df_sess[value_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError("セッション内データがありません。")

    last_pct = float(s.iloc[-1])

    # 線色：プラス→青 / マイナス→赤
    line_color = "#00E5FF" if last_pct >= 0 else "#FF4D4D"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    for sp in ax.spines.values():
        sp.set_color("#444444")

    ax.plot(s.index, s.values, linewidth=2.0, color=line_color, label=title_label)
    ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#DDDDDD")

    ax.set_title(
        f"{title_label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})  {last_pct:+.2f}%",
        color="#DDDDDD"
    )
    ax.set_xlabel("Time", color="#BBBBBB")
    ax.set_ylabel("Change vs Prev Close (%)", color="#BBBBBB")
    ax.tick_params(colors="#BBBBBB")
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(snapshot_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return last_pct, s.index[-1]

def main() -> None:
    args = parse_args()

    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, args.dt_col)

    value_col = find_value_column(df, args.index_key)
    # ％系列に統一（auto 判定対応）
    df[value_col] = to_percent_series(df[value_col], args.value_type)

    df_sess = filter_session(df, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    title_label = args.label if args.label else make_title_label(args.index_key)
    last_pct, last_ts = summarize_and_plot(
        df_sess, value_col, title_label, args.basis, args.snapshot_png
    )

    sign_arrow = "▲" if last_pct >= 0 else "▼"
    lines = [
        f"{sign_arrow} {title_label} 日中スナップショット ({last_ts.tz_convert(JST):%Y/%m/%d %H:%M})",
        f"{last_pct:+.2f}% (基準: {args.basis})",
        f"#{title_label} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # JSON は ratio（小数）で出力（Front と揃える）
    stats = {
        "index_key": args.index_key,
        "label": title_label,
        "pct_intraday": float(np.round(last_pct / 100.0, 6)),  # 例: -0.27% → -0.0027
        "basis": args.basis,
        "session": {"start": args.session_start, "end": args.session_end, "anchor": args.day_anchor},
        "updated_at": f"{pd.Timestamp.now(tz=JST).isoformat()}",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
