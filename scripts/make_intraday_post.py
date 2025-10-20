#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday snapshot & post generator (robust datetime autodetect)

- CSV の日時列を自動検出（--dt-col が存在しなくても動く）
- JST 時間帯に統一
- --value-type=ratio/percent をサポート（内部は % で処理）
- スナップショットPNG生成（黒背景 + シアン線）、テキスト/JSON出力
"""

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


# ---------- Utilities ----------

def _try_parse_col_as_datetime(s: pd.Series) -> pd.Series:
    """各列を datetime に変換（失敗は NaT）。返り値は tz-aware(JST) に揃える。"""
    dt = pd.to_datetime(s, errors="coerce")
    if dt.dt.tz is None:
        # tz naive は JST と仮定（国内CSV多くがJST naive）
        dt = dt.dt.tz_localize(JST)
    else:
        dt = dt.dt.tz_convert(JST)
    return dt


def _autodetect_dt_col(raw: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """優先候補→スコアリングの順で日時列を検出して列名を返す。見つからなければ None。"""
    # 1) 優先候補名で存在するものを試す
    for c in candidates:
        if c in raw.columns:
            dt = pd.to_datetime(raw[c], errors="coerce")
            valid = dt.notna().mean()
            if valid >= 0.8:  # 8割以上解釈できればOK
                return c
    # 2) すべての列を試して最もNaTが少ない列を採用
    best_col, best_valid = None, -1.0
    for c in raw.columns:
        dt = pd.to_datetime(raw[c], errors="coerce")
        valid = dt.notna().mean()
        if valid > best_valid:
            best_col, best_valid = c, valid
    if best_valid >= 0.8:
        return best_col
    return None


def to_jst_index(raw: pd.DataFrame, dt_col_opt: Optional[str]) -> pd.DataFrame:
    """日時列を検出して index に据え、JST tz-aware に統一。"""
    # 候補
    candidates = ["Datetime", "datetime", "Timestamp", "timestamp",
                  "Date", "date", "Time", "time"]

    dt_col = None
    if dt_col_opt and dt_col_opt in raw.columns:
        dt_col = dt_col_opt
    else:
        dt_col = _autodetect_dt_col(raw, candidates)

    if dt_col is None:
        raise ValueError("CSV内の日時列を自動検出できませんでした。--dt-col で列名を指定してください。"
                         f" 既存列: {list(raw.columns)}")

    dt = _try_parse_col_as_datetime(raw[dt_col])
    out = raw.copy()
    out.index = dt
    out = out.drop(columns=[dt_col])
    out = out.sort_index()
    return out


def filter_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """JST の HH:MM 範囲で直近日のセッションを抽出。"""
    if df_jst.empty:
        return df_jst
    day = df_jst.index[-1].date()
    start_ts = pd.Timestamp(f"{day} {start}", tz=JST)
    end_ts = pd.Timestamp(f"{day} {end}", tz=JST)
    return df_jst.loc[(df_jst.index >= start_ts) & (df_jst.index <= end_ts)]


def coerce_series_as_percent(series: pd.Series, value_type: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if value_type == "ratio":
        s = s * 100.0
    return s


def make_title_label(index_key: str) -> str:
    return index_key.upper()


# ---------- CLI ----------

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
    value_type: str


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
    p.add_argument("--dt-col", default=None, help="日時列名（未指定OK。自動検出します）")
    p.add_argument("--value-type", choices=["ratio", "percent"], default="percent",
                   help="ratio=0.035→3.5, percent=3.5→3.5 として内部%扱い")
    a = p.parse_args()
    return Args(
        index_key=a.index_key,
        csv=a.csv,
        out_json=a.out_json,
        out_text=a.out_text,
        snapshot_png=a.snapshot_png,
        session_start=a.session_start,
        session_end=a.session_end,
        day_anchor=a.day_anchor,
        basis=a.basis,
        label=a.label,
        dt_col=a.dt_col,
        value_type=a.value_type,
    )


# ---------- Core ----------

def summarize_and_plot(
    df_sess: pd.DataFrame,
    index_key: str,
    title_label: str,
    basis_label: str,
    snapshot_png: str,
) -> Tuple[float, pd.Timestamp]:

    if index_key not in df_sess.columns:
        raise ValueError(f"CSV に対象列 '{index_key}' がありません。列={list(df_sess.columns)}")

    s = pd.to_numeric(df_sess[index_key], errors="coerce").dropna()
    if s.empty:
        raise ValueError("セッション内データがありません。")

    # === plot ===
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    for sp in ax.spines.values():
        sp.set_color("#444444")

    ax.plot(s.index, s.values, linewidth=2.0, color="#00E5FF", label=title_label)
    ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#DDDDDD")

    ax.set_title(f"{title_label} Intraday Snapshot ({pd.Timestamp.now(tz=JST):%Y/%m/%d %H:%M})",
                 color="#DDDDDD")
    ax.set_xlabel("Time", color="#BBBBBB")
    ax.set_ylabel("Change vs Prev Close (%)", color="#BBBBBB")
    ax.tick_params(colors="#BBBBBB")
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(snapshot_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return float(s.iloc[-1]), s.index[-1]


def main() -> None:
    args = parse_args()

    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, args.dt_col)

    if args.index_key not in df.columns:
        raise ValueError(f"CSV に対象列 '{args.index_key}' がありません。列={list(df.columns)}")

    df[args.index_key] = coerce_series_as_percent(df[args.index_key], args.value_type)

    df_sess = filter_session(df, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    title_label = args.label if args.label else make_title_label(args.index_key)
    basis_label = args.basis

    last_pct, last_ts = summarize_and_plot(
        df_sess, args.index_key, title_label, basis_label, args.snapshot_png
    )

    sign = "+" if last_pct >= 0 else ""
    lines = [
        f"▲ {title_label} 日中スナップショット ({last_ts.tz_convert(JST):%Y/%m/%d %H:%M})",
        f"{sign}{last_pct:.2f}% (基準: {basis_label})",
        f"#{title_label} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    stats = {
        "index_key": title_label,
        "label": title_label,
        "pct_intraday": last_pct,
        "basis": basis_label,
        "session": {
            "start": args.session_start,
            "end": args.session_end,
            "anchor": args.day_anchor,
        },
        "updated_at": f"{pd.Timestamp.now(tz=JST).isoformat()}",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
