#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday snapshot & post generator

- CSV から対象列を読み込み（Datetime 列で時系列化）
- セッション時間で抽出（JST）
- 値の種類 (--value-type) を ratio/percent で受け取り、内部では % に揃える
- スナップショット画像（黒背景, シアン線）を生成
- 直近の値（%）をテキストと JSON に出力

想定 CSV 例:
Datetime,S-COIN+
2025-10-20 09:00:00,0.024  # ratio の例（2.4%）
...

呼び出し例（ASTRA4 の比率値を % として扱う）:
python scripts/make_intraday_post.py \
  --index-key astra4 \
  --csv docs/outputs/astra4_intraday.csv \
  --out-json docs/outputs/astra4_stats.json \
  --out-text docs/outputs/astra4_post_intraday.txt \
  --snapshot-png docs/outputs/astra4_intraday.png \
  --session-start 09:00 --session-end 15:30 \
  --day-anchor 09:00 \
  --basis prev_close \
  --value-type ratio
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# =========================
# Utilities
# =========================

JST = "Asia/Tokyo"


def to_jst_index(raw: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col not in raw.columns:
        raise ValueError(f"CSVに '{dt_col}' 列が見つかりません。")
    dt = pd.to_datetime(raw[dt_col], errors="coerce", utc=True)
    # CSV が naive の場合は UTC 前提で解釈 → JST へ変換
    # 既に tz-aware の場合も tz_convert で JST に
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC").dt.tz_convert(JST)
    else:
        dt = dt.dt.tz_convert(JST)
    out = raw.copy()
    out.index = dt
    out = out.drop(columns=[dt_col])
    out = out.sort_index()
    return out


def filter_session(df_jst: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """JST の時刻文字列 HH:MM で当日セッション抽出"""
    if df_jst.empty:
        return df_jst
    day = df_jst.index[-1].date()  # 直近日の営業日を採用
    start_ts = pd.Timestamp(f"{day} {start}", tz=JST)
    end_ts = pd.Timestamp(f"{day} {end}", tz=JST)
    mask = (df_jst.index >= start_ts) & (df_jst.index <= end_ts)
    return df_jst.loc[mask]


def coerce_series_as_percent(series: pd.Series, value_type: str) -> pd.Series:
    """ratio(0.035)→% (3.5) / percent(3.5)→% (3.5)"""
    s = pd.to_numeric(series, errors="coerce")
    if value_type == "ratio":
        s = s * 100.0
    # percent はそのまま
    return s


def make_title_label(index_key: str) -> str:
    # 画面や出力で見せたい表記（自由に調整してOK）
    return index_key.upper()


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
    dt_col: str
    value_type: str


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--index-key", required=True, help="CSV の対象列名")
    p.add_argument("--csv", required=True, help="入力 CSV パス")
    p.add_argument("--out-json", required=True, help="統計 JSON 出力先")
    p.add_argument("--out-text", required=True, help="投稿テキスト出力先")
    p.add_argument("--snapshot-png", required=True, help="スナップショット PNG 出力先")

    p.add_argument("--session-start", required=True, help="JST HH:MM")
    p.add_argument("--session-end", required=True, help="JST HH:MM")
    p.add_argument("--day-anchor", required=True, help="凡例用の表示アンカー (JST HH:MM)")

    p.add_argument("--basis", required=True, help="基準ラベル（例: prev_close / open@09:00 など表示用）")
    p.add_argument(
        "--label",
        default=None,
        help="チャートタイトルのラベル（未指定なら index_key を大文字化）",
    )
    p.add_argument(
        "--dt-col",
        default="Datetime",
        help="日時列名（デフォルト: Datetime）",
    )
    p.add_argument(
        "--value-type",
        dest="value_type",
        choices=["ratio", "percent"],
        default="percent",
        help="入力値の型: ratio=0.035→3.5%, percent=3.5→3.5%",
    )

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


# =========================
# Core
# =========================

def summarize_and_plot(
    df_sess: pd.DataFrame,
    index_key: str,
    title_label: str,
    basis_label: str,
    snapshot_png: str,
) -> Tuple[float, pd.Timestamp]:
    """描画して最新値(%)と最新時刻を返す"""
    if index_key not in df_sess.columns:
        raise ValueError(f"CSV に対象列 '{index_key}' がありません。列={list(df_sess.columns)}")

    s = pd.to_numeric(df_sess[index_key], errors="coerce")
    s = s.dropna()
    if s.empty:
        raise ValueError("セッション内データがありません。")

    # === プロット ===
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    # 黒テーマ
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    # 枠線（spines）をやや暗く
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

    last_ts = s.index[-1]
    last_val = float(s.iloc[-1])
    return last_val, last_ts


def main() -> None:
    args = parse_args()

    # === 入力読み込み → JST index ===
    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, args.dt_col)

    # 値の型を % に正規化
    if args.index_key not in df.columns:
        raise ValueError(f"CSV に対象列 '{args.index_key}' がありません。列={list(df.columns)}")
    df[args.index_key] = coerce_series_as_percent(df[args.index_key], args.value_type)

    # セッション抽出
    df_sess = filter_session(df, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    # ラベル整備
    title_label = args.label if args.label else make_title_label(args.index_key)
    basis_label = args.basis

    # スナップショット + 直近値
    last_pct, last_ts = summarize_and_plot(
        df_sess,
        args.index_key,
        title_label,
        basis_label,
        args.snapshot_png,
    )

    # === 出力（テキスト / JSON） ===
    # テキスト（例）:
    # ▲ ASTRA4 日中スナップショット (2025/10/20 13:25)
    # +3.12% (基準: prev_close)
    # #ASTRA4 #日本株
    sign = "+" if last_pct >= 0 else ""
    lines = [
        f"▲ {title_label} 日中スナップショット ({last_ts.tz_convert(JST):%Y/%m/%d %H:%M})",
        f"{sign}{last_pct:.2f}% (基準: {basis_label})",
        f"#{title_label} #日本株",
    ]
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # JSON（ダッシュボードやサイト反映用）
    stats = {
        "index_key": title_label,
        "label": title_label,
        "pct_intraday": last_pct,  # ％
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
