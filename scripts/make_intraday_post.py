#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_intraday_post.py

日本株の場中スナップショット画像（黒背景）とポスト文／統計JSONを生成します。
- CSV:   Datetime 列 + 任意の指数列（例: "S-COIN+", "ASTRA4", "R-BANK9"...）
- 基準:  prev_close（前日終値）/ open@HH:MM（当日寄り付き値）
- TZ:    CSVのDatetimeがnaive→UTC想定でtz_localize("UTC")後、JSTへtz_convert
         既にtzあり→そのままJSTへtz_convert
- セッション: HH:MM（JST）でフィルタ（例: 09:00–15:30）
- 出力:
    - スナップショットPNG
    - ポスト本文 .txt
    - stats JSON（騰落率、基準ラベル、セッション、更新時刻）

使い方（例）
python scripts/make_intraday_post.py \
  --index-key scoin_plus \
  --csv docs/outputs/scoin_plus_intraday.csv \
  --out-json docs/outputs/scoin_plus_stats.json \
  --out-text docs/outputs/scoin_plus_post_intraday.txt \
  --snapshot-png docs/outputs/scoin_plus_intraday.png \
  --session-start 09:00 --session-end 15:30 \
  --day-anchor 09:00 \
  --basis prev_close \
  --label S-COIN+
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

JST = pd.Timestamp.now(tz="Asia/Tokyo").tz

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate intraday snapshot + post text")
    p.add_argument("--index-key", required=True, help="識別用キー（列名推定にも使用）例: scoin_plus / astra4 / rbank9")
    p.add_argument("--csv", required=True, help="入力CSV（Datetime列 + 指数列）")
    p.add_argument("--out-json", required=True, help="出力 stats JSON")
    p.add_argument("--out-text", required=True, help="出力 ポスト本文 .txt")
    p.add_argument("--snapshot-png", required=True, help="出力 スナップショットPNG")

    p.add_argument("--session-start", required=True, help="JST セッション開始 HH:MM（例 09:00）")
    p.add_argument("--session-end", required=True, help="JST セッション終了 HH:MM（例 15:30）")
    p.add_argument("--day-anchor", required=True, help="見出し用の“日中”基準時間（ラベル用）HH:MM")

    p.add_argument("--basis", required=True,
                   help="基準ラベル（prev_close / open@HH:MM）。例: prev_close, open@09:00")
    p.add_argument("--label", default=None,
                   help="グラフ凡例・タイトル用ラベル（例: S-COIN+）。省略時は index-key から推定")
    return p.parse_args()

# ---------- Data / Time helpers ----------

def to_jst_index(df: pd.DataFrame, dt_col: str = "Datetime") -> pd.DataFrame:
    """Datetime列をJSTのDatetimeIndexに変換して返す。"""
    if dt_col not in df.columns:
        raise ValueError(f"CSVに '{dt_col}' 列が見つかりません。")

    # まずはto_datetime
    dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")

    # NaTが多い＝タイムゾーン情報付き/naive混在の可能性 → 再判定
    if dt.isna().any():
        # 一旦tzなしで解釈
        dt2 = pd.to_datetime(df[dt_col], errors="coerce")
        # tzなし→UTCと見なす（よくある: 夜間集計がUTCベース）
        mask_naive = dt2.notna() & (dt2.dt.tz is None)
        if mask_naive.any():
            dt.loc[mask_naive] = dt2.loc[mask_naive].dt.tz_localize("UTC")
        # tz付きはdtが既に解釈している想定

    # すべてJSTへ
    dt = dt.dt.tz_convert("Asia/Tokyo")
    out = df.copy()
    out.index = dt
    return out.drop(columns=[dt_col])

def pick_value_column(df: pd.DataFrame, index_key: str) -> str:
    """
    指数列名を推定する。
    例: index_key=scoin_plus -> 'S-COIN+' / 'SCOIN_PLUS' / 'scoin_plus' などを優先探索
    """
    candidates = [
        index_key,
        index_key.upper(),
        index_key.replace("_", " ").upper(),
        index_key.replace("_", "-").upper(),
        index_key.replace("-", "_").upper(),
    ]

    # CSVから大文字化リスト
    cols_upper = {c.upper(): c for c in df.columns}
    # 最も素直な候補も加える
    if index_key.lower() == "scoin_plus":
        candidates = ["S-COIN+", "SCOIN_PLUS", "S COIN+", "SCOIN+", *candidates]
    if index_key.lower() == "astra4":
        candidates = ["ASTRA4", *candidates]
    if index_key.lower() == "rbank9":
        candidates = ["R-BANK9", "RBANK9", *candidates]

    for k in candidates:
        if k in cols_upper:
            return cols_upper[k]

    # 最後の保険：Datetime以外の1列を使う
    numeric_cols = [c for c in df.columns if c.lower() != "datetime"]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(f"指数列を特定できませんでした。index-key={index_key}, CSV列={list(df.columns)}")

def hhmm_to_time(hhmm: str) -> Tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)

def filter_session(df_jst: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    """JST IndexのDataFrameを、[start, end]（両端含む）でフィルタ。"""
    sh, sm = hhmm_to_time(start_hhmm)
    eh, em = hhmm_to_time(end_hhmm)
    # between_time(, inclusive="both") はPandas>=2で引数名が変更
    mask = (
        (df_jst.index.hour > sh) | ((df_jst.index.hour == sh) & (df_jst.index.minute >= sm))
    ) & (
        (df_jst.index.hour < eh) | ((df_jst.index.hour == eh) & (df_jst.index.minute <= em))
    )
    return df_jst.loc[mask]

def pick_anchor_value(df_jst: pd.DataFrame, basis: str, anchor_hhmm: str, value_col: str) -> float:
    """
    騰落率計算の基準値を返す。
    - prev_close: 前営業日終値。見つからなければ当日初値で代用（安定化）。
    - open@HH:MM: 指定時刻以降の最初の値（当日寄りの代替として使用）。
    """
    if basis == "prev_close":
        df_local = df_jst.copy()
        today = df_local.index[-1].date()
        prev = df_local[df_local.index.date < today]
        if not prev.empty:
            return float(prev[value_col].iloc[-1])
        # 前日データが無い（CSVが当日分のみ等）→当日初値を基準に
        print("[warn] 前日データが見つからないため、当日初値をprev_closeとして使用します。")
        return float(df_local[value_col].iloc[0])

    if basis.startswith("open@"):
        hhmm = basis.split("@", 1)[1] if "@" in basis else anchor_hhmm
        a_h, a_m = hhmm_to_time(hhmm)
        after = df_jst[
            (df_jst.index.hour > a_h) | ((df_jst.index.hour == a_h) & (df_jst.index.minute >= a_m))
        ]
        if not after.empty:
            return float(after[value_col].iloc[0])
        print("[warn] 指定open時間のデータが無いため、当日初値を基準にします。")
        return float(df_jst[value_col].iloc[0])

    raise ValueError(f"未知のbasis: {basis}")

def compute_change_pct(series: pd.Series, anchor: float) -> pd.Series:
    """(value / anchor - 1) * 100"""
    return (series.astype(float) / float(anchor) - 1.0) * 100.0

# ---------- Plot ----------

def plot_intraday_png(
    df_plot: pd.DataFrame,
    value_col: str,
    label: str,
    png_path: str,
    title_time: pd.Timestamp,
):
    """黒背景・シアン線・白スパイン無しで保存。"""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 線
    ax.plot(df_plot.index, df_plot[value_col], linewidth=2.2, color="#00e5e5", label=label)

    # 軸・スパイン
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#cccccc")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, color="#333333", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("Time", color="#cccccc")
    ax.set_ylabel("Change vs Prev Close (%)", color="#cccccc")

    title = f"{label} Intraday Snapshot ({title_time.strftime('%Y/%m/%d %H:%M')})"
    ax.set_title(title, color="#dddddd", fontsize=16)

    # 凡例
    leg = ax.legend(facecolor="black", edgecolor="none", labelcolor="#cccccc")
    for t in leg.get_texts():
        t.set_color("#cccccc")

    fig.tight_layout()
    fig.savefig(png_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

# ---------- Text / JSON ----------

@dataclass
class SnapshotStats:
    index_key: str
    label: str
    pct_intraday: float
    basis: str
    session_start: str
    session_end: str
    anchor: str
    updated_at: str

def build_post_text(label: str, pct: float, now_jst: pd.Timestamp, basis: str) -> str:
    arrow = "▲" if pct >= 0 else "▼"
    return (
        f"{arrow} {label} 日中スナップショット（{now_jst.strftime('%Y/%m/%d %H:%M')}）\n"
        f"{pct:+.2f}%（基準: {basis}）\n"
        f"#{label.replace('+','PLUS').replace('-','').replace(' ','_')} #日本株"
    )

# ---------- Main ----------

def main():
    args = parse_args()

    # 表示用ラベル
    label = args.label or args.index_key.upper()

    # CSV読込 → JSTIndex化
    raw = pd.read_csv(args.csv)
    df = to_jst_index(raw, dt_col="Datetime")

    # 値列の特定
    value_col = pick_value_column(df, args.index_key)

    # セッションで絞る（JST）
    df_jst = df[[value_col]].sort_index()
    df_sess = filter_session(df_jst, args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    # 基準値の決定
    anchor_val = pick_anchor_value(df_jst, args.basis, args.day_anchor, value_col)

    # 騰落率（％）
    df_sess_pct = df_sess.copy()
    df_sess_pct[value_col] = compute_change_pct(df_sess_pct[value_col], anchor_val)

    # 最新値の騰落率
    latest_pct = float(df_sess_pct[value_col].iloc[-1])

    # スナップショットPNG（黒背景）
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo")
    plot_intraday_png(df_sess_pct, value_col, label, args.snapshot_png, now_jst)

    # ポスト本文
    post_text = build_post_text(label, latest_pct, now_jst, args.basis)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(post_text.strip() + "\n")

    # stats JSON
    stats = SnapshotStats(
        index_key=label.replace("+", "_PLUS").replace("-", "_").upper(),
        label=label,
        pct_intraday=latest_pct,
        basis=args.basis,
        session_start=args.session_start,
        session_end=args.session_end,
        anchor=args.day_anchor,
        updated_at=now_jst.isoformat(),
    )
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_key": stats.index_key,
                "label": stats.label,
                "pct_intraday": stats.pct_intraday,
                "basis": stats.basis,
                "session": {
                    "start": stats.session_start,
                    "end": stats.session_end,
                    "anchor": stats.anchor,
                },
                "updated_at": stats.updated_at,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[ok] snapshot: {args.snapshot_png}")
    print(f"[ok] text    : {args.out_text}")
    print(f"[ok] json    : {args.out_json}")


if __name__ == "__main__":
    main()
