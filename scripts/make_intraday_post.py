#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from datetime import time as dtime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Argparse
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate intraday snapshot & post text from CSV."
    )
    p.add_argument("--index-key", required=True, help="Logical index key (e.g., scoin_plus, astra4)")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out-json", required=True, help="Output JSON stats path")
    p.add_argument("--out-text", required=True, help="Output post text path")
    p.add_argument("--snapshot-png", required=True, help="Output snapshot PNG path")
    p.add_argument("--session-start", required=True, help='Session start "HH:MM" JST')
    p.add_argument("--session-end", required=True, help='Session end "HH:MM" JST')
    p.add_argument("--day-anchor", required=True, help='Day anchor time for labeling "HH:MM" JST (used in title only)')
    p.add_argument(
        "--basis",
        required=True,
        help='Return basis: "prev_close" or "open@HH:MM"'
    )
    p.add_argument(
        "--label",
        default=None,
        help="Label to use on chart & post (default: UPPER(index-key))"
    )
    p.add_argument(
        "--dt-col",
        default=None,
        help="Datetime column name in CSV (optional; auto-detect if omitted)"
    )
    return p


# ---------------------------
# Helpers
# ---------------------------

def _resolve_col(cols, preferred: Optional[str] = None, fallbacks: Optional[list] = None) -> Optional[str]:
    """Resolve a column name with case-insensitive and fallback support."""
    cols_list = list(cols)

    if preferred:
        # exact
        if preferred in cols_list:
            return preferred
        # case-insensitive
        low = preferred.lower()
        for c in cols_list:
            if c.lower() == low:
                return c

    if fallbacks:
        for cand in fallbacks:
            low = cand.lower()
            for c in cols_list:
                if c.lower() == low:
                    return c
    return None


def to_jst_index(raw: pd.DataFrame, dt_col: Optional[str]) -> pd.DataFrame:
    """
    Make DatetimeIndex in JST.
    - If tz-aware: convert to Asia/Tokyo
    - If naive: assume it's already JST and localize to Asia/Tokyo
    Auto-detect datetime column when not specified.
    """
    dt_name = _resolve_col(
        raw.columns,
        preferred=dt_col,
        fallbacks=["Datetime", "timestamp", "time", "date", "datetime"]
    )
    if dt_name is None:
        # as the last resort, take the first column
        dt_name = raw.columns[0]

    ts = pd.to_datetime(raw[dt_name], errors="coerce")

    if ts.isna().all():
        raise ValueError(f"CSVの日時列を解釈できませんでした: 候補='{dt_name}', cols={list(raw.columns)}")

    # time zone handling
    if getattr(ts.dt, "tz", None) is None:
        # treat as JST already (no conversion of wall clock)
        ts = ts.dt.tz_localize("Asia/Tokyo")
    else:
        ts = ts.dt.tz_convert("Asia/Tokyo")

    df = raw.copy()
    df.index = ts
    # keep other columns only
    if dt_name in df.columns:
        df = df.drop(columns=[dt_name])
    return df


def resolve_value_col(df: pd.DataFrame, index_key: str, label: Optional[str]) -> str:
    """
    Find the value column by:
      1) exact/case-insensitive match to index_key
      2) exact/case-insensitive match to label (if provided)
      3) substring match of index_key (case-insensitive)
      4) if only one numeric column exists, use it
    """
    # 1) index_key exact/ci
    cand = _resolve_col(df.columns, preferred=index_key)
    if cand:
        return cand

    # 2) label exact/ci
    if label:
        cand = _resolve_col(df.columns, preferred=label)
        if cand:
            return cand

    # 3) substring
    lk = index_key.lower()
    for c in df.columns:
        if lk in c.lower():
            return c

    # 4) only numeric column?
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(f"CSVに '{index_key}' に対応する列が見つかりません。候補: {list(df.columns)}")


def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{2}):(\d{2})", s)
    if not m:
        raise ValueError(f"時刻形式が不正です: {s!r} （HH:MM）")
    return int(m.group(1)), int(m.group(2))


def filter_session(df: pd.DataFrame, session_start: str, session_end: str) -> pd.DataFrame:
    """
    Keep rows between session_start and session_end (inclusive).
    Index must be tz-aware (JST).
    """
    sh, sm = parse_hhmm(session_start)
    eh, em = parse_hhmm(session_end)
    t0 = dtime(sh, sm)
    t1 = dtime(eh, em)
    idx_time = df.index.time
    mask = (idx_time >= t0) & (idx_time <= t1)
    return df.loc[mask].copy()


def pick_anchor_value(series: pd.Series, anchor_hhmm: str) -> Optional[float]:
    """
    Get value at/after the anchor time on the same day.
    If exact match is missing, choose the first record at or after the time.
    """
    ah, am = parse_hhmm(anchor_hhmm)
    anchor_t = dtime(ah, am)
    # mask at/after anchor time
    m = series.index.time >= anchor_t
    if not m.any():
        return None
    return float(series.loc[m].iloc[0])


def compute_series_by_basis(series: pd.Series, basis: str, day_anchor: str) -> Tuple[pd.Series, str]:
    """
    Two modes:
      - "prev_close": values are already % vs previous close -> use as-is
      - "open@HH:MM": subtract the anchor's value (assumes series itself is % vs prev close)
    Return (series_% , y_label)
    """
    basis = basis.strip().lower()
    if basis == "prev_close":
        y_label = "Change vs Prev Close (%)"
        return series, y_label

    m = re.fullmatch(r"open@(\d{2}:\d{2})", basis)
    if m:
        anchor = pick_anchor_value(series, m.group(1))
        if anchor is None:
            # fallback: use first value of day
            anchor = float(series.iloc[0])
        y_label = "Change vs Anchor (%)"
        return series - anchor, y_label

    # unknown basis -> fallback to as-is
    y_label = "Change (%)"
    return series, y_label


# ---------------------------
# Plot
# ---------------------------

def plot_snapshot(
    ts: pd.Series,
    label: str,
    title_dt_str: str,
    y_label: str,
    out_png: str
) -> None:
    # Dark figure (no white border)
    plt.close("all")
    fig = plt.figure(figsize=(12, 6), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # hide spines (frame)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # grid & ticks
    ax.grid(False)
    ax.tick_params(colors="white", which="both")

    # line
    ax.plot(ts.index, ts.values, linewidth=2.0, color="#00e5e6", label=label)

    # labels
    ax.set_title(f"{label} Intraday Snapshot ({title_dt_str})", color="white", fontsize=14, pad=14)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel(y_label, color="white")

    # legend (dark)
    leg = ax.legend(facecolor="black", edgecolor="black", labelcolor="white")
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Post text & JSON
# ---------------------------

def build_post_text(label_jp: str, pct_now: float, basis: str, now_jst: pd.Timestamp, hashtag: str) -> str:
    ts_str = now_jst.strftime("%Y/%m/%d %H:%M")
    sign = "▲" if pct_now >= 0 else "▼"
    return (
        f"{sign} {label_jp} 日中スナップショット（{ts_str}）\n"
        f"{pct_now:+.2f}%（基準: {basis}）\n"
        f"#{hashtag} #日本株\n"
    )


def dump_stats_json(
    index_key: str,
    label: str,
    pct_now: float,
    basis: str,
    session_start: str,
    session_end: str,
    day_anchor: str,
    out_json: str,
) -> None:
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo")
    data = {
        "index_key": index_key.upper(),
        "label": label,
        "pct_intraday": float(pct_now),
        "basis": basis,
        "session": {
            "start": session_start,
            "end": session_end,
            "anchor": day_anchor,
        },
        "updated_at": now_jst.isoformat(),
    }
    pd.Series(data).to_json(out_json, force_ascii=False, indent=2)


# ---------------------------
# Main
# ---------------------------

def main():
    args = build_parser().parse_args()

    # Label
    label = args.label or args.index_key.upper()
    hashtag = label  # 例: S-COIN+ / ASTRA4 など

    # Load & index
    raw = pd.read_csv(args.csv)
    df_jst = to_jst_index(raw, dt_col=args.dt_col)

    # Resolve value column
    value_col = resolve_value_col(df_jst, index_key=args.index_key, label=label)
    series = pd.to_numeric(df_jst[value_col], errors="coerce").dropna()

    # Filter by session
    df_sess = filter_session(series.to_frame("v"), args.session_start, args.session_end)
    if df_sess.empty:
        raise ValueError("セッション内データがありません。")

    series_sess = df_sess["v"]

    # Compute by basis
    series_pct, y_label = compute_series_by_basis(series_sess, args.basis, args.day_anchor)

    # Current stats
    pct_now = float(series_pct.iloc[-1])
    now_jst = pd.Timestamp.now(tz="Asia/Tokyo")

    # Snapshot
    title_dt_str = now_jst.strftime("%Y/%m/%d %H:%M")
    plot_snapshot(series_pct, label=label, title_dt_str=title_dt_str, y_label=y_label, out_png=args.snapshot_png)

    # Post text
    post_text = build_post_text(label_jp=label, pct_now=pct_now, basis=args.basis, now_jst=now_jst, hashtag=hashtag)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(post_text)

    # JSON
    dump_stats_json(
        index_key=args.index_key,
        label=label,
        pct_now=pct_now,
        basis=args.basis,
        session_start=args.session_start,
        session_end=args.session_end,
        day_anchor=args.day_anchor,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
