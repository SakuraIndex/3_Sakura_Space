#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTRA4 long-term level charts (7d / 1m / 1y)
- 入力: docs/outputs/astra4_history.csv（date,value）
- 出力: docs/outputs/astra4_7d.png, _1m.png, _1y.png
- 「点が少ない」場合は単点プロット＋注記を描画
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

INDEX_KEY = os.environ.get("INDEX_KEY", "astra4")
OUT_DIR = Path(os.environ.get("OUT_DIR", "docs/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
HIST_CSV = OUT_DIR / f"{INDEX_KEY}_history.csv"

def theme(fig, ax):
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)
    fig.patch.set_facecolor("#111317")
    ax.set_facecolor("#111317")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", colors="#ffffff", labelsize=10)
    ax.yaxis.label.set_color("#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.18, color="#ffffff")
    ax.grid(True, which="minor", linestyle="-", linewidth=0.10, alpha=0.10, color="#ffffff")

def load_history() -> pd.DataFrame:
    if not HIST_CSV.exists():
        print(f"[long] history not found: {HIST_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(HIST_CSV)
    if df.shape[1] < 2:
        print(f"[long] invalid history shape: {HIST_CSV}")
        return pd.DataFrame()
    dcol, vcol = df.columns[:2]
    df = df.rename(columns={dcol: "date", vcol: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
    return df

def slice_span(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty: return df
    last = df["date"].max()
    return df[df["date"] >= (last - pd.Timedelta(days=days))].copy()

def plot_span(df: pd.DataFrame, title: str, ylabel: str, out_png: Path):
    fig, ax = plt.subplots()
    theme(fig, ax)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    if len(df) >= 2:
        ax.plot(df["date"].values, df["value"].values, linewidth=2.6)
    elif len(df) == 1:
        ax.scatter(df["date"].values, df["value"].values, s=30)
        ax.text(df["date"].iloc[0], df["value"].iloc[0], "Insufficient history (need ≥ 2 days)",
                color="#cfd3dc", fontsize=12, va="bottom", ha="left")
    else:
        ax.text(0.5, 0.5, "No data", color="#cfd3dc", fontsize=14, transform=ax.transAxes, ha="center")

    major = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_major_formatter(AutoDateFormatter(major))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(MaxNLocator(nbins=50))

    fig.tight_layout()
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[long] WROTE: {out_png}")

def main():
    df = load_history()
    if df.empty:
        print("[long] no history -> skip")
        return

    spans = [("7d", 7), ("1m", 30), ("1y", 365)]
    for tag, days in spans:
        d = slice_span(df, days)
        plot_span(d, f"{INDEX_KEY.upper()} ({tag} level)", "Level (index)", OUT_DIR / f"{INDEX_KEY}_{tag}.png")

if __name__ == "__main__":
    main()
