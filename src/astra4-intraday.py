# -*- coding: utf-8 -*-
"""
Astra-4 Intraday (1-day) Chart Generator
- 4銘柄を等金額加重で前日終値比の平均変化率を時系列化（5分足）
- 黒背景のSNS映えグラフを生成
- 出力: docs/outputs/astra4_intraday.png, astra4_intraday.csv, astra4_post_intraday.txt
"""
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ========= 設定 =========
TICKERS: List[str] = ["9348.T", "5595.T", "402A.T", "186A.T"]  # ispace, QPS研究所, アクセルスペースHD, アストロスケールHD
OUT_DIR = "docs/outputs"
IMG_PATH = os.path.join(OUT_DIR, "astra4_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "astra4_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "astra4_post_intraday.txt")

# 画像サイズ（X/note映え: 16:9）
FIG_W, FIG_H = 16, 9

# ========= ユーティリティ =========
def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def fetch_prev_close(ticker: str) -> float:
    """
    直近日の終値（当日を除く最新の有効終値）を取得
    """
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if d is None or d.empty:
        raise RuntimeError(f"prev close not found: {ticker}")
    # 当日がまだ完結していない場合があるので、直近の「確定終値」を採用
    prev = d["Close"].dropna()
    if len(prev) < 2:
        # 最低2本必要（前営業日）
        raise RuntimeError(f"insufficient daily data for: {ticker}")
    # 最後の1本が当日分の途中足でも close は出るので、直近「前営業日」を選ぶ
    return float(prev.iloc[-2])

def fetch_intraday_series(ticker: str) -> pd.Series:
    """
    当日の5分足終値シリーズを取得
    """
    # 当日分を period="1d", interval="5m" で取得
    d = yf.download(ticker, period="1d", interval="5m", auto_adjust=False, progress=False)
    if d is None or d.empty:
        raise RuntimeError(f"intraday not found: {ticker}")
    s = d["Close"].dropna()
    s.name = ticker
    return s

# ========= メインロジック =========
def main():
    ensure_dirs()

    # 前日終値（各銘柄）
    prev_close: Dict[str, float] = {}
    for t in TICKERS:
        try:
            prev_close[t] = fetch_prev_close(t)
        except Exception as e:
            print(f"[WARN] prev close fetch failed for {t}: {e}")

    # 5分足を銘柄ごとに取得し、同一時刻にアライン
    close_df = pd.DataFrame()
    for t in TICKERS:
        try:
            s = fetch_intraday_series(t)
            close_df = close_df.join(s, how="outer") if not close_df.empty else s.to_frame()
        except Exception as e:
            print(f"[WARN] intraday fetch failed for {t}: {e}")

    if close_df.empty:
        raise RuntimeError("no intraday data for any ticker.")

    # 変化率（各銘柄）= 当日価格 / 前日終値 - 1
    for t in TICKERS:
        if t in close_df.columns and t in prev_close:
            close_df[t] = close_df[t] / prev_close[t] - 1.0
        else:
            # データ欠損は列ごと除外
            if t in close_df.columns:
                close_df.drop(columns=[t], inplace=True)

    # 有効列のみで等金額平均
    if close_df.empty:
        raise RuntimeError("all tickers lacked prev close; nothing to compute.")
    basket_pct = close_df.mean(axis=1).dropna()  # 前日比（平均）

    # CSV 保存（%表示とインデックス値の両方）
    df_out = pd.DataFrame({
        "datetime": basket_pct.index.tz_localize(None) if basket_pct.index.tz is not None else basket_pct.index,
        "pct_change": (basket_pct * 100.0).round(3),
    })
    # 参考: 直近のヒストリーindex（任意。無ければ省略）
    latest_index_val = None
    hist_path = os.path.join(OUT_DIR, "astra4_history.csv")
    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            if "0" in hist.columns:
                # 既存ヒストリーが1列名「0」の場合がある想定に合わせる
                latest_index_val = float(hist.iloc[-1]["0"])
            elif "index" in hist.columns:
                latest_index_val = float(hist.iloc[-1]["index"])
        except Exception:
            pass

    if latest_index_val is not None:
        df_out["index_like"] = (latest_index_val * (1.0 + basket_pct)).round(2)

    df_out.to_csv(CSV_PATH, index=False)

    # 最終値でヘッダ（+x.xx%）を作る
    last_pct = float(basket_pct.iloc[-1]) if len(basket_pct) else 0.0
    sign = "+" if last_pct >= 0 else ""
    header = f"Astra-4 日中チャート（{jst_now().strftime('%Y/%m/%d')}）  {sign}{last_pct*100:.2f}%"

    # ====== 黒背景のSNS映えグラフ ======
    plt.close("all")
    plt.style.use("default")
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 線
    ax.plot(basket_pct.index, basket_pct.values * 100.0, linewidth=3.0, color="white")

    # 0%の基準線
    ax.axhline(0, color="#666666", linewidth=1.0)

    # 目盛・ラベルを白系
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.set_title(header, color="white", fontsize=22, pad=14)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")

    # 余白・レイアウト
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 投稿文（下書き）も保存
    post = (
        f"🚀 ASTRA-4 日中チャート {jst_now().strftime('%Y/%m/%d')}\n"
        f"{sign}{last_pct*100:.2f}%（前日終値比）\n"
        f"構成銘柄: ispace / QPS研究所 / アクセルスペースHD / アストロスケールHD（等金額加重）\n"
        f"#宇宙株 #Astra4 #日本株\n"
    )
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(post)

    print("✅ intraday outputs written:",
          os.path.abspath(IMG_PATH),
          os.path.abspath(CSV_PATH),
          os.path.abspath(POST_PATH))

if __name__ == "__main__":
    main()
