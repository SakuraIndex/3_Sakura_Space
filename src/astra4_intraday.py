# -*- coding: utf-8 -*-
# Astra-4 Intraday: 前日終値比(%)の変金額平均を5分足で可視化（SNS向け黒背景）

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===== 設定 =====
TICKERS: List[str] = ["9348.T", "5595.T", "402A.T", "186A.T"]
OUT_DIR = "docs/outputs"
IMG_PATH = os.path.join(OUT_DIR, "astra4_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "astra4_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "astra4_post_intraday.txt")

os.makedirs(OUT_DIR, exist_ok=True)

def jst_now():
    return datetime.now(timezone(timedelta(hours=9)))

# ===== 前日終値を取得 =====
def fetch_prev_close(ticker: str) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if d is None or d.empty:
        raise RuntimeError(f"prev close not found: {ticker}")
    prev = d["Close"].dropna()
    return float(prev.iloc[-2])

# =====  Intraday データ取得 =====
def fetch_intraday(ticker: str, prev_close: float) -> pd.Series:
    df = yf.download(ticker, period="1d", interval="5m", auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"[WARN] no intraday data for {ticker}")
        return pd.Series(dtype=float)
    close_pct = (df["Close"] / prev_close - 1.0)
    return close_pct

# ===== メイン処理 =====
def main():
    intraday_data: Dict[str, pd.Series] = {}
    for t in TICKERS:
        print(f"[INFO] Fetching {t} ...")
        try:
            prev_close = fetch_prev_close(t)
            s = fetch_intraday(t, prev_close)
            if not s.empty:
                intraday_data[t] = s
        except Exception as e:
            print(f"[WARN] intraday fetch failed for {t}: {e}")

    if not intraday_data:
        raise RuntimeError("no intraday data for any ticker.")

    # DataFrameへ結合
    df = pd.DataFrame(intraday_data)
    basket_pct = df.mean(axis=1)

    # CSV出力
    df.to_csv(CSV_PATH, encoding="utf-8-sig", index_label="日時")

    # ===== グラフ描画 =====
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 陽線・陰線で色を動的に変更
    last_change = basket_pct.values[-1] * 100
    line_color = "#00e5d0" if last_change >= 0 else "#ff3b3b"  # 青緑 or 赤

    ax.plot(basket_pct.index, basket_pct.values * 100.0, linewidth=3.0, color=line_color)
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.set_title(
        f"Astra-4 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})",
        color="white", fontsize=22, pad=14
    )
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ===== SNS投稿文 =====
    sign = "🔺" if last_change >= 0 else "🔻"
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"{sign} アストラ4日中取引（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{last_change:+.2f}%（前日終値比）\n"
            "構成銘柄：OP研究所 / アクセルスペースHD / アストロスケールHD / 宇宙銘柄A\n"
            "#宇宙株 #Astra4 #日本株 #株式市場\n"
        )

    print("✅ intraday outputs:")
    print(" -", os.path.abspath(IMG_PATH))
    print(" -", os.path.abspath(CSV_PATH))
    print(" -", os.path.abspath(POST_PATH))


if __name__ == "__main__":
    main()
