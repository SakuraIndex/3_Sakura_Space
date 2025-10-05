# -*- coding: utf-8 -*-
# Astra-4 Intraday: 前日終値比(%)の平均を5分足で可視化（SNS向け黒背景）
# 陽線=青緑(#00e5d0) / 陰線=赤(#ff3b3b)

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import numpy as np
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
    if d is None or d.empty or "Close" not in d:
        raise RuntimeError(f"prev close not found: {ticker}")
    prev = d["Close"].dropna()
    if len(prev) < 2:
        raise RuntimeError(f"prev close not enough history: {ticker}")
    return float(prev.iloc[-2])

# ===== Intraday データ取得（必ず 1D Series を返す） =====
def fetch_intraday_series(ticker: str, prev_close: float) -> pd.Series:
    df = yf.download(ticker, period="1d", interval="5m", auto_adjust=False, progress=False)

    if df is None or df.empty:
        print(f"[WARN] no intraday data for {ticker}")
        return pd.Series(dtype=float, name=ticker)

    # Close 列の取り出し（MultiIndex/単一列/2D( n,1 ) どれでも 1D に矯正）
    close = None
    if "Close" in df.columns:
        close = df["Close"]
    else:
        # yfinance が MultiIndex を返すケース
        cand = [c for c in df.columns if (isinstance(c, tuple) and c[0] == "Close")]
        if cand:
            close = df[cand[0]]

    if close is None:
        print(f"[WARN] Close not found for {ticker}")
        return pd.Series(dtype=float, name=ticker)

    # 2D -> 1D に矯正
    if isinstance(close, pd.DataFrame):
        close = close.squeeze("columns")  # (n,1) -> (n,)
    # numpy/リスト等でも 1D に落とし込む
    if not isinstance(close, pd.Series):
        arr = np.asarray(close).reshape(-1)  # 明示的に 1 次元へ
        close = pd.Series(arr, index=df.index)

    # index を DatetimeIndex に統一
    if not isinstance(close.index, pd.DatetimeIndex):
        try:
            close.index = pd.to_datetime(close.index)
        except Exception:
            pass

    # 前日終値比（割合）
    s = (pd.to_numeric(close, errors="coerce") / prev_close - 1.0)
    s.name = ticker
    return s.dropna()

# ===== メイン処理 =====
def main():
    series_map: Dict[str, pd.Series] = {}

    for t in TICKERS:
        print(f"[INFO] Fetching {t} ...")
        try:
            prev_close = fetch_prev_close(t)
            s = fetch_intraday_series(t, prev_close)
            if isinstance(s, pd.Series) and s.size > 0:
                series_map[t] = s
            else:
                print(f"[WARN] empty intraday series for {t}")
        except Exception as e:
            print(f"[WARN] intraday fetch failed for {t}: {e}")

    if not series_map:
        raise RuntimeError("no intraday series for any ticker.")

    # Series 同士を列方向へ連結（index は自動アライン）
    df = pd.concat(series_map.values(), axis=1)

    # 平均（バスケット）
    basket_pct = df.mean(axis=1)

    # ===== CSV 保存 =====
    out_csv = df.copy()
    out_csv["Astra4_mean"] = basket_pct
    out_csv.to_csv(CSV_PATH, encoding="utf-8-sig", index_label="日時")

    # ===== グラフ描画（陽線/陰線で色を切替） =====
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    last_change = float(basket_pct.iloc[-1] * 100.0)
    line_color = "#00e5d0" if last_change >= 0 else "#ff3b3b"

    ax.plot(basket_pct.index, basket_pct.values * 100.0, linewidth=3.0,
            color=line_color, label="Astra-4 Basket")
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
    ax.legend(facecolor="black", edgecolor="#444444", labelcolor="white")
    fig.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ===== SNS 投稿文 =====
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
