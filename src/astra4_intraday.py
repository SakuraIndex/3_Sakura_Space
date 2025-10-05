# -*- coding: utf-8 -*-
# Astra-4 Intraday (SNS向け黒背景版)
# 東証銘柄の分足データを取得し、前日終値比を可視化

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==== 設定 ====
TICKERS: List[str] = ["9348.T", "5595.T", "402A.T", "186A.T"]
OUT_DIR = "docs/outputs"
IMG_PATH = os.path.join(OUT_DIR, "astra4_intraday.png")
CSV_PATH = os.path.join(OUT_DIR, "astra4_intraday.csv")
POST_PATH = os.path.join(OUT_DIR, "astra4_post_intraday.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ==== 共通関数 ====
def jst_now() -> datetime:
    """現在時刻（JST）"""
    return datetime.now(timezone(timedelta(hours=9)))

def _pick_close(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """
    yfinanceの返り値（単列/マルチカラム）両対応でClose列を返す。
    """
    if df is None or len(df) == 0:
        return None
    # 単列DataFrame
    if "Close" in df.columns:
        s = df["Close"].dropna()
        return s if len(s) else None
    # マルチカラムDataFrame
    if isinstance(df.columns, pd.MultiIndex):
        if (ticker, "Close") in df.columns:
            s = df[(ticker, "Close")].dropna()
            return s if len(s) else None
        for col in df.columns:
            try:
                top, leaf = col
                if leaf == "Close" and str(top).split()[0].upper() == ticker.upper():
                    s = df[col].dropna()
                    return s if len(s) else None
            except Exception:
                pass
    return None

def load_intraday_close(ticker: str) -> Optional[pd.Series]:
    """
    直近分足の終値Seriesを返す。
    1日×5分足 → ダメなら5日×15分足 にフォールバック。
    """
    for period, interval in [("1d", "5m"), ("5d", "15m")]:
        try:
            df = yf.download(
                tickers=ticker, period=period, interval=interval,
                auto_adjust=False, progress=False, prepost=False, threads=False
            )
            s = _pick_close(df, ticker)
            if s is not None and len(s) > 0:
                return s
        except Exception as e:
            print(f"[WARN] intraday fetch failed for {ticker}: {e}")
    return None

def fetch_prev_close(ticker: str) -> Optional[float]:
    """
    日足終値を取得。
    """
    try:
        d = yf.download(
            tickers=ticker, period="10d", interval="1d",
            auto_adjust=False, progress=False, prepost=False, threads=False
        )
        s = _pick_close(d, ticker)
        if s is not None and len(s) > 0:
            return float(s.iloc[-1])
    except Exception as e:
        print(f"[WARN] daily fetch failed for {ticker}: {e}")
    return None


# ==== メイン処理 ====
def main():
    intraday_data: Dict[str, pd.Series] = {}
    prev_close: Dict[str, float] = {}

    for t in TICKERS:
        print(f"[INFO] Fetching {t} ...")
        intraday = load_intraday_close(t)
        prev = fetch_prev_close(t)
        if intraday is None or prev is None:
            print(f"[WARN] Skipping {t} (no data)")
            continue

        # 前日比%
        intraday = intraday / prev - 1.0
        intraday_data[t] = intraday
        prev_close[t] = prev

    if not intraday_data:
        raise RuntimeError("No intraday data for any ticker.")

    # DataFrame統合
    df = pd.DataFrame(intraday_data)
    df.to_csv(CSV_PATH)
    print(f"[OK] Saved CSV: {CSV_PATH}")

    # 平均バスケット（単純平均）
    basket_pct = df.mean(axis=1)
    title = f"Astra-4 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M')})"

    # ==== 可視化 ====
    plt.close("all")
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(
        basket_pct.index,
        basket_pct.values * 100.0,
        linewidth=3.0,
        color="#00FFAA",
        label="Astra-4 Basket"
    )

    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_title(title, color="white", fontsize=20, pad=14)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    plt.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ==== 投稿文 ====
    last_pct = basket_pct.iloc[-1] * 100.0
    sign = "＋" if last_pct >= 0 else "－"

    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"📊 Astra-4 Intraday（{jst_now().strftime('%Y/%m/%d %H:%M')}）\n"
            f"{sign}{abs(last_pct):.2f}%（前日終値比）\n"
            "構成銘柄：OP研究所 / アクセルスペースHD / アストロスケールHD / 宇宙銘柄A\n"
            "#宇宙株 #Astra4 #日本株 #株式市場\n"
        )

    print(f"[OK] Intraday outputs:")
    print(os.path.abspath(IMG_PATH))
    print(os.path.abspath(CSV_PATH))
    print(os.path.abspath(POST_PATH))


# ==== 実行 ====
if __name__ == "__main__":
    main()
