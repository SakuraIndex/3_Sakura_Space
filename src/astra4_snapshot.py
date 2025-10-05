# -*- coding: utf-8 -*-
# Astra-4 snapshot: always writes outputs under docs/outputs

import os
import sys
import matplotlib
matplotlib.use("Agg")  # ランナーでも画像保存OKにする
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import yfinance as yf

# ===== 設定 =====
TICKERS = ["9348.T", "5595.T", "402A.T", "186A.T"]  # 402A/186A はyfinanceで取れない可能性あり
BASE_DATE = "2024-01-04"
BASE_VALUE = 100
OUTPUT_DIR = "docs/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg: str):
    print(msg, flush=True)

log(f"[INFO] Start Astra4 snapshot at {datetime.utcnow().isoformat()}Z")
log(f"[INFO] TICKERS={TICKERS}")
log(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")

# ===== データ取得 =====
prices = {}
available = []
for t in TICKERS:
    df = yf.download(t, start=BASE_DATE, interval="1d", progress=False)
    if df is None or df.empty:
        log(f"[WARN] No data for {t}")
        continue
    prices[t] = df["Close"]
    available.append(t)
    log(f"[OK] {t}: rows={len(df)}")

if not prices:
    # 1銘柄も取れない場合でも結果ファイルを残す（デバッグしやすく）
    note = os.path.join(OUTPUT_DIR, "astra4_latest.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write("Astra4: データが取得できませんでした（全銘柄empty）。\n")
        f.write(f"UTC: {datetime.utcnow().isoformat()}Z\n")
    log("[ERROR] No data for all tickers. Wrote note file.")
    sys.exit(0)

df_all = pd.DataFrame(prices)

# index列の平均（等金額加重）
df_all["index"] = df_all.mean(axis=1)

# 基準値の算出（基準日以降の最初の値）
valid = df_all.loc[df_all.index >= BASE_DATE, "index"].dropna()
if valid.empty:
    # 取れた銘柄はあるが、基準日以降の値が無い場合
    note = os.path.join(OUTPUT_DIR, "astra4_latest.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write("Astra4: 基準日以降の有効なデータがありませんでした。\n")
        f.write(f"使用銘柄: {available}\n")
        f.write(f"UTC: {datetime.utcnow().isoformat()}Z\n")
    log("[ERROR] No valid base value. Wrote note file.")
    sys.exit(0)

base = valid.iloc[0]
df_all["astra4"] = df_all["index"] / base * BASE_VALUE

# ===== 最新値 =====
latest_val = float(df_all["astra4"].iloc[-1])
latest_date = pd.to_datetime(df_all.index[-1]).strftime("%Y-%m-%d")
log(f"[INFO] Latest={latest_val:.2f} on {latest_date}")

# ===== グラフ出力 =====
plt.figure(figsize=(10, 5))
plt.plot(df_all.index, df_all["astra4"], label="Astra-4 Index")
plt.title(f"Astra-4 Index (Latest: {latest_val:.2f} on {latest_date})")
plt.xlabel("Date"); plt.ylabel("Index"); plt.grid(True); plt.legend(); plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "astra4_chart.png")
plt.savefig(chart_path)
plt.close()
log(f"[INFO] Saved chart -> {chart_path}")

# ===== CSV出力 =====
csv_path = os.path.join(OUTPUT_DIR, "astra4_history.csv")
df_all.to_csv(csv_path)
log(f"[INFO] Saved history -> {csv_path}")

# ===== テキスト出力 =====
txt_path = os.path.join(OUTPUT_DIR, "astra4_latest.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Latest Astra-4: {latest_val:.2f}\nDate: {latest_date}\nTickers used: {available}\nUTC: {datetime.utcnow().isoformat()}Z\n")
log(f"[INFO] Saved latest -> {txt_path}")

# 最終的に存在確認（デバッグ用）
log("[INFO] Final outputs:")
for p in [chart_path, csv_path, txt_path]:
    log(f" - exists={os.path.exists(p)}  path={p}")
