import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# === 出力先フォルダ ===
OUTPUT_DIR = "docs/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 取得する銘柄（まずは動作確認用の無難なシンボル） ===
TICKERS = ["^N225", "^DJI", "^IXIC", "EURJPY=X"]  # 後でASTRA-4に戻せます
BASE_DATE = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")

print("[INFO] Downloading data from:", BASE_DATE, "tickers:", TICKERS)

# yfinanceは複数ティッカーだと列がMultiIndexになることがあるので注意
df = yf.download(TICKERS, start=BASE_DATE, interval="1d", progress=False)
if df is None or getattr(df, "empty", True):
    print("[WARN] No data retrieved from yfinance.")
    with open(os.path.join(OUTPUT_DIR, "astra4_latest.txt"), "w") as f:
        f.write("No data retrieved from yfinance.")
    raise SystemExit(0)

# "Adj Close"があればそれを使う。なければCloseや最初のレベルを使う
if isinstance(df.columns, pd.MultiIndex):
    if ("Adj Close" in df.columns.get_level_values(0)):
        df = df["Adj Close"]
    elif ("Close" in df.columns.get_level_values(0)):
        df = df["Close"]
    else:
        # 最初のレベルの最初の要素を使う（保険）
        first_level = df.columns.levels[0][0]
        df = df[first_level]
else:
    # 単一インデックスの場合の保険
    for candidate in ["Adj Close", "Close"]:
        if candidate in df.columns:
            df = df[[candidate]].rename(columns={candidate: TICKERS[0]})
            break

# dfをTicker列が並ぶ形に正規化（単一列ならそのまま）
if isinstance(df, pd.Series):
    df = df.to_frame(name=TICKERS[0])

# 使えない列は落としておく
existing_cols = [c for c in df.columns if c in TICKERS]
if existing_cols:
    df = df[existing_cols]

print("[INFO] DataFrame shape after normalize:", df.shape)

# === グラフ ===
plt.figure(figsize=(10, 6))
for col in df.columns:
    plt.plot(df.index, df[col], label=col)
plt.title("Market Snapshot")
plt.legend()
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, "astra4_chart.png")
plt.savefig(chart_path)
plt.close()
print(f"[INFO] Saved chart -> {chart_path}")

# === CSV ===
csv_path = os.path.join(OUTPUT_DIR, "astra4_history.csv")
df.to_csv(csv_path)
print(f"[INFO] Saved history -> {csv_path}")

# === 最新値 TXT ===
latest_path = os.path.join(OUTPUT_DIR, "astra4_latest.txt")
latest = df.iloc[-1]
with open(latest_path, "w") as f:
    for col in df.columns:
        val = latest.get(col)
        if pd.notna(val):
            f.write(f"{col}: {float(val):.2f}\n")
print(f"[INFO] Saved latest -> {latest_path}")
