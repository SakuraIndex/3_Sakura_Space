import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# === 出力先フォルダを明示的に作成 ===
OUTPUT_DIR = "docs/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 取得する銘柄 ===
TICKERS = ["^N225", "^DJI", "^IXIC", "EURJPY=X"]
BASE_DATE = datetime.now() - timedelta(days=10)  # 最近10日分

print("[INFO] Downloading data...")
prices = yf.download(TICKERS, start=BASE_DATE.strftime("%Y-%m-%d"), interval="1d")

if prices.empty:
    print("[WARN] No data retrieved from yfinance.")
    with open(os.path.join(OUTPUT_DIR, "astra4_latest.txt"), "w") as f:
        f.write("No data retrieved from yfinance.")
    exit(0)

print("[INFO] Data shape:", prices.shape)
df = prices["Adj Close"] if "Adj Close" in prices.columns else prices

# === グラフ作成 ===
plt.figure(figsize=(10, 6))
for t in TICKERS:
    if t in df.columns:
        plt.plot(df.index, df[t], label=t)
plt.title("Market Snapshot")
plt.legend()
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, "astra4_chart.png")
plt.savefig(chart_path)
plt.close()
print(f"[INFO] Saved chart -> {chart_path}")

# === CSV 保存 ===
csv_path = os.path.join(OUTPUT_DIR, "astra4_history.csv")
df.to_csv(csv_path)
print(f"[INFO] Saved history -> {csv_path}")

# === 最新値 TXT ===
latest_path = os.path.join(OUTPUT_DIR, "astra4_latest.txt")
latest = df.iloc[-1]
with open(latest_path, "w") as f:
    for t in TICKERS:
        if t in latest:
            f.write(f"{t}: {latest[t]:.2f}\n")
print(f"[INFO] Saved latest -> {latest_path}")
