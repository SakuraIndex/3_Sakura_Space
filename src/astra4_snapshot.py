import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ===== 設定 =====
TICKERS = ["9348.T", "5595.T", "402A.T", "186A.T"]
BASE_DATE = "2024-01-04"
BASE_VALUE = 100

# ===== データ取得 =====
prices = {}
for t in TICKERS:
    df = yf.download(t, start=BASE_DATE, interval="1d", progress=False)
    if not df.empty:
        prices[t] = df["Close"]

df_all = pd.DataFrame(prices)
df_all["index"] = df_all.mean(axis=1)
base = df_all.loc[df_all.index >= BASE_DATE, "index"].iloc[0]
df_all["astra4"] = df_all["index"] / base * BASE_VALUE

# ===== 最新値 =====
latest_val = df_all["astra4"].iloc[-1]
latest_date = df_all.index[-1].strftime("%Y-%m-%d")

# ===== 出力ディレクトリ =====
output_dir = "docs/outputs"
os.makedirs(output_dir, exist_ok=True)

# ===== グラフ出力 =====
plt.figure(figsize=(10, 5))
plt.plot(df_all.index, df_all["astra4"], label="Astra4 Index")
plt.title("Astra4 Index (Latest: {:.2f} on {})".format(latest_val, latest_date))
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "astra4_chart.png"))

# ===== CSV出力 =====
df_all.to_csv(os.path.join(output_dir, "astra4_history.csv"))

# ===== テキスト出力 =====
with open(os.path.join(output_dir, "astra4_latest.txt"), "w") as f:
    f.write(f"Latest Astra4 Index: {latest_val:.2f}\nDate: {latest_date}\n")
