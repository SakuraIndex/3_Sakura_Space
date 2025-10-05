import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ===== 設定 =====
TICKERS = ["9348.T", "5595.T", "402A.T", "186A.T"]
BASE_DATE = "2024-01-04"
BASE_VALUE = 1000

# ===== 出力フォルダを作成 =====
os.makedirs("docs/outputs", exist_ok=True)

# ===== データ取得 =====
prices = {}
for t in TICKERS:
    df = yf.download(t, start=BASE_DATE, interval="1d", progress=False)
    if not df.empty:
        prices[t] = df["Close"]

df_all = pd.DataFrame(prices)

# ===== 等金額平均指数 =====
df_all["index"] = df_all.mean(axis=1)
base = df_all.loc[df_all.index >= BASE_DATE, "index"].iloc[0]
df_all["astra4"] = df_all["index"] / base * BASE_VALUE

# ===== 最新値 =====
latest_val = df_all["astra4"].iloc[-1]
latest_date = df_all.index[-1].strftime("%Y-%m-%d")
print(f"Astra4 index on {latest_date}: {latest_val:.2f}")

# ===== グラフ保存 =====
plt.figure(figsize=(8, 4))
plt.plot(df_all.index, df_all["astra4"], label="Astra4 Index", linewidth=2)
plt.title("Astra4 Index")
plt.legend()
plt.grid(True)
plt.savefig("docs/outputs/ASTRA-4_chart.png", bbox_inches="tight")

# ===== CSV出力 =====
df_all.to_csv("docs/outputs/ASTRA-4_history.csv")

# ===== テキスト出力 =====
with open("docs/outputs/ASTRA-4_post.txt", "w") as f:
    f.write(f"Astra4 index on {latest_date}: {latest_val:.2f}\n")
