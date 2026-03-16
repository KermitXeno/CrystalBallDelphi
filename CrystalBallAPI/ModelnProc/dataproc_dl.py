import requests
import pandas as pd
import time
from datetime import datetime
import os

BASE_URL = "https://api.binance.us/api/v3/klines"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(BASE_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "MATICUSDT"
]

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 1)

def download_symbol(symbol):

    start_ts = int(START_DATE.timestamp() * 1000)
    end_ts = int(END_DATE.timestamp() * 1000)

    all_rows = []

    print(f"Downloading {symbol}...")

    while start_ts < end_ts:

        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
        except requests.exceptions.RequestException:
            print("Network error, retrying...")
            time.sleep(10)
            continue

        if response.status_code != 200:
            retry_count += 1
            print("Request failed:", response.text)

            if retry_count >= max_retries:
                print("Too many failures, stopping symbol.")
                break

            time.sleep(10)
            continue
        else:
            retry_count = 0

        data = response.json()

        if len(data) < 1000:
            all_rows.extend(data)
            break

        if len(data) == 0:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]

        start_ts = last_open_time + 60000

        time.sleep(0.15)

        print(f"{symbol} rows downloaded:", len(all_rows))

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(all_rows, columns = columns)
    df = df.drop_duplicates(subset=["open_time"])

    df["open_time"] = pd.to_datetime(df["open_time"], unit = "ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit = "ms")

    numeric_cols = [
    "open","high","low","close","volume",
    "quote_asset_volume","taker_buy_base_volume",
    "taker_buy_quote_volume"]

    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    df = df.drop(columns=["ignore"])

    filename = f"{symbol}2024.csv"
    filename = os.path.join(BASE_DIR, filename)

    df.to_csv(filename, index = False)

    print(f"{symbol} complete. Rows saved:", len(df))

def main():

    for symbol in SYMBOLS:
        download_symbol(symbol)

if __name__ == "__main__":
    main()