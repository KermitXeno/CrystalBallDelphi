import os
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
 
import pandas as pd
import requests
 
BASE_URL = "https://api.binance.us/api/v3/klines"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
 
os.makedirs(BASE_DIR, exist_ok = True)
 
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT",
]
 
INTERVALS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}
 
START_MS = int(datetime(2020, 1, 1).timestamp() * 1000)
END_MS = int(datetime(2026, 1, 1).timestamp() * 1000)
 
BATCH_LIMIT = 1000
RATE_LIMIT = 9
MAX_RETRIES = 5
 
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
 
FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
]

class RateLimiter:
    def __init__(self, calls_per_second):
        self._interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last = 0.0
 
    def wait(self):
        with self._lock:
            gap = self._interval - (time.time() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.time()
 
 
_limiter = RateLimiter(RATE_LIMIT)
 
 
def resume_ts(path, interval, step_ms):
    if not os.path.exists(path):
        return START_MS
    df = pd.read_parquet(path, columns = ["interval", "open_time"])
    sub = df[df["interval"] == interval]
    if sub.empty:
        return START_MS
    return int(sub["open_time"].max().timestamp() * 1000) + step_ms
 
 
def fetch_klines(symbol, interval, start_ts):
    retries = 0
    while retries < MAX_RETRIES:
        _limiter.wait()
        try:
            resp = requests.get(BASE_URL, params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "endTime": END_MS,
                "limit": BATCH_LIMIT,
            }, timeout = 10)
        except requests.exceptions.RequestException:
            time.sleep(10)
            retries += 1
            continue
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 60)))
            retries += 1
            continue
        if resp.status_code != 200:
            print(f"request failed {resp.status_code} {resp.text[:120]}")
            time.sleep(10)
            retries += 1
            continue
        return resp.json()
    return None
 
 
def rows_to_df(rows, interval):
    df = pd.DataFrame(rows, columns = COLUMNS).drop(columns = ["ignore"])
    df.insert(0, "interval", interval)
    df["open_time"] = pd.to_datetime(df["open_time"], unit = "ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit = "ms")
    df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    return df
 
 
def flush(path, df_new):
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df_new], ignore_index = True)
    else:
        df = df_new
    df = (df
          .drop_duplicates(subset = ["interval", "open_time"])
          .sort_values(["interval", "open_time"])
          .reset_index(drop = True))
    df.to_parquet(path, index = False)
    return len(df)
 
 
def download_interval(symbol, path, interval, step_ms):
    ts = resume_ts(path, interval, step_ms)
    if ts >= END_MS:
        print(f"{symbol} {interval} already complete")
        return
 
    pending = []
    current_year = datetime.utcfromtimestamp(ts / 1000).year
    total = 0
 
    print(f"{symbol} {interval} starting {datetime.utcfromtimestamp(ts / 1000).date()}")
 
    while ts < END_MS:
        data = fetch_klines(symbol, interval, ts)
        if not data:
            break
 
        pending.extend(data)
        batch_year = datetime.utcfromtimestamp(data[-1][0] / 1000).year
 
        if batch_year != current_year:
            total = flush(path, rows_to_df(pending, interval))
            print(f"{symbol} {interval} flushed {current_year} total {total}")
            pending = []
            current_year = batch_year
 
        if len(data) < BATCH_LIMIT:
            break
 
        ts = data[-1][0] + step_ms
 
    if pending:
        total = flush(path, rows_to_df(pending, interval))
 
    print(f"{symbol} {interval} done total {total}")
 
 
def download_symbol(symbol):
    path = os.path.join(BASE_DIR, f"{symbol}.parquet")
    for interval, step_ms in INTERVALS.items():
        download_interval(symbol, path, interval, step_ms)
 
 
def main():
    with ThreadPoolExecutor(max_workers = len(SYMBOLS)) as executor:
        task_map = {executor.submit(download_symbol, s): s for s in SYMBOLS}
        for task in as_completed(task_map):
            symbol = task_map[task]
            try:
                task.result()
            except Exception as e:
                print(f"{symbol} error {e}")
 
 
if __name__ == "__main__":
    main()