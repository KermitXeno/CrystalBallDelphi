import pandas as pd
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(BASE_DIR, exist_ok = True)

files = os.listdir(BASE_DIR)

dfs = []

for f in files:

    if not f.endswith(".csv"):
        continue

    symbol = f.replace("2024.csv", "")

    df = pd.read_csv(os.path.join(BASE_DIR, f))
    df["open_time"] = pd.to_datetime(df["open_time"])

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]

    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    df["asset"] = symbol

    dfs.append(df)

data = pd.concat(dfs)
data = data.drop_duplicates(subset = ["open_time","asset"])
data = data.sort_values(["open_time","asset"])

price_matrix = data.pivot(index = "open_time", columns = "asset", values = "close").ffill()
volume_matrix = data.pivot(index = "open_time", columns = "asset", values = "volume").ffill()
trades_matrix = data.pivot(index = "open_time", columns = "asset", values = "num_trades").ffill()
taker_buy_matrix = data.pivot(index = "open_time", columns = "asset", values = "taker_buy_base_volume").ffill()

returns = np.log(price_matrix / price_matrix.shift(1))

volatility = returns.rolling(30).std()

momentum = price_matrix.pct_change(30)

buy_pressure = taker_buy_matrix / volume_matrix

feature_dict = {"returns": returns, "volatility": volatility, "momentum": momentum, "volume": volume_matrix, "buy_pressure": buy_pressure, "num_trades": trades_matrix}
features = []

for name, df in feature_dict.items():

    df_copy = df.copy()
    df_copy.columns = pd.MultiIndex.from_product([[name], df_copy.columns])
    features.append(df_copy)

features_df = pd.concat(features, axis=1)
features_df = features_df.dropna()

horizon = 5

targets = returns.shift(-horizon)
targets = targets.loc[features_df.index]
graph_edges = returns.rolling(120).corr()

features_df.to_parquet(os.path.join(BASE_DIR,"crypto_features_2024.parquet"))

targets.to_parquet(os.path.join(BASE_DIR,"crypto_targets_2024.parquet"))

graph_edges.to_parquet(os.path.join(BASE_DIR,"crypto_graph_edges_2024.parquet"))


print("Features shape:", features_df.shape)
print("Targets shape:", targets.shape)