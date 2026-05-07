import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(BASE_DIR, exist_ok = True)

EDGE_THRESHOLD = 0.15
GRANGER_LAG = 2
GRANGER_WINDOW_1H = 72
GRANGER_WINDOW_4H = 60
GRANGER_STABILITY_WINDOW = 24
GRANGER_SPARSE_THRESHOLD = 0.0
SENTIMENT_FEATURES = 5
_LOG2 = np.log(2)
_2LOG2M1 = 2 * np.log(2) - 1

# inputs: dataframe of returns, dataframe of lows, dataframe of closes, horizon in timesteps
# outputs: future returns, sharpe, drawdown
# all outputs are winsorized to 5 MAD and aligned with the input index (i.e. shifted by -horizon)
def winsorize(df, n_mad = 5, window = 500, min_periods = 50):
    med = df.rolling(window, min_periods = min_periods).median().shift(1).ffill().bfill()
    abs_dev = (df - med).abs()
    mad = abs_dev.rolling(window, min_periods = min_periods).median().shift(1).ffill().bfill()
    sigma = (mad * 1.4826).clip(lower = 1e-6)
    lo = med - n_mad * sigma
    hi = med + n_mad * sigma
    return df.clip(lower = lo, upper = hi, axis = 0)

# inputs: DataFrame of features, window size for rolling stats, optional min_periods for rolling
# output: DataFrame of normalized features, winsorized to 5 MAD and standardized by rolling mean and std
# if min_periods is not provided, it defaults to max(20, window // 4) to ensure enough data for stable stats
def normalise(df, window, min_periods = None):
    mp = min_periods if min_periods is not None else max(20, window // 4)
    df = winsorize(df)
    mu = df.rolling(window, min_periods = mp).mean()
    sigma = df.rolling(window, min_periods = mp).std().clip(lower = 1e-4)
    return (df - mu) / sigma

#inputs: dict of feature name to DataFrame, window size for rolling stats
# output: dict of feature name to normalized DataFrame
# applies the normalise function to each DataFrame in the input dict using the specified window size
def build_features(feature_dict, window):
    return {name: normalise(df, window) for name, df in feature_dict.items()}

# inputs: list of DataFrames or Series
# output: index of rows where all input DataFrames have valid (non-inf, non-na) values
# this is used to find the common valid time steps across multiple feature DataFrames, ensuring that it only uses rows where all features are present and finite
def get_valid_index(frames):
    mask = None
    for f in frames:
        f = f if isinstance(f, pd.DataFrame) else f.to_frame()
        fm = f.replace([np.inf, -np.inf], np.nan).notna().all(axis = 1)
        mask = fm if mask is None else (mask & fm)
    return mask[mask].index

# inputs: source index (from a feature DataFrame), target index (from the returns DataFrame)
# output: boolean Series indicating which rows in the source index are present in the target index
# this is used to align feature DataFrames with the target returns DataFrame, ensuring that it only keeps rows where the features and targets are both available
def pos_mask(source_index, target_index):
    return source_index.isin(target_index)

# inputs: dict of feature name to DataFrame, index of valid rows
# output: 3D numpy array of shape (T, N, F) where T is the number of valid time steps, N is the number of assets, and F is the number of features
# this function stacks the feature DataFrames into a single 3D array, selecting only the rows corresponding to the valid index and converting the values to float32 for efficient storage and computation
def stack_node_array(feat_dict, valid_index):
    return np.stack(
        [df.loc[valid_index].values.astype(np.float32) for df in feat_dict.values()],
        axis = 2,
    )

# inputs: fine index (from a higher frequency DataFrame), coarse index (from a lower frequency DataFrame)
# output: numpy array of integers representing the mapping from fine index to coarse index
# this function computes the hierarchy indices by finding the position of each timestamp in the fine index within the coarse index,
# effectively mapping each fine time step to its corresponding coarse time step. It uses np.searchsorted for efficient lookup and ensures that the resulting indices are valid by clipping them to the range of the coarse index.
def compute_hierarchy_indices(fine_index, coarse_index):
    idx = np.searchsorted(coarse_index.asi8, fine_index.asi8, side = "right") - 1
    return np.clip(idx, 0, len(coarse_index) - 1).astype(np.int32)

# inputs: DataFrame of features, window size for rolling correlation, optional threshold for edge inclusion
# output: 3D numpy array of shape (T, N, N) where T is the number of time steps, N is the number of assets
# this function computes the rolling correlation matrix for the given DataFrame of features, using a specified window size. It iterates over the time steps starting from the point where enough data is available
# for the rolling window, computes the covariance and standard deviation to derive the correlation matrix, and applies a threshold to determine which edges to include in the resulting adjacency matrices. The output
# is a 3D array where each slice along the first dimension represents the adjacency matrix at a given time step.
def rolling_corr_matrix(df, window, threshold = EDGE_THRESHOLD):
    arr = df.to_numpy().astype(np.float64)
    T, N = arr.shape
    adj = np.zeros((T, N, N), dtype = np.float16)
    for t in range(window - 1, T):
        w = arr[t - window + 1 : t + 1]
        w_c = w - w.mean(axis = 0)
        cov = (w_c.T @ w_c) / window
        std = np.sqrt(np.maximum(np.diag(cov), 0.0)) + 1e-9
        corr = cov / np.outer(std, std)
        np.clip(corr, -1.0, 1.0, out = corr)
        corr[np.abs(corr) < threshold] = 0.0
        adj[t] = corr
    return adj

# inputs: array of timestamps in nanoseconds, frequency string (e.g. "15m", "1h")
# output: 2D numpy array of shape (T, F) where T is the number of time steps and F is the number of time encoding features
# this function computes time encoding features based on the provided timestamps and frequency. It extracts components such as day of week, day of year, hour, and minute, and applies sine and cosine transformations 
# to capture cyclical patterns in the time data. The resulting features are normalized to the range [-1, 1] and include a linear time component to capture trends over time. The specific features included depend on 
# the frequency, with finer frequencies including more granular time components.
def compute_time_encoding(times_ns, freq):
    dt = pd.DatetimeIndex(times_ns)
    parts = [
        np.sin(2 * np.pi * dt.dayofweek / 7).astype(np.float32),
        np.cos(2 * np.pi * dt.dayofweek / 7).astype(np.float32),
        np.sin(2 * np.pi * dt.dayofyear / 365.25).astype(np.float32),
        np.cos(2 * np.pi * dt.dayofyear / 365.25).astype(np.float32),
    ]
    if freq in ("15m", "30m", "1h", "4h"):
        parts += [
            np.sin(2 * np.pi * dt.hour / 24).astype(np.float32),
            np.cos(2 * np.pi * dt.hour / 24).astype(np.float32),
        ]
    if freq in ("15m", "30m"):
        parts += [
            np.sin(2 * np.pi * dt.minute / 60).astype(np.float32),
            np.cos(2 * np.pi * dt.minute / 60).astype(np.float32),
        ]
    parts.append(np.linspace(0.0, 1.0, len(times_ns), dtype = np.float32))
    return np.stack(parts, axis = 1)

# inputs: adjacency matrix of shape (T, N, N), signal array of shape (T, N)
# output: 4D numpy array of shape (T, N, N, F) where F is the number of edge features
# this function computes edge features for a graph based on the provided adjacency matrix and signal array. It calculates three types of features: the masked adjacency values, the masked 
# differences in adjacency values between consecutive time steps, and the masked differences in signal values between connected nodes. The resulting edge features are stored in a 4D array where 
# each slice along the first dimension corresponds to a time step, and each slice along the last dimension corresponds to a different edge feature. The features are computed only for the edges present 
# in the adjacency matrix, as indicated by the mask.
def compute_edge_features(adj, signal_arr):
    adj_f = adj.astype(np.float32)
    delta = np.concatenate([np.zeros_like(adj_f[:1]), adj_f[1:] - adj_f[:-1]], axis = 0)
    spread = signal_arr[:, :, np.newaxis] - signal_arr[:, np.newaxis, :]
    mask = adj_f != 0.0
    return np.stack([adj_f * mask, delta * mask, spread * mask], axis = -1).astype(np.float16)

# inputs: Series of close prices, Series of returns, Series of low prices, horizon in timesteps
# outputs: Series of future returns, Series of sharpe ratios, Series of drawdowns
# this function builds the target variables for a predictive model based on the provided price and return data. It calculates the future returns by shifting the returns Series by the specified horizon,
def build_targets(close, returns, low, horizon):
    returns = winsorize(returns)
    future_ret = returns.shift(-horizon)
    fwd_vol = returns.rolling(max(2, horizon), min_periods = 2).std().shift(-horizon)
    sharpe = winsorize(future_ret / fwd_vol.clip(lower = 1e-4))
    drawdown = winsorize((low.rolling(horizon, min_periods = 1).min().shift(-horizon) - close) / close)
    return future_ret, sharpe, drawdown

# inputs: name of the file (without extension), dict of array name to numpy array
# output: saves the arrays to a compressed .npz file and prints the shapes and data types of the saved arrays
# this function saves the provided numpy arrays to a compressed .npz file in the specified base directory. It constructs the file path using the given name, saves the arrays using np.savez_compressed, 
# and then prints out the shape and data type of each array that was saved for verification. Finally, it confirms that the file has been saved successfully.
def save_npz(name, **arrays):
    path = os.path.join(BASE_DIR, f"{name}.npz")
    np.savez_compressed(path, **arrays)
    for k, v in arrays.items():
        print(f"  {k}: {v.shape}  {v.dtype}")
    print(f"Saved {name}.npz")

# inputs: interval string (e.g. "15m", "1h")
# output: DataFrame containing the combined data for all symbols at the specified interval, with columns for open_time, asset, and other relevant features
# this function loads the data for all symbols from the parquet files in the base directory, filters the data based on the specified interval, and combines it into a single DataFrame. 
# It iterates through the files in the base directory, checks if they are parquet files corresponding to symbols, and reads them into DataFrames. It then filters each 
# DataFrame to include only rows matching the specified interval, adds an "asset" column to identify the symbol, and appends it to a list. Finally, it concatenates all the 
# DataFrames in the list, removes duplicates based on open_time and asset, sorts the combined DataFrame by open_time and asset, and resets the index before returning it.
def load_all_symbols(interval):
    dfs = []
    for f in os.listdir(BASE_DIR):
        if not f.endswith(".parquet") or "_" in f:
            continue
        symbol = os.path.splitext(f)[0]
        df = pd.read_parquet(os.path.join(BASE_DIR, f))
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df[df["interval"] == interval].copy()
        if df.empty:
            continue
        df["asset"] = symbol
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No parquets with interval={interval}. Run dp_download first.")
    return (pd.concat(dfs, ignore_index = True)
              .drop_duplicates(subset = ["open_time", "asset"])
              .sort_values(["open_time", "asset"])
              .reset_index(drop = True))

# inputs: DataFrame containing the combined data for all symbols, with columns for open_time, asset, and other relevant features
# output: dict of feature name to pivoted DataFrame, where each DataFrame has open_time as index, asset as columns, and the corresponding feature values as the data
# this function takes the combined DataFrame of all symbols and pivots it to create separate DataFrames for each feature. It iterates through a predefined list of columns, pivots the 
# DataFrame to have open_time as the index and asset as the columns, and applies forward-fill to handle missing values for price and volume features. The resulting pivoted DataFrames 
# are stored in a dictionary with the feature names as keys, which is then returned for further processing.
def build_price_matrices(data):
    cols = ["close", "open", "high", "low", "volume", "quote_asset_volume",
            "num_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
    price_cols = ("close", "open", "high", "low")
    out = {}
    for col in cols:
        m = data.pivot(index = "open_time", columns = "asset", values = col)
        if col in price_cols:
            m = m.where(m > 0)
        if col in price_cols + ("volume",):
            m = m.ffill()
        out[col] = m
    return out

# inputs: Series of high prices, Series of low prices, window size for rolling calculation
# output: Series of Parkinson volatility estimates, calculated using the high and low prices over the specified rolling window
# this function computes the Parkinson volatility, which is a measure of volatility that uses the high and low prices of an asset. It calculates the 
# logarithmic range between the high and low prices, squares it, and then takes the rolling mean over the specified window. The result is then normalized by 
# dividing by (4 * log(2)) and taking the square root to obtain the volatility estimate. The output is clipped to ensure non-negative values.
def parkinson_vol(high, low, window = 14):
    return (np.log(high / low).pow(2).rolling(window).mean() / (4 * _LOG2)).clip(lower = 0).pow(0.5)

# inputs: Series of high prices, Series of low prices, Series of close prices, Series of open prices, window size for rolling calculation
# output: Series of Garman-Klass volatility estimates, calculated using the high, low, close, and open prices over the specified rolling window
# this function computes the Garman-Klass volatility, which is an extension of the Parkinson volatility that incorporates the open and close prices. It calculates a 
# term based on the logarithmic range between the high and low prices, adjusted by a factor that accounts for the difference between the close and open prices. The resulting 
# term is then averaged over the specified rolling window, clipped to ensure non-negative values, and square-rooted to obtain the volatility estimate.
def garman_klass_vol(high, low, close, open_, window = 14):
    term = 0.5 * np.log(high / low).pow(2) - _2LOG2M1 * np.log(close / open_).pow(2)
    return term.rolling(window).mean().clip(lower = 0).pow(0.5)

# inputs: Series of high prices, Series of low prices, Series of close prices, Series of open prices, window size for rolling calculation
# output: Series of Yang-Zhang volatility estimates, calculated using the high, low, close, and open prices over the specified rolling window
# this function computes the Yang-Zhang volatility, which is a more comprehensive measure of volatility that combines the Parkinson and Garman-Klass estimates 
# while also accounting for overnight price changes. It calculates a weighted average of the logarithmic ranges and variances based on the high, low, close, and open 
# prices, using a specific weighting factor k. The resulting variance is then averaged over the specified rolling window, clipped to ensure non-negative values, and square-rooted to obtain the volatility estimate.
def yang_zhang_vol(high, low, close, open_, window = 14):
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    rs = (np.log(high / close.clip(lower = 1e-9)) * np.log(high / open_.clip(lower = 1e-9))
          + np.log(low / close.clip(lower = 1e-9)) * np.log(low / open_.clip(lower = 1e-9)))
    var = (np.log(open_.clip(lower = 1e-9) / close.shift(1).clip(lower = 1e-9)).rolling(window).var()
           + k * np.log(close.clip(lower = 1e-9) / open_.clip(lower = 1e-9)).rolling(window).var()
           + (1 - k) * rs.rolling(window).mean())
    return var.clip(lower = 0).pow(0.5)

# inputs: Series of high prices, Series of low prices, Series of close prices, window size for rolling calculation
# output: Series of Average True Range (ATR) estimates, calculated using the high, low, and close prices over the specified rolling window
# this function computes the Average True Range (ATR), which is a measure of market volatility that considers the range of price movement. It calculates the true range (TR) by 
# taking the maximum of three values: the difference between the high and low prices, the absolute difference between the high and the previous close, and the absolute difference 
# between the low and the previous close. The ATR is then computed as an exponentially weighted moving average of the true range over the specified window, providing a smoothed estimate of volatility.
def compute_atr(high, low, close, period = 14):
    pc = close.shift(1)
    tr = pd.DataFrame(np.maximum(
        (high - low).values,
        np.maximum((high - pc).abs().values, (low - pc).abs().values),
    ), index = close.index, columns = close.columns)
    return tr.ewm(alpha = 1.0 / period, adjust = False).mean()

# inputs: Series of close prices, window size for rolling calculation
# output: Series of Relative Strength Index (RSI) values, calculated using the close prices over the specified rolling window
# this function computes the Relative Strength Index (RSI), which is a momentum oscillator that measures the speed and change of price movements. It calculates the 
# difference between consecutive close prices to determine gains and losses, then computes the average gain and average loss over the specified rolling window. The 
# RSI is derived from the ratio of average gain to average loss, normalized to a range between 0 and 1 by applying a transformation that scales it to the range [0, 100] and then dividing by 100.
def compute_rsi(prices, period = 14):
    delta = prices.diff()
    gain = delta.clip(lower = 0).rolling(period).mean()
    loss = (-delta.clip(upper = 0)).rolling(period).mean()
    return (100 - (100 / (1 + gain / (loss + 1e-9)))) / 100

# inputs: Series of close prices, window size for RSI calculation, window size for stochastic calculation
# output: Series of Stochastic RSI values, calculated using the close prices over the specified rolling windows for RSI and stochastic calculation
# this function computes the Stochastic RSI, which is a momentum indicator that applies the stochastic oscillator formula to the RSI values. It first calculates the RSI 
# using the provided close prices and RSI period, then computes the rolling minimum and maximum of the RSI over the specified stochastic period. The Stochastic RSI is 
# derived by normalizing the RSI values within the range defined by the rolling minimum and maximum, resulting in a value between 0 and 1 that indicates the position of the 
# RSI relative to its recent range.
def compute_stoch_rsi(prices, rsi_period = 14, stoch_period = 14):
    rsi = compute_rsi(prices, rsi_period)
    lo = rsi.rolling(stoch_period).min()
    hi = rsi.rolling(stoch_period).max()
    return (rsi - lo) / (hi - lo + 1e-9)

# inputs: Series of high prices, Series of low prices, Series of close prices, window size for rolling calculation
# output: Series of Commodity Channel Index (CCI) values, calculated using the high, low, and close prices over the specified rolling window
# this function computes the Commodity Channel Index (CCI), which is a momentum-based oscillator that measures the deviation of the price from its average. It 
# calculates the typical price (TP) as the average of the high, low, and close prices, then computes the simple moving average (SMA) of the TP over the specified 
# rolling window. The mean absolute deviation (MAD) of the TP is also calculated over the same window. The CCI is derived by normalizing the difference between 
# the TP and its SMA by a factor of 0.015 times the MAD, resulting in a value that indicates how far the price has deviated from its average.
def compute_cci(high, low, close, period = 14):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).std()
    return (tp - sma) / (0.015 * mad + 1e-9)

# inputs: Series of high prices, Series of low prices, Series of close prices, window size for rolling calculation
# output: Series of Williams %R values, calculated using the high, low, and close prices over the specified rolling window
# this function computes the Williams %R, which is a momentum indicator that measures overbought and oversold levels. It calculates the highest high (HH) and lowest 
# low (LL) over the specified rolling window, then derives the Williams %R by normalizing the difference between the HH and the close price by the range defined by 
# HH and LL. The result is multiplied by -100 to scale it to a range between -100 and 0, where values closer to -100 indicate oversold conditions and values closer 
# to 0 indicate overbought conditions.
def compute_williams_r(high, low, close, period = 14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (hh - close) / (hh - ll + 1e-9) * -100

# inputs: Series of high prices, Series of low prices, Series of close prices, window size for rolling calculation
# output: Series of Average Directional Index (ADX) values, calculated using the high, low, and close prices over the specified rolling window
# this function computes the Average Directional Index (ADX), which is a technical indicator used to quantify the strength of a trend. It calculates the 
# positive and negative directional movements (DM+ and DM-) based on the differences between consecutive high and low prices. The true range (TR) is computed 
# using the high, low, and previous close prices. The average true range (ATR) is then calculated as an exponentially weighted moving average of the TR. The 
# directional indicators (DI+ and DI-) are derived by normalizing the DM+ and DM- by the ATR. Finally, the ADX is computed as an exponentially weighted moving 
# average of the directional movement index (DX), which measures the absolute difference between DI+ and DI- relative to their sum.
def compute_adx(high, low, close, period = 14):
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)
    up, dn = high - ph, pl - low
    dm_p = up.where((up > dn) & (up > 0), 0.0)
    dm_m = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.DataFrame(np.maximum(
        (high - low).values,
        np.maximum((high - pc).abs().values, (low - pc).abs().values),
    ), index = close.index, columns = close.columns)
    alpha = 1.0 / period
    atr_w = tr.ewm(alpha = alpha, adjust = False).mean()
    di_p = 100 * dm_p.ewm(alpha = alpha, adjust = False).mean() / (atr_w + 1e-9)
    di_m = 100 * dm_m.ewm(alpha = alpha, adjust = False).mean() / (atr_w + 1e-9)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    adx = dx.ewm(alpha = alpha, adjust = False).mean() / 100
    di_diff = (di_p - di_m) / (di_p + di_m + 1e-9)
    return adx, di_diff

# inputs: Series of high prices, Series of low prices, Series of close prices, Series of volume, window size for rolling calculation
# output: Series of Chaikin Money Flow (CMF) values, calculated using the high, low, close prices and volume over the specified rolling window
# this function computes the Chaikin Money Flow (CMF), which is a volume-weighted average of accumulation and distribution over a specified rolling window. It calculates the money flow volume (MFV) by
def compute_cmf(high, low, close, volume, period = 20):
    mfv = ((close - low) - (high - close)) / (high - low + 1e-9) * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)

# inputs: Series of close prices, Series of volume, window size for momentum calculation
# output: Series of On-Balance Volume (OBV) momentum values, calculated using the close prices and volume over the specified rolling window
# this function computes the On-Balance Volume (OBV) momentum, which is a cumulative measure of buying and selling pressure based on volume. It calculates the direction of price movement by 
# taking the sign of the logarithmic returns of the close prices, multiplies it by the volume to get the OBV, and then computes the momentum by taking the difference of the OBV over the specified period.
def compute_obv_mom(close, volume, period = 14):
    direction = np.sign(np.log(close / close.shift(1)).fillna(0))
    obv = (direction * volume).cumsum()
    return obv.diff(period)

# inputs: Series of returns, Series of volume, window size for rolling calculation
# output: Series of Amihud Illiquidity estimates, calculated using the returns and volume over the specified rolling window
# this function computes the Amihud Illiquidity measure, which is a proxy for market illiquidity. It calculates the absolute returns divided by the volume 
# (with a lower bound to avoid division by zero), and then takes the rolling mean over the specified window to obtain a smoothed estimate of illiquidity.
def amihud_illiq(returns, volume, window = 20):
    return (returns.abs() / volume.clip(lower = 1)).rolling(window).mean()

# inputs: Series of returns, window size for rolling calculation
# output: Series of rolling spread estimates, calculated using the returns over the specified rolling window
# this function computes a proxy for the bid-ask spread based on the returns. It calculates the rolling covariance of the returns with their lagged values, 
# takes the negative of this covariance, clips it to ensure non-negative values, and then takes the square root to obtain an estimate of the spread. 
# The result is multiplied by 2 to account for both sides of the spread.
def roll_spread(returns, window = 20):
    cov = returns.rolling(window).cov(returns.shift(1))
    return 2 * np.sqrt((-cov).clip(lower = 0))

# inputs: Series of returns, Series of volume, window size for rolling calculation
# output: Series of Kyle's lambda estimates, calculated using the returns and volume over the specified rolling window
# this function computes Kyle's lambda, which is a measure of market impact. It calculates the absolute returns divided by the square root of the 
# volume (with a lower bound to avoid division by zero), and then takes the rolling mean over the specified window to obtain a smoothed estimate of market impact.
def kyle_lambda(returns, volume, window = 20):
    return (returns.abs() / volume.clip(lower = 1).pow(0.5)).rolling(window).mean()

# inputs: Series of returns, window size for rolling calculation, optional parameter k for scaling
# output: Series of Hurst exponent proxy values, calculated using the returns over the specified rolling window
# this function computes a proxy for the Hurst exponent, which is a measure of long-term memory in time series data. It calculates the variance of the returns 
# over the specified window (var_1) and the variance of the sum of returns over a smaller window k (var_k). The Hurst proxy is then derived by taking the logarithm 
# of the ratio of var_k to var_1, scaling it by 0.5, and normalizing by the logarithm of k. The result is clipped to the range [0, 1] to provide a bounded estimate of the Hurst exponent.
def hurst_proxy(returns, window = 60, k = 5):
    var_1 = returns.rolling(window).var().clip(lower = 1e-9)
    var_k = returns.rolling(k).sum().rolling(window).var().clip(lower = 1e-9)
    return (0.5 * np.log(var_k / var_1) / np.log(k)).clip(0, 1)

# inputs: Series of high prices, Series of low prices, Series of close prices
# output: dict of Ichimoku feature name to Series, calculated using the high, low, and close prices
# this function computes features based on the Ichimoku Kinko Hyo indicator, which is a technical analysis tool that 
# provides insights into support and resistance levels, trend direction, and momentum. It calculates the Tenkan-sen (conversion line) 
# and Kijun-sen (base line) using rolling maximum and minimum of the high and low prices over specified periods. The Senkou Span A and B are derived 
# from the Tenkan-sen and Kijun-sen, as well as the high and low prices over a longer period. The cloud top and bottom are determined by taking the 
# maximum and minimum of the Senkou Spans. Finally, the function returns a dictionary containing the deviations of the close price from the Tenkan-sen 
# and Kijun-sen, as well as the position of the close price relative to the cloud.
def ichimoku_features(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.DataFrame(np.maximum(senkou_a.values, senkou_b.values),
                             index = close.index, columns = close.columns)
    cloud_bot = pd.DataFrame(np.minimum(senkou_a.values, senkou_b.values),
                             index = close.index, columns = close.columns)
    return {
        "tenkan_dev": (close - tenkan) / (tenkan + 1e-9),
        "kijun_dev": (close - kijun) / (kijun + 1e-9),
        "cloud_pos": (close - cloud_bot) / (cloud_top - cloud_bot + 1e-9),
    }

# inputs: 3D numpy array of shape (T, N, N) representing the adjacency matrices over time
# outputs: 1D numpy array of shape (T,) containing the frustration values, 2D numpy array of shape (T, N) containing the Fiedler vectors, and 1D numpy array of shape (T,) containing the signed spectral gap values
# this function computes the Signed Laplacian Frustration Index (SLFI) for a series of adjacency matrices. For each time step, it constructs the signed Laplacian matrix based on the positive 
# and negative edges, computes its eigenvalues and eigenvectors, and extracts the frustration (the smallest eigenvalue), the Fiedler vector (the eigenvector corresponding to the second smallest
#  absolute eigenvalue), and the signed spectral gap (the difference between the second smallest and smallest absolute eigenvalues). The results are stored in separate arrays for frustration, Fiedler 
#  vectors, and signed spectral gaps, which are returned at the end.
def compute_slfi(adj):
    T, N = adj.shape[0], adj.shape[1]
    frustration = np.full(T, np.nan, dtype = np.float32)
    fiedler = np.full((T, N), np.nan, dtype = np.float32)
    signed_gap = np.full(T, np.nan, dtype = np.float32)
    for t in range(T):
        C = adj[t].astype(np.float64)
        if np.all(C == 0):
            continue
        C_pos = np.maximum(C, 0.0)
        C_neg = np.maximum(-C, 0.0)
        np.fill_diagonal(C_pos, 0.0)
        np.fill_diagonal(C_neg, 0.0)
        L = np.diag(C_pos.sum(1)) - np.diag(C_neg.sum(1)) - C_pos + C_neg
        evals, evecs = np.linalg.eigh(L)
        frustration[t] = float(evals[0])
        abs_ord = np.argsort(np.abs(evals))
        fiedler[t] = evecs[:, abs_ord[1]] if N > 1 else evecs[:, abs_ord[0]]
        signed_gap[t] = float(np.abs(evals[abs_ord[1]]) - np.abs(evals[abs_ord[0]]))
    return frustration, fiedler, signed_gap

# inputs: 3D numpy array of shape (T, N, N) representing the adjacency matrices over time
# outputs: 1D numpy array of shape (T,) containing the persistence entropy values, 1D numpy array of shape (T,) containing the derivative of persistence entropy, and 2D numpy array of shape (T, N) containing the component lifetimes
# this function computes the persistence entropy and related features for a series of adjacency matrices. For each time step, it constructs a filtration based on the edge weights, 
# performs a union-find algorithm to track the merging of components, and calculates the persistence of each component. The persistence entropy is derived from the distribution of persistences, 
# while the component lifetimes are determined by the first merge time for each node. The results are stored in separate arrays for persistence entropy, its derivative, and component lifetimes, 
# which are returned at the end.
def compute_pef(adj):
    T, N = adj.shape[0], adj.shape[1]
    pe = np.full(T, np.nan, dtype = np.float32)
    comp_lifetime = np.full((T, N), np.nan, dtype = np.float32)
    iu = np.triu_indices(N, k = 1)
    for t in range(T):
        C = adj[t].astype(np.float64)
        edges = sorted(
            [(C[i, j], int(i), int(j)) for i, j in zip(*iu) if C[i, j] > 0],
            reverse = True,
        )
        if not edges:
            continue
        parent = list(range(N))
        rank = [0] * N
        birth = np.ones(N, dtype = np.float64)
        first_merge = np.zeros(N, dtype = np.float64)
        persistences = []

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for thresh, i, j in edges:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            dying = rj if birth[ri] <= birth[rj] else ri
            surviving = ri if dying == rj else rj
            p = float(birth[dying] - thresh)
            if p > 1e-9:
                persistences.append(p)
            if first_merge[dying] == 0:
                first_merge[dying] = thresh
            if first_merge[surviving] == 0:
                first_merge[surviving] = thresh
            if rank[surviving] < rank[dying]:
                surviving, dying = dying, surviving
            parent[dying] = surviving
            if rank[surviving] == rank[dying]:
                rank[surviving] += 1

        if persistences:
            p_arr = np.array(persistences, dtype = np.float64)
            probs = p_arr / (p_arr.sum() + 1e-9)
            pe[t] = float(-np.sum(probs * np.log(probs + 1e-12)))

        for n in range(N):
            comp_lifetime[t, n] = float(1.0 - first_merge[n]) if first_merge[n] > 0 else 1.0

    valid = ~np.isnan(pe)
    pe_deriv = np.full(T, np.nan, dtype = np.float32)
    pe_deriv[1:] = np.where(valid[1:] & valid[:-1], pe[1:] - pe[:-1], np.nan)
    return pe, pe_deriv, comp_lifetime

# inputs: 3D numpy array of shape (T, N, N) representing the adjacency matrices over time
# outputs: 1D numpy array of shape (T,) containing the maximum partial correlation values, 1D numpy array of shape (T,) containing the angles between leading eigenvectors, 1D numpy array of shape (T,) containing the spectral gap values, and 2D numpy array of shape (T, N) containing the hub scores
# this function computes the Conditional Mutual Transfer Entropy (CMT) features for a series of adjacency matrices. For each time step, it constructs a window of past adjacency matrices,
def compute_cmt(arr, window):
    T, N = arr.shape
    rho_A = np.full(T, np.nan, dtype = np.float32)
    theta = np.full(T, np.nan, dtype = np.float32)
    spec_gap = np.full(T, np.nan, dtype = np.float32)
    hub = np.full((T, N), np.nan, dtype = np.float32)
    prev_v = None

    for t in range(window, T):
        w = arr[t - window : t]
        x, y = w[:-1], w[1:]
        Wm = len(x)
        xc = x - x.mean(0)
        yc = y - y.mean(0)
        std_x = np.sqrt((xc ** 2).mean(0)).clip(min = 1e-9)
        std_y = np.sqrt((yc ** 2).mean(0)).clip(min = 1e-9)
        rho_xy = (xc.T @ yc) / (Wm * std_x[:, None] * std_y[None, :])
        rho_xx = (xc.T @ xc) / (Wm * std_x[:, None] * std_x[None, :])
        ac = np.diag(rho_xy)
        denom = np.sqrt(np.clip(
            (1 - rho_xx ** 2) * (1 - ac[None, :] ** 2), 1e-9, None))
        partial = np.clip(
            (rho_xy - rho_xx * ac[None, :]) / denom, -1 + 1e-7, 1 - 1e-7)
        TE = -0.5 * np.log1p(-partial ** 2)
        np.fill_diagonal(TE, 0.0)
        S = (TE + TE.T) * 0.5
        A = (TE - TE.T) * 0.5
        rho_A[t] = np.abs(np.linalg.eigvalsh(1j * A)).max()
        evals_S, evecs_S = np.linalg.eigh(S)
        v = evecs_S[:, -1]
        hub[t] = np.abs(v)
        spec_gap[t] = evals_S[-1] - evals_S[-2] if N > 1 else 0.0
        if prev_v is not None:
            theta[t] = np.arccos(np.clip(float(np.abs(v @ prev_v)), 0.0, 1.0))
        prev_v = v

    return rho_A, theta, spec_gap, hub

# inputs: 2D numpy array of shape (T, N) representing the time series data, window size for rolling calculation, list of scales for variance aggregation
# outputs: 2D numpy array of shape (T, N) containing the optimal scale k* values, 2D numpy array of shape (T, N) containing the left slope values, 2D numpy array of shape (T, N) containing the right slope values, and 2D numpy array of shape (T, N) containing the slope change values
# this function computes the Multiscale Variance Breakpoint (MSVB) features for a time series dataset. For each time step and each variable, it aggregates the 
# variance of the data over different scales defined by the list of scales. It then fits two linear models to the log-log plot of the variance against the scales, 
# one for the left segment and one for the right segment, and finds the breakpoint that minimizes the sum of squared errors. The optimal scale k*, the slopes 
# of the left and right segments, and the change in slope at the breakpoint are stored in separate arrays, which are returned at the end.
def compute_msvb(arr, window, scales):
    T, N = arr.shape
    log_k = np.log(np.array(scales, dtype = np.float64))
    k_star = np.full((T, N), np.nan, dtype = np.float32)
    slope_left = np.full((T, N), np.nan, dtype = np.float32)
    slope_right = np.full((T, N), np.nan, dtype = np.float32)
    slope_change = np.full((T, N), np.nan, dtype = np.float32)

    for t in range(window, T):
        w = arr[t - window : t]
        for n in range(N):
            rv = w[:, n]
            log_var = np.empty(len(scales))
            for ki, k in enumerate(scales):
                n_blocks = len(rv) // k
                if n_blocks < 2:
                    log_var[ki] = np.nan
                    continue
                agg = rv[: n_blocks * k].reshape(n_blocks, k).sum(1)
                log_var[ki] = np.log(np.var(agg) + 1e-12)
            if np.any(np.isnan(log_var)):
                continue
            best_sse, best_bp = np.inf, 1
            best_ml, best_mr = 0.0, 0.0
            for bp in range(1, len(scales) - 1):
                xl, yl = log_k[: bp + 1], log_var[: bp + 1]
                xr, yr = log_k[bp :], log_var[bp :]
                ml, bl = np.polyfit(xl, yl, 1)
                mr, br = np.polyfit(xr, yr, 1)
                sse = (np.sum((yl - (ml * xl + bl)) ** 2)
                       + np.sum((yr - (mr * xr + br)) ** 2))
                if sse < best_sse:
                    best_sse, best_bp = sse, bp
                    best_ml, best_mr = ml, mr
            k_star[t, n] = scales[best_bp]
            slope_left[t, n] = best_ml * 0.5
            slope_right[t, n] = best_mr * 0.5
            slope_change[t, n] = (best_mr - best_ml) * 0.5

    return k_star, slope_left, slope_right, slope_change

# inputs: DataFrame of shape (T, N) representing the time series data, window size for rolling calculation, lag for Granger causality test
# output: 3D numpy array of shape (T, N, N) containing the pairwise Granger causality weights over time
# this function computes the pairwise Granger causality weights for a time series dataset. For each time step, it takes a rolling window of the data and performs a 
# Granger causality test for each pair of variables. The test involves fitting two linear regression models: one that includes only the past values of the effect 
# variable (restricted model) and another that includes both the past values of the effect variable and the cause variable (unrestricted model). The F-statistic is 
# calculated based on the residual sum of squares from both models, and the weight is derived by normalizing the F-statistic with a critical value. The resulting 
# weights are stored in a 3D array, which is returned at the end.
def granger_pairwise_weights(df, window = GRANGER_WINDOW_1H, lag = GRANGER_LAG):
    try:
        from scipy.stats import f as f_dist
        n_obs = window - lag
        dfd = n_obs - 2 * lag - 1
        f_crit = float(f_dist.ppf(0.95, lag, dfd)) if dfd > 0 else 3.15
    except Exception:
        f_crit = 3.15
    arr = df.values.astype(np.float64) if hasattr(df, "values") else df.astype(np.float64)
    T, N = arr.shape
    n_obs = window - lag
    dfd = n_obs - 2 * lag - 1
    weights = np.zeros((T, N, N), dtype = np.float32)
    if dfd <= 0:
        return weights
    ones = np.ones(n_obs)
    for t in range(window, T):
        w = arr[t - window : t]
        for effect in range(N):
            y = w[lag:, effect]
            eff_cols = [w[lag - 1 - k : window - 1 - k, effect] for k in range(lag)]
            Xr = np.column_stack([ones] + eff_cols)
            br = np.linalg.lstsq(Xr, y, rcond = None)[0]
            rss_r = float(np.dot(y - Xr @ br, y - Xr @ br))
            for cause in range(N):
                if cause == effect:
                    continue
                cau_cols = [w[lag - 1 - k : window - 1 - k, cause] for k in range(lag)]
                Xu = np.column_stack([Xr] + cau_cols)
                bu = np.linalg.lstsq(Xu, y, rcond = None)[0]
                rss_u = float(np.dot(y - Xu @ bu, y - Xu @ bu))
                if rss_u < 1e-12:
                    continue
                f_stat = ((rss_r - rss_u) / lag) / (rss_u / dfd)
                weights[t, cause, effect] = float(max(f_stat, 0.0) / f_crit)
    return weights

# inputs: 3D numpy array of shape (T, N, N) representing the pairwise Granger causality weights over time, window size for stability calculation
# output: 3D numpy array of shape (T, N, N) containing the edge stability values over time
# this function computes the edge stability for a series of Granger causality weight matrices. For each time step, it calculates the standard deviation of the 
# weights over a rolling window of specified size. The resulting stability values are stored in a 3D array, which is returned at the end.
def edge_stability(granger_w, window = GRANGER_STABILITY_WINDOW):
    T, N, _ = granger_w.shape
    stability = np.zeros_like(granger_w)
    for t in range(window, T):
        stability[t] = granger_w[t - window : t].std(axis = 0)
    return stability

# inputs: 3D numpy array of shape (T, N, N) representing the pairwise Granger causality weights over time
# outputs: 2D numpy array of shape (T, N) containing the in-degree values, 2D numpy array of shape (T, N) containing the out-degree values, 2D numpy array of shape (T, N) containing the leading eigenvector centrality values, 2D numpy array of shape (T, N) containing the clustering coefficient values, and 2D numpy array of shape (T, N) containing the betweenness centrality values
# this function computes various graph topology features for a series of Granger causality weight matrices. For each time step, it calculates the in-degree and 
# out-degree by summing the weights along the appropriate axes. The leading eigenvector centrality is derived from the eigenvector corresponding to the largest 
# eigenvalue of the symmetrized weight matrix. The clustering coefficient is calculated based on the number of closed triplets relative to the number of connected 
# triplets for each node, while the betweenness centrality is computed by counting the number of shortest paths that pass through each node. The resulting features 
# are stored in separate arrays, which are returned at the end.
def graph_topology_features(granger_w):
    T, N, _ = granger_w.shape
    in_deg = granger_w.sum(axis = 1).astype(np.float32)
    out_deg = granger_w.sum(axis = 2).astype(np.float32)
    eigvec = np.zeros((T, N), dtype = np.float32)
    clust = np.zeros((T, N), dtype = np.float32)
    between = np.zeros((T, N), dtype = np.float32)

    for t in range(T):
        W = granger_w[t].astype(np.float64)
        if not np.any(W > 0):
            continue

        S = (W + W.T) * 0.5
        try:
            eigvec[t] = np.abs(np.linalg.eigh(S)[1][:, -1])
        except np.linalg.LinAlgError:
            pass

        W2diag = np.diag(W @ W)
        for i in range(N):
            denom = float(out_deg[t, i]) * float(in_deg[t, i]) - W[i, i]
            if denom > 1e-9:
                clust[t, i] = float(W2diag[i] / denom)

        active = (W > 0) & ~np.eye(N, dtype = bool)
        dist = np.where(active, -np.log(W + 1e-3), np.inf)
        np.fill_diagonal(dist, 0.0)
        for k in range(N):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])

        sv = dist[:, :, np.newaxis]
        vt = dist[np.newaxis, :, :]
        st = dist[:, np.newaxis, :]
        on_path = np.isfinite(st) & (np.abs(sv + vt - st) < 1e-9)
        for i in range(N):
            on_path[i, i, :] = False
            on_path[:, i, i] = False
            on_path[i, :, i] = False
        between[t] = on_path.sum(axis = (0, 2)).astype(np.float32)

    return in_deg, out_deg, eigvec, clust, between

# inputs: 3D numpy array of shape (T, N, N) representing the pairwise Granger causality weights over time
# outputs: 1D numpy array of shape (T,) containing the density values and 1D numpy array of shape (T,) containing the entropy values
# this function computes graph-level features for a series of Granger causality weight matrices. For each time step, it calculates the density of active edges 
# (weights greater than zero) and the entropy of the distribution of active edge weights. The density is computed as the mean of the active edge weights, while the entropy 
# is calculated using the probabilities derived from the active edge weights. The resulting density and entropy values are stored in separate arrays, which are returned at the end.
def graph_level_features(granger_w):
    T, N, _ = granger_w.shape
    density = np.zeros(T, dtype = np.float32)
    entropy = np.zeros(T, dtype = np.float32)
    off_diag = ~np.eye(N, dtype = bool)
    for t in range(T):
        edges = granger_w[t][off_diag]
        active = edges[edges > 0]
        if len(active) == 0:
            continue
        density[t] = float(active.mean())
        p = active / (active.sum() + 1e-9)
        entropy[t] = float(-np.sum(p * np.log(p + 1e-12)))
    return density, entropy

# inputs: dict of price data with keys 'close', 'high', and 'low', where each value is a DataFrame of shape (T, N) representing the respective price data for multiple assets over time
# output: Series of shape (T,) containing the regime labels, where 0 indicates bullish, 1 indicates bearish, and 2 indicates sideways
# this function computes regime labels based on the price data of a selected asset (preferably BTC). It calculates the 200-period exponential moving average (EMA) and its slope, 
# as well as the Average Directional Index (ADX) to determine the strength of the trend. The regime is classified as bullish if the ADX indicates a strong trend and the EMA slope is 
# positive, bearish if the ADX indicates a strong trend and the EMA slope is non-positive, and sideways if the ADX indicates a weak trend. The resulting regime labels are forward-filled 
# to handle any missing values and returned as a Series.
def compute_regime_labels(px_1h):
    btc = next((c for c in px_1h["close"].columns if "BTC" in c), px_1h["close"].columns[0])
    c = px_1h["close"][btc]
    h_s = px_1h["high"][btc]
    l_s = px_1h["low"][btc]
    ema200 = c.ewm(span = 200, adjust = False).mean()
    ema_slope = ema200.diff()
    adx, _ = compute_adx(h_s.to_frame(), l_s.to_frame(), c.to_frame(), 14)
    adx = adx.iloc[:, 0]
    trending = adx >= 0.20
    regime = pd.Series(np.nan, index = c.index, dtype = float)
    regime[trending & (ema_slope > 0)] = 0.0
    regime[trending & (ema_slope <= 0)] = 1.0
    regime[~trending] = 2.0
    return regime.ffill().fillna(0.0).astype(np.int8)

# inputs: number of time steps (T) and number of assets (N)
# outputs: 3D numpy array of shape (T, N, F) containing the sentiment scores and 1D numpy array of shape (T,) containing the missing data indicators
# this function generates placeholder sentiment features for a given number of time steps and assets. It creates a 3D array filled with zeros to represent the sentiment scores,
def sentiment_placeholder(n_timesteps, n_assets):
    scores = np.zeros((n_timesteps, n_assets, SENTIMENT_FEATURES), dtype = np.float32)
    missing = np.ones(n_timesteps, dtype = np.float32)
    return scores, missing

# inputs: Series of returns, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling kurtosis values, calculated using the returns over the specified rolling window
# this function computes the realized kurtosis of a series of returns over a rolling window. It calculates the second and fourth moments of the returns, and 
# then derives the kurtosis by taking the ratio of the fourth moment to the square of the second moment. The result is clipped to a reasonable range to avoid 
# extreme values and filled with a default value of 3.0 for any missing data.
def compute_realized_kurtosis(r, window, min_periods = None):
    mp = min_periods or max(window // 4, 8)
    r2 = r.pow(2)
    r4 = r.pow(4)
    m2 = r2.rolling(window, min_periods = mp).mean().clip(lower = 1e-12)
    m4 = r4.rolling(window, min_periods = mp).mean()
    return (m4 / m2.pow(2)).clip(0, 50).fillna(3.0)

# inputs: Series of returns, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling semivariance values, calculated separately for negative and positive returns over the specified rolling window
# these functions compute the downside and upside semivariance of a series of returns over a rolling window. The downside semivariance 
# focuses on negative returns by clipping the returns at an upper bound of zero, while the upside semivariance focuses on positive returns 
# by clipping at a lower bound of zero. Both functions calculate the mean of the squared clipped returns over the specified window, with an 
# optional parameter for minimum periods to ensure sufficient data for the calculation. The results are filled with zeros for any missing data.
def compute_downside_semivariance(r, window, min_periods = None):
    mp = min_periods or max(window // 4, 8)
    neg = r.clip(upper = 0)
    return neg.pow(2).rolling(window, min_periods = mp).mean().fillna(0)

# inputs: Series of returns, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling semivariance values, calculated separately for negative and positive returns over the specified rolling window
# these functions compute the downside and upside semivariance of a series of returns over a rolling window. The downside semivariance
def compute_upside_semivariance(r, window, min_periods = None):
    mp = min_periods or max(window // 4, 8)
    pos = r.clip(lower = 0)
    return pos.pow(2).rolling(window, min_periods = mp).mean().fillna(0)

# inputs: Series of returns, window size for rolling calculation, threshold for identifying jumps, optional parameter for minimum periods   
# output: Series of rolling jump variance values, calculated separately for negative and positive jumps over the specified rolling window
# this function computes the signed jump variance of a series of returns over a rolling window. It identifies jumps based on a threshold defined as a multiple of the rolling standard deviation.
def compute_signed_jump_var(r, window, threshold_std = 2.0, min_periods = None):
    mp = min_periods or max(window // 4, 8)
    roll_std = r.rolling(window, min_periods = mp).std().clip(lower = 1e-8)
    is_jump = r.abs() > threshold_std * roll_std
    pos_jump = (r * is_jump).clip(lower = 0).pow(2).rolling(window, min_periods = mp).mean()
    neg_jump = (r * is_jump).clip(upper = 0).pow(2).rolling(window, min_periods = mp).mean()
    return pos_jump.fillna(0), neg_jump.fillna(0)

# inputs: Series of close prices, short window size for momentum calculation, long window size for momentum calculation
# output: Series of price acceleration values, calculated as the difference between the short-term momentum and its lagged value over the specified windows
# this function computes the price acceleration by first calculating the short-term and long-term momentum of the close prices using logarithmic returns over specified windows.
def compute_price_acceleration(c, short_window = 4, long_window = 24):
    mom_short = np.log(c / c.shift(short_window))
    mom_long = np.log(c / c.shift(long_window))
    mom_short_prev = mom_short.shift(short_window)
    return (mom_short - mom_short_prev).fillna(0)

# inputs: Series of asset returns, Series of benchmark returns, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling relative strength values, calculated as the difference between the cumulative returns of the asset and the benchmark over the specified rolling window
# this function computes the relative strength of an asset compared to a benchmark by calculating the cumulative returns of both the asset and the benchmark over a rolling window, 
# and then taking the difference between them. The cumulative returns are calculated by summing the returns over the specified window, with an optional parameter for minimum periods 
# to ensure sufficient data for the calculation. The resulting relative strength values are filled with zeros for any missing data.
def compute_relative_strength(r, benchmark_r, window, min_periods = None):
    mp = min_periods or max(window // 4, 8)
    cum_asset = r.rolling(window, min_periods = mp).sum()
    cum_bench = benchmark_r.rolling(window, min_periods = mp).sum()
    return (cum_asset.sub(cum_bench, axis = 0)).fillna(0)

# inputs: Series of bid-ask spreads, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling net flow persistence values, calculated as the mean of the centered bid-ask spreads over the specified rolling window
# this function computes the net flow persistence by centering the bid-ask spreads around 0.5 and then calculating the rolling mean over a specified window. 
# The resulting values indicate the persistence of net flows, with positive values suggesting a bias towards buying and negative values indicating a bias towards 
# selling. The results are filled with zeros for any missing data.
def compute_net_flow_persistence(bp, window = 12, min_periods = 6):
    centered = bp - 0.5
    return centered.rolling(window, min_periods = min_periods).mean().fillna(0)

# inputs: Series of returns, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling tail ratio values, calculated as the ratio of the 95th percentile to the 5th percentile of the returns over the specified rolling window
# this function computes the tail ratio of a series of returns over a rolling window. It calculates the 95th percentile (upper) and the 5th percentile (lower) of the returns within the window,
def compute_tail_ratio(r, window = 24, min_periods = 12):
    upper = r.rolling(window, min_periods = min_periods).quantile(0.95)
    lower = r.rolling(window, min_periods = min_periods).quantile(0.05).abs().clip(lower = 1e-8)
    return (upper / lower).clip(0.1, 10).fillna(1.0)

# inputs: Series of returns, window size for rolling calculation, optional parameter for minimum periods
# output: Series of rolling maximum return values, calculated as the maximum return over the specified rolling window
# this function computes the maximum return over a rolling window by applying the rolling maximum function to the series of returns. 
# The resulting values represent the highest return observed within each window, and any missing data is filled with zeros.
def compute_max_return(r, window = 24, min_periods = 12):
    return r.rolling(window, min_periods = min_periods).max().fillna(0)

# inputs: dict of price data with keys 'close', 'high', and 'low', where each value is a DataFrame of shape (T, N) representing the respective price data for multiple assets over time, optional parameters for slope z-score scaling, ADX pivot, ADX scaling, and basket weighting method
# output: DataFrame of shape (T, 3) containing the regime scores for bullish, bearish, and neutral regimes over time
# this function computes regime scores based on the price data of multiple assets. It first calculates a weighted average price series (using either inverse volatility 
# or equal weighting) and then computes the 200-period EMA and its slope. The slope is standardized to obtain a z-score, which is then transformed using a hyperbolic 
# tangent function to derive the trend strength. The Average Directional Index (ADX) is also calculated to assess the strength of the trend. Finally, the bullish, bearish, 
# and neutral regime scores are computed based on the combination of trend strength and ADX, normalized to ensure they sum to 1, and returned as a DataFrame.
def compute_regime_scores(px_1h, slope_zscore_scale = 1.5, adx_pivot = 0.20, adx_scale = 50.0,
                          basket = "inv_vol"):
    c_all = px_1h["close"]
    h_all = px_1h["high"]
    l_all = px_1h["low"]
    log_ret = np.log(c_all / c_all.shift(1))
    if basket == "inv_vol":
        rolling_std = log_ret.rolling(256, min_periods = 64).std().clip(lower = 1e-6)
        inv_vol = (1.0 / rolling_std).fillna(0.0)
        weights = inv_vol.div(inv_vol.sum(axis = 1).clip(lower = 1e-9), axis = 0)
    elif basket == "equal":
        n = c_all.shape[1]
        weights = pd.DataFrame(1.0 / n, index = c_all.index, columns = c_all.columns)
    else:
        raise ValueError(f"unknown basket weighting: {basket}")
    basket_log_ret = (log_ret * weights).sum(axis = 1)
    basket_c = np.exp(basket_log_ret.cumsum().fillna(0.0))
    basket_h = (h_all * weights).sum(axis = 1)
    basket_l = (l_all * weights).sum(axis = 1)
    ema = basket_c.ewm(span = 256, adjust = False).mean()
    ema_slope_pct = ema.diff() / ema.shift(1).clip(lower = 1e-9)
    slope_mean = ema_slope_pct.rolling(256, min_periods = 64).mean()
    slope_std = ema_slope_pct.rolling(256, min_periods = 64).std().clip(lower = 1e-9)
    slope_z = ((ema_slope_pct - slope_mean) / slope_std).fillna(0.0)
    adx, _ = compute_adx(basket_h.to_frame("b"), basket_l.to_frame("b"), basket_c.to_frame("b"), 16)
    adx = adx.iloc[:, 0]
    trend_strength = np.tanh(slope_z * slope_zscore_scale)
    adx_strength = 1.0 / (1.0 + np.exp(-(adx.fillna(0.0) - adx_pivot) * adx_scale))
    bull = (trend_strength.clip(lower = 0.0) * adx_strength).clip(0.0, 1.0)
    bear = ((-trend_strength).clip(lower = 0.0) * adx_strength).clip(0.0, 1.0)
    neutral = (1.0 - bull - bear).clip(0.0, 1.0)
    total = (bull + bear + neutral).clip(lower = 1e-9)
    bull = (bull / total).astype(np.float32)
    bear = (bear / total).astype(np.float32)
    neutral = (neutral / total).astype(np.float32)
    return pd.DataFrame({"bull": bull, "bear": bear, "neutral": neutral}, index = c_all.index)


HORIZON_BARS_15M = {"1h": 4, "4h": 16, "16h": 64, "64h": 256}

# inputs: number of bars in the horizon, number of blocks to divide the horizon into, number of subpoints to sample within each block
# output: list of lists containing the offsets for sampling subpoints within each block, or None if the horizon cannot be divided into the specified number of blocks and subpoints
# this function calculates the offsets for sampling subpoints within blocks of a given horizon. It divides the horizon into a specified number of 
# blocks and then determines the offsets for sampling subpoints within each block based on the spacing and shift calculated from the horizon bars and 
# the number of blocks. If the horizon cannot be divided into the specified number of blocks and subpoints, it returns None.
def block_offsets(horizon_bars, n_blocks = 4, n_subpoints = 4):
    if horizon_bars == n_blocks:
        return None
    spacing = horizon_bars // n_blocks
    shift = spacing // n_blocks
    if shift < 1:
        return None
    block_offs = []
    for b in range(n_blocks):
        block_end_offset = b * shift
        sub_offs = [block_end_offset + s * spacing for s in range(n_subpoints)]
        block_offs.append(sub_offs)
    return block_offs

# inputs: DataFrame of 15-minute price data with keys 'open', 'high', 'low', 'close', and 'volume', number of bars in the horizon, number of blocks to divide the horizon into, number of subpoints to sample within each block
# output: list of dictionaries containing the aggregated block bars for each block, or None if the horizon cannot be divided into the specified number of blocks and subpoints
# this function builds block bars by aggregating the 15-minute price data over specified blocks and subpoints within a given horizon. It first calculates the 
# offsets for sampling subpoints using the block_offsets function, and then for each block, it aggregates the open, high, low, close, and volume data by averaging or 
# taking the maximum/minimum as appropriate. The resulting block bars are stored in a list of dictionaries, which is returned at the end. If the horizon cannot be 
# divided into the specified number of blocks and subpoints, it returns None.
def build_block_bars(px_15m, horizon_bars, n_blocks = 4, n_subpoints = 4):
    o = px_15m["open"]
    h = px_15m["high"]
    l = px_15m["low"]
    c = px_15m["close"]
    v = px_15m["volume"]
    offsets = block_offsets(horizon_bars, n_blocks, n_subpoints)
    if offsets is None:
        return None
    blocks = []
    for sub_offs in offsets:
        sub_o_list = [o.shift(off) for off in sub_offs]
        sub_h_list = [h.shift(off) for off in sub_offs]
        sub_l_list = [l.shift(off) for off in sub_offs]
        sub_c_list = [c.shift(off) for off in sub_offs]
        sub_v_list = [v.shift(off) for off in sub_offs]
        block_o = sum(sub_o_list) / n_subpoints
        block_h = sub_h_list[0]
        for s in sub_h_list[1:]:
            block_h = np.maximum(block_h, s)
        block_l = sub_l_list[0]
        for s in sub_l_list[1:]:
            block_l = np.minimum(block_l, s)
        block_c = sum(sub_c_list) / n_subpoints
        block_v = sum(sub_v_list) / n_subpoints
        blocks.append({
            "open": block_o, "high": block_h, "low": block_l,
            "close": block_c, "volume": block_v,
        })
    return blocks

# inputs: DataFrame of 15-minute price data with keys 'open', 'high', 'low', 'close', and 'volume', number of subpoints to sample within each block
# output: list of dictionaries containing the subpoint bars for each subpoint, where each dictionary contains the shifted open, high, low, close, and volume data for the respective subpoint
# this function builds subpoint bars by directly sampling the 15-minute price data at specified offsets for a given number of subpoints. It shifts the open, 
# high, low, close, and volume data by the respective offsets for each subpoint and stores the resulting bars in a list of dictionaries, which is returned at the end.
def build_1h_subpoint_bars(px_15m, n_subpoints = 4):
    o = px_15m["open"]
    h = px_15m["high"]
    l = px_15m["low"]
    c = px_15m["close"]
    v = px_15m["volume"]
    bars = []
    for s in range(n_subpoints):
        bars.append({
            "open": o.shift(s), "high": h.shift(s), "low": l.shift(s),
            "close": c.shift(s), "volume": v.shift(s),
        })
    return bars

# inputs: list of dictionaries containing the block features for each block, where each dictionary contains DataFrames of shape (T, N) for various features
# output: dictionary containing the aggregated mean and slope features for each original feature, where the mean is calculated as the average across blocks and the slope is calculated as the linear regression slope of the feature values across blocks
# this function aggregates block features by calculating the mean and slope for each original feature across the blocks. It first centers the block indices and 
# calculates the variance of the centered indices. Then, for each feature, it stacks the feature values from all blocks, computes the mean across blocks, and 
# calculates the slope using a linear regression approach. The resulting mean and slope features are stored in a dictionary and returned at the end.
def aggregate_blocks_mean_slope(block_features):
    n_blocks = len(block_features)
    out = {}
    feat_names = list(block_features[0].keys())
    x_centered = np.arange(n_blocks, dtype = np.float64) - (n_blocks - 1) / 2.0
    x_var = (x_centered ** 2).sum()
    for name in feat_names:
        stacked = np.stack([bf[name].values for bf in block_features], axis = 0)
        mean_arr = stacked.mean(axis = 0)
        diffs = stacked - mean_arr[None]
        slope_arr = (x_centered[:, None, None] * diffs).sum(axis = 0) / x_var
        idx = block_features[0][name].index
        cols = block_features[0][name].columns
        out[f"{name}_mean"] = pd.DataFrame(mean_arr, index = idx, columns = cols)
        out[f"{name}_slope"] = pd.DataFrame(slope_arr, index = idx, columns = cols)
    return out

# inputs: DataFrame of 15-minute price data with keys 'open', 'high', 'low', 'close', and 'volume', number of bars in the horizon
# output: dictionary containing various horizon indicators calculated from the price data, where each key corresponds to a specific indicator and the value is a DataFrame of shape (T, N) representing the indicator values for multiple assets over time
# this function computes various horizon indicators based on the 15-minute price data. It calculates the log returns, Bollinger Bands, MACD, ADX, and other technical 
# indicators using the open, high, low, close, and volume data. The resulting indicators are stored in a dictionary and returned at the end.
def compute_horizon_indicators(px_15m, lookback):
    o = px_15m["open"]
    h = px_15m["high"]
    l = px_15m["low"]
    c = px_15m["close"]
    v = px_15m["volume"]
    r = np.log(c / c.shift(1))
    r_horizon = np.log(c / c.shift(max(lookback, 1)))
    bb_mid = c.rolling(lookback).mean()
    bb_std = c.rolling(lookback).std()
    macd_fast = c.ewm(span = lookback, adjust = False).mean()
    macd_slow = c.ewm(span = lookback * 4, adjust = False).mean()
    macd_line = macd_fast - macd_slow
    macd_sig = macd_line.ewm(span = lookback, adjust = False).mean()
    adx, di_diff = compute_adx(h, l, c, lookback)
    return {
        "ret": r_horizon,
        "vol_yz": yang_zhang_vol(h, l, c, o, lookback),
        "rsi": compute_rsi(c, lookback),
        "macd_hist": (macd_line - macd_sig) / (c + 1e-9),
        "bb_pos": (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9),
        "bb_width": 4 * bb_std / (bb_mid + 1e-9),
        "adx": adx,
        "di_diff": di_diff,
        "hl_spread": (h - l) / (c + 1e-9),
        "oc_body": (c - o) / (o + 1e-9),
        "vol_zscore": v / (v.rolling(lookback * 4).mean() + 1e-9),
        "amihud": amihud_illiq(r, v, lookback),
        "hurst": hurst_proxy(r, window = lookback * 4, k = 4),
    }

# inputs: dictionary containing various horizon indicators calculated from the price data, number of bars in the horizon, number of blocks to divide the horizon into, number of subpoints to sample within each block
# output: list of dictionaries containing the sampled block features for each block, where each dictionary contains DataFrames of shape (T, N) for the respective indicators averaged across the sampled subpoints
# this function samples block features by averaging the specified indicators across the subpoints defined by the block offsets. It first calculates the offsets for sampling
def sample_at_block_offsets(indicator_dict, horizon_bars, n_blocks = 4, n_subpoints = 4):
    if horizon_bars <= n_blocks:
        sub_offsets = [[s] for s in range(n_blocks)]
    else:
        sub_offsets = block_offsets(horizon_bars, n_blocks, n_subpoints)
    block_features = []
    for sub_offs in sub_offsets:
        block_feat = {}
        for name, df in indicator_dict.items():
            sampled = [df.shift(off) for off in sub_offs]
            block_feat[name] = sum(sampled) / len(sampled)
        block_features.append(block_feat)
    return block_features

# inputs: DataFrame of 15-minute price data with keys 'open', 'high', 'low', 'close', and 'volume'
# output: DataFrame of shape (T, 3) containing the regime scores for bullish, bearish, and neutral regimes over time, where the scores are 
# derived from the 1-hour price data by forward-filling the regime scores and filling any missing values with a default score of 0 for bullish and bearish regimes and 1 for the neutral regime
def compute_regime_scores_15m(px_15m):
    c_all = px_15m["close"]
    h_all = px_15m["high"]
    l_all = px_15m["low"]
    idx_1h = c_all.index[3::4]
    c_1h = c_all.reindex(idx_1h)
    h_1h = h_all.reindex(idx_1h)
    l_1h = l_all.reindex(idx_1h)
    px_1h = {"close": c_1h, "high": h_1h, "low": l_1h}
    scores_1h = compute_regime_scores(px_1h)
    scores_15m = scores_1h.reindex(c_all.index, method = "ffill")
    scores_15m = scores_15m.fillna(pd.DataFrame({"bull": 0.0, "bear": 0.0, "neutral": 1.0},
                                                  index = c_all.index))
    return scores_15m
