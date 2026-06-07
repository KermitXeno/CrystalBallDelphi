import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(BASE_DIR, exist_ok = True)

SENTIMENT_FEATURES = 5

def winsorize(df, n_mad = 5, window = 500, min_periods = 50):
    med = df.rolling(window, min_periods = min_periods).median().shift(1).ffill().bfill()
    abs_dev = (df - med).abs()
    mad = abs_dev.rolling(window, min_periods = min_periods).median().shift(1).ffill().bfill()
    sigma = (mad * 1.4826).clip(lower = 1e-6)
    lo = med - n_mad * sigma
    hi = med + n_mad * sigma
    return df.clip(lower = lo, upper = hi, axis = 0)

def normalise(df, window, min_periods = None):
    mp = min_periods if min_periods is not None else max(20, window // 4)
    df = winsorize(df)
    mu = df.rolling(window, min_periods = mp).mean()
    sigma = df.rolling(window, min_periods = mp).std().clip(lower = 1e-4)
    return (df - mu) / sigma

def get_valid_index(frames):
    mask = None
    for f in frames:
        f = f if isinstance(f, pd.DataFrame) else f.to_frame()
        fm = f.replace([np.inf, -np.inf], np.nan).notna().all(axis = 1)
        mask = fm if mask is None else (mask & fm)
    return mask[mask].index

def stack_node_array(feat_dict, valid_index):
    return np.stack(
        [df.loc[valid_index].values.astype(np.float32) for df in feat_dict.values()],
        axis = 2,
    )

def rolling_corr_matrix(df, window, threshold = 0.15):
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
    return np.stack(parts, axis = 1)

def save_npz(name, **arrays):
    path = os.path.join(BASE_DIR, f"{name}.npz")
    np.savez_compressed(path, **arrays)
    for k, v in arrays.items():
        print(f"  {k}: {v.shape}  {v.dtype}")
    print(f"Saved {name}.npz")

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

def yang_zhang_vol(high, low, close, open_, window = 14):
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    rs = (np.log(high / close.clip(lower = 1e-9)) * np.log(high / open_.clip(lower = 1e-9))
          + np.log(low / close.clip(lower = 1e-9)) * np.log(low / open_.clip(lower = 1e-9)))
    var = (np.log(open_.clip(lower = 1e-9) / close.shift(1).clip(lower = 1e-9)).rolling(window).var()
           + k * np.log(close.clip(lower = 1e-9) / open_.clip(lower = 1e-9)).rolling(window).var()
           + (1 - k) * rs.rolling(window).mean())
    return var.clip(lower = 0).pow(0.5)

def compute_rsi(prices, period = 14):
    delta = prices.diff()
    gain = delta.clip(lower = 0).rolling(period).mean()
    loss = (-delta.clip(upper = 0)).rolling(period).mean()
    return (100 - (100 / (1 + gain / (loss + 1e-9)))) / 100

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

def amihud_illiq(returns, volume, window = 20):
    return (returns.abs() / volume.clip(lower = 1)).rolling(window).mean()

def hurst_proxy(returns, window = 60, k = 5):
    var_1 = returns.rolling(window).var().clip(lower = 1e-9)
    var_k = returns.rolling(k).sum().rolling(window).var().clip(lower = 1e-9)
    return (0.5 * np.log(var_k / var_1) / np.log(k)).clip(0, 1)

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

def sentiment_placeholder(n_timesteps, n_assets):
    scores = np.zeros((n_timesteps, n_assets, SENTIMENT_FEATURES), dtype = np.float32)
    missing = np.ones(n_timesteps, dtype = np.float32)
    return scores, missing

def compute_regime_scores(px_1h, long_span = 256, short_span = 16,
                          slope_scale = 1.5, basket = "inv_vol"):
    c_all = px_1h["close"]
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
    ema_long = basket_c.ewm(span = long_span, adjust = False).mean()
    ema_short = basket_c.ewm(span = short_span, adjust = False).mean()
    def slope_z(ema, window):
        slope = ema.diff() / ema.shift(1).clip(lower = 1e-9)
        mu = slope.rolling(window, min_periods = 64).mean()
        sd = slope.rolling(window, min_periods = 64).std().clip(lower = 1e-9)
        return ((slope - mu) / sd).fillna(0.0)
    long_z = slope_z(ema_long, long_span)
    short_z = slope_z(ema_short, long_span)
    long_dir = np.tanh(long_z * slope_scale)
    short_dir = np.tanh(short_z * slope_scale)
    long_bull = long_dir.clip(lower = 0.0)
    long_bear = (-long_dir).clip(lower = 0.0)
    short_bull = short_dir.clip(lower = 0.0)
    short_bear = (-short_dir).clip(lower = 0.0)
    bull = (long_bull * short_bull).astype(np.float32)
    bear = (long_bear * short_bear).astype(np.float32)
    accumulating = (long_bear * short_bull).astype(np.float32)
    distributing = (long_bull * short_bear).astype(np.float32)
    total = (bull + bear + accumulating + distributing).clip(lower = 1e-9)
    bull = bull / total
    bear = bear / total
    accumulating = accumulating / total
    distributing = distributing / total
    return pd.DataFrame({"bull": bull, "bear": bear, "accumulating": accumulating,
                          "distributing": distributing}, index = c_all.index)


HORIZON_BARS_15M = {"1h": 4, "4h": 16, "16h": 64, "64h": 256}

HORIZON_CFG = {
    "1h": {
        "window": 4,
        "cadence": 1,
        "phase_shift": 1,
    },
    "4h": {
        "window": 16,
        "cadence": 4,
        "phase_shift": 1,
    },
    "16h": {
        "window": 64,
        "cadence": 4,
        "phase_shift": 4,
    },
    "64h": {
        "window": 256,
        "cadence": 16,
        "phase_shift": 16,
    },
}

INDICATOR_LOOKBACK = 14

KEEP_INDICATORS = {"ret", "vol_yz", "bb_pos", "bb_width", "adx", "di_diff", "hl_spread", "vol_zscore"}

def aggregate_ohlcv(px_15m, window):
    o = px_15m["open"]
    h = px_15m["high"]
    l = px_15m["low"]
    c = px_15m["close"]
    v = px_15m["volume"]
    return {
        "open": o.shift(window - 1),
        "high": h.rolling(window, min_periods = 1).max(),
        "low": l.rolling(window, min_periods = 1).min(),
        "close": c,
        "volume": v.rolling(window, min_periods = 1).sum(),
    }

def compute_hierarchical_indicators(px_15m, h_name, lookback = None, keep = None):
    if lookback is None:
        lookback = INDICATOR_LOOKBACK
    if keep is None:
        keep = KEEP_INDICATORS
    cfg = HORIZON_CFG[h_name]
    agg_px = aggregate_ohlcv(px_15m, cfg["window"])
    cadence = cfg["cadence"]
    if cadence > 1:
        idx_sub = agg_px["close"].index[::cadence]
        sub_px = {k: v.reindex(idx_sub) for k, v in agg_px.items()}
        indicators = compute_horizon_indicators(sub_px, lookback)
        full_idx = agg_px["close"].index
        indicators = {k: v.reindex(full_idx, method = "ffill") for k, v in indicators.items()}
    else:
        indicators = compute_horizon_indicators(agg_px, lookback)
    return {k: v for k, v in indicators.items() if k in keep}

def sample_blocks_hierarchical(indicators, phase_shift, n_blocks = 4):
    block_features = []
    for b in range(n_blocks):
        offset = b * phase_shift
        block_feat = {}
        for name, df in indicators.items():
            block_feat[name] = df.shift(offset)
        block_features.append(block_feat)
    return block_features


def compute_range_position(close, window):
    roll_high = close.rolling(window, min_periods = 1).max()
    roll_low = close.rolling(window, min_periods = 1).min()
    rng = roll_high - roll_low
    pos = ((close - roll_low) / (rng + 1e-9)).clip(0.0, 1.0).fillna(0.5)
    return pos


def compute_breakout(close, window):
    roll_high = close.rolling(window, min_periods = 1).max()
    roll_low = close.rolling(window, min_periods = 1).min()
    at_high = (close >= roll_high).astype(np.float32)
    at_low = (close <= roll_low).astype(np.float32)
    return (at_high - at_low).fillna(0.0)


def compute_breakout_breadth(close_df, window):
    roll_high = close_df.rolling(window, min_periods = 1).max()
    roll_low = close_df.rolling(window, min_periods = 1).min()
    at_high = (close_df >= roll_high).astype(float)
    at_low = (close_df <= roll_low).astype(float)
    return (at_high.mean(axis = 1) - at_low.mean(axis = 1)).fillna(0.0)

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
    scores_15m = scores_15m.fillna(pd.DataFrame(
        {"bull": 0.0, "bear": 0.0, "accumulating": 0.5, "distributing": 0.5},
        index = c_all.index))
    return scores_15m

def compute_atr(high, low, close, window = 14):
    if isinstance(high, pd.DataFrame):
        atr = pd.DataFrame(index = high.index, columns = high.columns, dtype = np.float64)
        for col in high.columns:
            pc = close[col].shift(1)
            tr = pd.concat([high[col] - low[col], (high[col] - pc).abs(), (low[col] - pc).abs()], axis = 1).max(axis = 1)
            atr[col] = tr.rolling(window, min_periods = 1).mean()
        return atr.astype(np.float32)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis = 1).max(axis = 1)
    return tr.rolling(window, min_periods = 1).mean()

def build_features(raw_dict, window = 120):
    out = {}
    for name, df in raw_dict.items():
        df = df.replace([np.inf, -np.inf], np.nan)
        out[name] = normalise(df.ffill().fillna(0), window = window)
    return out

def compute_stoch_rsi(close, rsi_period = 14, stoch_period = 14, smooth_k = 3):
    rsi = compute_rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch = ((rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)).rolling(smooth_k).mean()
    return stoch

def compute_cci(high, low, close, period = 14):
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw = True)
    return ((tp - sma) / (0.015 * mad + 1e-9)).clip(-3, 3) / 3.0

def compute_williams_r(high, low, close, period = 14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return ((hh - close) / (hh - ll + 1e-9))

def compute_cmf(high, low, close, volume, period = 20):
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)

def compute_obv_mom(close, volume, period = 14):
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    return obv.diff(period) / (volume.rolling(period).sum() + 1e-9)

def roll_spread(returns, window = 20):
    cov = returns.rolling(window).apply(lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0, raw = True)
    return (2 * np.sqrt((-cov).clip(lower = 0))).fillna(0)

def kyle_lambda(returns, volume, window = 20):
    abs_r = returns.abs()
    vol_mean = volume.rolling(window).mean().clip(lower = 1e-9)
    r_mean = abs_r.rolling(window).mean()
    return (r_mean / vol_mean).fillna(0)

def ichimoku_features(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    return {
        "ichi_tenkan_dev": (close - tenkan) / (close + 1e-9),
        "ichi_kijun_dev": (close - kijun) / (close + 1e-9),
        "ichi_cloud_width": (senkou_a - senkou_b) / (close + 1e-9),
        "ichi_above_cloud": ((close > senkou_a.shift(26)) & (close > senkou_b.shift(26))).astype(float) - ((close < senkou_a.shift(26)) & (close < senkou_b.shift(26))).astype(float),
    }

def compute_hierarchy_indices(sub_index, parent_index):
    sub_ns = sub_index.asi8
    parent_ns = parent_index.asi8
    idx = np.searchsorted(sub_ns, parent_ns, side = "right") - 1
    return idx.clip(0).astype(np.int32)

def build_targets(close, returns, low, horizon = 1):
    fwd_ret = returns.shift(-horizon).fillna(0)
    fwd_sharpe = fwd_ret / (returns.rolling(24, min_periods = 8).std().clip(lower = 1e-6))
    rolling_max = close.rolling(24, min_periods = 1).max()
    dd = (close - rolling_max) / (rolling_max + 1e-9)
    return fwd_ret, fwd_sharpe, dd

def compute_realized_kurtosis(returns, window = 24):
    r2 = returns.pow(2)
    r4 = returns.pow(4)
    r2_sum = r2.rolling(window, min_periods = window // 2).sum().clip(lower = 1e-12)
    r4_sum = r4.rolling(window, min_periods = window // 2).sum()
    return ((window * r4_sum / r2_sum.pow(2))).clip(0, 20).fillna(3.0)

def compute_downside_semivariance(returns, window = 24):
    neg_r = returns.clip(upper = 0).pow(2)
    return neg_r.rolling(window, min_periods = window // 2).mean().fillna(0)

def compute_upside_semivariance(returns, window = 24):
    pos_r = returns.clip(lower = 0).pow(2)
    return pos_r.rolling(window, min_periods = window // 2).mean().fillna(0)

def compute_signed_jump_var(returns, window = 24):
    r2 = returns.pow(2)
    rv = r2.rolling(window, min_periods = window // 2).sum()
    bv = (returns.abs() * returns.abs().shift(1)).rolling(window, min_periods = window // 2).sum() * (np.pi / 2)
    jump = (rv - bv).clip(lower = 0)
    pos_sign = (returns > 0).astype(float).rolling(window).mean()
    return jump * pos_sign, jump * (1 - pos_sign)

def compute_price_acceleration(close, short_window = 4, long_window = 24):
    mom_short = close.pct_change(short_window)
    mom_long = close.pct_change(long_window)
    return mom_short - mom_long / (long_window / short_window)

def compute_relative_strength(returns, benchmark_returns, window = 24):
    cum_r = returns.rolling(window, min_periods = window // 2).sum()
    cum_b = benchmark_returns.rolling(window, min_periods = window // 2).sum()
    return cum_r.sub(cum_b, axis = 0)

def compute_net_flow_persistence(buy_pressure, window = 12):
    net = buy_pressure * 2 - 1
    pos = (net > 0).astype(float).rolling(window, min_periods = window // 2).mean()
    return pos.fillna(0.5)

def compute_tail_ratio(returns, window = 24):
    q95 = returns.rolling(window, min_periods = window // 2).quantile(0.95).abs()
    q05 = returns.rolling(window, min_periods = window // 2).quantile(0.05).abs()
    return (q95 / (q05 + 1e-9)).clip(0, 10).fillna(1.0)

def compute_max_return(returns, window = 24):
    return returns.rolling(window, min_periods = window // 2).max().fillna(0)
