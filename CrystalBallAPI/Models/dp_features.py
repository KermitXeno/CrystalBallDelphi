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


def winsorize(df, n_std = 3):
    mu = df.expanding(min_periods = 20).mean()
    sigma = df.expanding(min_periods = 20).std().clip(lower = 1e-4)
    lo = (mu - n_std * sigma).shift(1).ffill()
    hi = (mu + n_std * sigma).shift(1).ffill()
    return df.clip(lower = lo, upper = hi, axis = 0)


def normalise(df, window, min_periods = None):
    mp = min_periods if min_periods is not None else max(20, window // 4)
    df = winsorize(df)
    mu = df.rolling(window, min_periods = mp).mean()
    sigma = df.rolling(window, min_periods = mp).std().clip(lower = 1e-4)
    return (df - mu) / sigma


def build_features(feature_dict, window):
    return {name: normalise(df, window) for name, df in feature_dict.items()}


def get_valid_index(frames):
    mask = None
    for f in frames:
        f = f if isinstance(f, pd.DataFrame) else f.to_frame()
        fm = f.replace([np.inf, -np.inf], np.nan).notna().all(axis = 1)
        mask = fm if mask is None else (mask & fm)
    return mask[mask].index


def pos_mask(source_index, target_index):
    return source_index.isin(target_index)


def stack_node_array(feat_dict, valid_index):
    return np.stack(
        [df.loc[valid_index].values.astype(np.float32) for df in feat_dict.values()],
        axis = 2,
    )


def compute_hierarchy_indices(fine_index, coarse_index):
    idx = np.searchsorted(coarse_index.asi8, fine_index.asi8, side = "right") - 1
    return np.clip(idx, 0, len(coarse_index) - 1).astype(np.int32)


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


def compute_time_encoding(times_ns, freq):
    dt = pd.DatetimeIndex(times_ns)
    parts = [
        np.sin(2 * np.pi * dt.dayofweek / 7).astype(np.float32),
        np.cos(2 * np.pi * dt.dayofweek / 7).astype(np.float32),
        np.sin(2 * np.pi * dt.dayofyear / 365.25).astype(np.float32),
        np.cos(2 * np.pi * dt.dayofyear / 365.25).astype(np.float32),
    ]
    if freq in ("15m", "30m", "1h"):
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


def compute_edge_features(adj, signal_arr):
    adj_f = adj.astype(np.float32)
    delta = np.concatenate([np.zeros_like(adj_f[:1]), adj_f[1:] - adj_f[:-1]], axis = 0)
    spread = signal_arr[:, :, np.newaxis] - signal_arr[:, np.newaxis, :]
    mask = adj_f != 0.0
    return np.stack([adj_f * mask, delta * mask, spread * mask], axis = -1).astype(np.float16)


def build_targets(close, returns, low, horizon):
    future_ret = returns.shift(-horizon)
    fwd_vol = returns.rolling(max(2, horizon), min_periods = 2).std().shift(-horizon)
    sharpe = future_ret / (fwd_vol + 1e-9)
    drawdown = (low.rolling(horizon, min_periods = 1).min().shift(-horizon) - close) / (close + 1e-9)
    return future_ret, sharpe, drawdown


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
    return {col: data.pivot(index = "open_time", columns = "asset", values = col).ffill()
            for col in cols}


def parkinson_vol(high, low, window = 14):
    return (np.log(high / low).pow(2).rolling(window).mean() / (4 * _LOG2)).clip(lower = 0).pow(0.5)


def garman_klass_vol(high, low, close, open_, window = 14):
    term = 0.5 * np.log(high / low).pow(2) - _2LOG2M1 * np.log(close / open_).pow(2)
    return term.rolling(window).mean().clip(lower = 0).pow(0.5)


def yang_zhang_vol(high, low, close, open_, window = 14):
    k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
    rs = (np.log(high / close.clip(lower = 1e-9)) * np.log(high / open_.clip(lower = 1e-9))
          + np.log(low / close.clip(lower = 1e-9)) * np.log(low / open_.clip(lower = 1e-9)))
    var = (np.log(open_.clip(lower = 1e-9) / close.shift(1).clip(lower = 1e-9)).rolling(window).var()
           + k * np.log(close.clip(lower = 1e-9) / open_.clip(lower = 1e-9)).rolling(window).var()
           + (1 - k) * rs.rolling(window).mean())
    return var.clip(lower = 0).pow(0.5)


def compute_atr(high, low, close, period = 14):
    pc = close.shift(1)
    tr = pd.DataFrame(np.maximum(
        (high - low).values,
        np.maximum((high - pc).abs().values, (low - pc).abs().values),
    ), index = close.index, columns = close.columns)
    return tr.ewm(alpha = 1.0 / period, adjust = False).mean()


def compute_rsi(prices, period = 14):
    delta = prices.diff()
    gain = delta.clip(lower = 0).rolling(period).mean()
    loss = (-delta.clip(upper = 0)).rolling(period).mean()
    return (100 - (100 / (1 + gain / (loss + 1e-9)))) / 100


def compute_stoch_rsi(prices, rsi_period = 14, stoch_period = 14):
    rsi = compute_rsi(prices, rsi_period)
    lo = rsi.rolling(stoch_period).min()
    hi = rsi.rolling(stoch_period).max()
    return (rsi - lo) / (hi - lo + 1e-9)


def compute_cci(high, low, close, period = 14):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).std()
    return (tp - sma) / (0.015 * mad + 1e-9)


def compute_williams_r(high, low, close, period = 14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (hh - close) / (hh - ll + 1e-9) * -100


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


def compute_cmf(high, low, close, volume, period = 20):
    mfv = ((close - low) - (high - close)) / (high - low + 1e-9) * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)


def compute_obv_mom(close, volume, period = 14):
    direction = np.sign(np.log(close / close.shift(1)).fillna(0))
    obv = (direction * volume).cumsum()
    return obv.diff(period)


def amihud_illiq(returns, volume, window = 20):
    return (returns.abs() / volume.clip(lower = 1)).rolling(window).mean()


def roll_spread(returns, window = 20):
    cov = returns.rolling(window).cov(returns.shift(1))
    return 2 * np.sqrt((-cov).clip(lower = 0))


def kyle_lambda(returns, volume, window = 20):
    return (returns.abs() / volume.clip(lower = 1).pow(0.5)).rolling(window).mean()


def hurst_proxy(returns, window = 60, k = 5):
    var_1 = returns.rolling(window).var().clip(lower = 1e-9)
    var_k = returns.rolling(k).sum().rolling(window).var().clip(lower = 1e-9)
    return (0.5 * np.log(var_k / var_1) / np.log(k)).clip(0, 1)


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


def edge_stability(granger_w, window = GRANGER_STABILITY_WINDOW):
    T, N, _ = granger_w.shape
    stability = np.zeros_like(granger_w)
    for t in range(window, T):
        stability[t] = granger_w[t - window : t].std(axis = 0)
    return stability


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


def sentiment_placeholder(n_timesteps, n_assets):
    scores = np.zeros((n_timesteps, n_assets, SENTIMENT_FEATURES), dtype = np.float32)
    missing = np.ones(n_timesteps, dtype = np.float32)
    return scores, missing