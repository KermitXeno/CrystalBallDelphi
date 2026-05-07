#to be deleted 

from dp_features import *


HORIZON_BARS_15M = {"1h": 4, "4h": 16, "16h": 64, "64h": 256}
GRANGER_WINDOWS = {"1h": 192, "4h": 96, "16h": 48, "64h": 32}
EDGE_INIT_SAMPLES = 500


def _make_init_edge_features(granger_native, returns_native, n_samples = EDGE_INIT_SAMPLES):
    T_native = granger_native.shape[0]
    N = granger_native.shape[1]
    if T_native <= 1:
        return np.zeros((N, N, 3), dtype = np.float32)
    start_idx = max(T_native // 4, 1)
    sample_idx = np.linspace(start_idx, T_native - 1, num = min(n_samples, T_native - start_idx), dtype = int)
    sample_idx = np.unique(sample_idx)
    adj_samples = granger_native[sample_idx]
    sparse = adj_samples.copy()
    sparse[sparse < GRANGER_SPARSE_THRESHOLD] = 0.0
    ret_arr = returns_native.values.astype(np.float32) if hasattr(returns_native, "values") else returns_native.astype(np.float32)
    ret_samples = ret_arr[sample_idx]
    edge_full = compute_edge_features(sparse, ret_samples)
    edge_avg = edge_full.astype(np.float32).mean(axis = 0)
    return edge_avg.astype(np.float32)


def _granger_init_per_horizon(c):
    log_ret_15m = np.log(c / c.shift(1))
    inits = {}
    summaries = {}
    for h_name, h_bars in HORIZON_BARS_15M.items():
        if h_name == "1h":
            r_native = log_ret_15m
            native_idx = c.index
        else:
            native_idx = c.index[h_bars - 1::h_bars]
            ds_close = c.reindex(native_idx)
            r_native = np.log(ds_close / ds_close.shift(1))
        r_clean = r_native.ffill().fillna(0)
        gw_native = granger_pairwise_weights(r_clean, window = GRANGER_WINDOWS[h_name], lag = GRANGER_LAG)
        edge_init = _make_init_edge_features(gw_native, r_clean)
        inits[h_name] = edge_init
        summaries[h_name] = (gw_native.shape[0], (gw_native > 0).mean())
    return inits, summaries


def process_graph_inits(px_15m):
    print("Building per-horizon graph adjacency initializations...")
    c = px_15m["close"]
    assets = c.columns.tolist()
    inits, summaries = _granger_init_per_horizon(c)
    for h_name in HORIZON_BARS_15M:
        T_native, frac = summaries[h_name]
        edge_init = inits[h_name]
        print(f"  {h_name}: native_T={T_native}  edge_init_shape={edge_init.shape}  granger_density={frac:.3f}")
    return inits, assets
