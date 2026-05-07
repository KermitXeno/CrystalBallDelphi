from dp_features import *

TRANSITION_LOOKAHEAD_15M = 16
TRANSITION_PERSISTENCE_15M = 32
TRANSITION_MIN_AGREEMENT = 0.75


def _persistent_transition_labels(regime_argmax, n_classes):
    windows = [regime_argmax.shift(-i) for i in range(TRANSITION_LOOKAHEAD_15M,
                                                      TRANSITION_LOOKAHEAD_15M + TRANSITION_PERSISTENCE_15M)]
    future = pd.concat(windows, axis = 1)
    threshold_count = TRANSITION_PERSISTENCE_15M * TRANSITION_MIN_AGREEMENT
    transition = pd.Series(0.0, index = regime_argmax.index, dtype = np.float32)
    for k in range(n_classes):
        eq_k = (future == k).sum(axis = 1)
        persistent_to_k = (eq_k >= threshold_count) & (regime_argmax != k)
        transition[persistent_to_k.values] = 1.0
    invalid_tail = TRANSITION_LOOKAHEAD_15M + TRANSITION_PERSISTENCE_15M - 1
    transition.iloc[-invalid_tail:] = np.nan
    invalid_rows = future.isna().any(axis = 1)
    transition[invalid_rows.values] = np.nan
    return transition


def _features_for_horizon(px_15m, h_name):
    h_bars = HORIZON_BARS_15M[h_name]
    lookback = h_bars
    indicators = compute_horizon_indicators(px_15m, lookback)
    block_features = sample_at_block_offsets(indicators, h_bars)
    aggregated = aggregate_blocks_mean_slope(block_features)
    norm_window = max(64, h_bars * 4)
    aggregated_norm = {}
    for name, df in aggregated.items():
        aggregated_norm[f"{h_name}_{name}"] = normalise(df, window = norm_window)
    return aggregated_norm


def _compute_pef_deriv_1h(px_15m):
    c = px_15m["close"]
    r_15m = np.log(c / c.shift(1)).ffill().fillna(0)
    idx_1h = c.index[3::4]
    r_1h = r_15m.reindex(idx_1h)
    adj_corr = rolling_corr_matrix(r_1h, window = 64)
    _, pe_deriv, _ = compute_pef(adj_corr)
    pe_deriv_s = pd.Series(pe_deriv, index = idx_1h, name = "pef_deriv")
    pe_deriv_15m = pe_deriv_s.reindex(c.index, method = "ffill").fillna(0)
    return pe_deriv_15m


def _global_features_15m(px_15m):
    c = px_15m["close"]
    r = np.log(c / c.shift(1))
    btc_col = next((col for col in c.columns if "BTC" in col), c.columns[0])
    print("  computing pef_deriv at 1h cadence...")
    pef_deriv = _compute_pef_deriv_1h(px_15m)
    breadth_64 = (c > c.rolling(64).mean()).mean(axis = 1)
    breadth_256 = (c > c.rolling(256).mean()).mean(axis = 1)
    ret_dispersion_64 = np.log(c / c.shift(64)).std(axis = 1)
    ret_dispersion_256 = np.log(c / c.shift(256)).std(axis = 1)
    btc_vol = r[btc_col].rolling(64).std()
    avg_vol = r.rolling(64).std().mean(axis = 1)
    btc_dominance = btc_vol / (avg_vol + 1e-9)
    vol_of_vol = r.rolling(64).std().mean(axis = 1).rolling(64).std()
    rolling_corr = r.rolling(64).corr().groupby(level = 0).mean().mean(axis = 1)
    return normalise(pd.DataFrame({
        "breadth_64": breadth_64,
        "breadth_256": breadth_256,
        "ret_dispersion_64": ret_dispersion_64,
        "ret_dispersion_256": ret_dispersion_256,
        "btc_dominance": btc_dominance,
        "vol_of_vol": vol_of_vol,
        "corr_regime": rolling_corr,
        "pef_deriv": pef_deriv,
    }), window = 256)


def _btc_raw_features_15m(px_15m):
    c = px_15m["close"]
    h = px_15m["high"]
    l = px_15m["low"]
    btc_col = next((col for col in c.columns if "BTC" in col), c.columns[0])
    btc_c = c[btc_col]
    btc_h = h[btc_col]
    btc_l = l[btc_col]
    btc_ema = btc_c.ewm(span = 1024, adjust = False).mean()
    btc_ema_slope = btc_ema.diff()
    btc_r = np.log(btc_c / btc_c.shift(1))
    btc_rv = btc_r.rolling(1024, min_periods = 256).std()
    btc_rv_q50 = btc_rv.rolling(4096, min_periods = 1024).quantile(0.50)
    adx, _ = compute_adx(btc_h.to_frame("b"), btc_l.to_frame("b"), btc_c.to_frame("b"), 64)
    btc_adx = adx.iloc[:, 0]
    return pd.DataFrame({
        "btc_ema_positive": (btc_ema_slope > 0).astype(float).fillna(0),
        "btc_adx_raw": btc_adx.clip(0, 1).fillna(0),
        "btc_rv_vs_q50": (btc_rv / (btc_rv_q50 + 1e-9)).clip(0, 5).fillna(1.0),
    }, index = c.index)


def process_model1(px_15m):
    print("Building model1 dataset (15m cadence, multi-horizon blocks)...")
    c = px_15m["close"]
    assets = c.columns.tolist()
    N = len(assets)
    print(f"  assets: {N}  raw 15m bars: {len(c)}")

    all_node_feats = {}
    for h_name in HORIZON_BARS_15M:
        feats = _features_for_horizon(px_15m, h_name)
        n_feat = len(feats)
        all_node_feats.update(feats)
        print(f"  {h_name}: {n_feat} features (mean+slope per base)")
    print(f"  total node features per asset: {len(all_node_feats)}")

    global_feats = _global_features_15m(px_15m)
    print(f"  global features: {global_feats.shape[1]}")

    btc_raw = _btc_raw_features_15m(px_15m)
    print(f"  btc_raw features: {btc_raw.shape[1]}")

    regime_scores = compute_regime_scores_15m(px_15m)
    regime_argmax = pd.Series(regime_scores.values.argmax(axis = 1).astype(np.int8),
                              index = regime_scores.index)
    bull_avg = float(regime_scores["bull"].mean())
    bear_avg = float(regime_scores["bear"].mean())
    neutral_avg = float(regime_scores["neutral"].mean())
    print(f"  regime scores avg: bull={bull_avg:.3f} bear={bear_avg:.3f} neutral={neutral_avg:.3f}")

    n_classes = 3
    transition_label = _persistent_transition_labels(regime_argmax, n_classes)
    raw_flips = (regime_argmax.shift(-TRANSITION_LOOKAHEAD_15M) != regime_argmax).astype(np.float32)
    raw_flips.iloc[-TRANSITION_LOOKAHEAD_15M:] = np.nan
    raw_rate = float(raw_flips.dropna().mean())
    persistent_rate = float(transition_label.dropna().mean())
    print(f"  transition: raw={raw_rate:.4f}  persistent={persistent_rate:.4f}")

    valid = get_valid_index(list(all_node_feats.values()) + [global_feats, regime_scores,
                                                              transition_label.to_frame()])
    print(f"  valid timesteps: {len(valid)}")

    sent_scores, sent_missing = sentiment_placeholder(len(valid), N)

    save_npz("model1_dataset",
             node_features = stack_node_array(all_node_feats, valid),
             global_features = global_feats.loc[valid].values.astype(np.float32),
             btc_raw = btc_raw.loc[valid].values.astype(np.float32),
             regime_scores = regime_scores.loc[valid].values.astype(np.float32),
             regime_labels = regime_argmax.loc[valid].values.astype(np.int8),
             transition_labels = transition_label.loc[valid].values.astype(np.float32),
             times = valid.asi8,
             time_enc = compute_time_encoding(valid.asi8, "15m"),
             sentiment_scores = sent_scores,
             sentiment_missing = sent_missing,
             horizon_names = np.array(list(HORIZON_BARS_15M.keys()), dtype = "U8"))
