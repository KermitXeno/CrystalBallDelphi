from dp_features import *


def _build_sub_hourly_feats(px, window):
    c, h, l, o = px["close"], px["high"], px["low"], px["open"]
    r = np.log(c / c.shift(1))
    vol = px["volume"]
    bp = px["taker_buy_base_volume"] / (vol + 1e-9)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    macd_line = c.ewm(span = 12, adjust = False).mean() - c.ewm(span = 26, adjust = False).mean()
    vwap = ((c + h + l) / 3 * vol).rolling(48).sum() / (vol.rolling(48).sum() + 1e-9)
    ema20 = c.ewm(span = 20, adjust = False).mean()
    atr = compute_atr(h, l, c, 14)

    feats = build_features({
        "ret_1": r,
        "ret_4": np.log(c / c.shift(4)),
        "ret_16": np.log(c / c.shift(16)),
        "ret_48": np.log(c / c.shift(48)),
        "vol_yz": yang_zhang_vol(h, l, c, o, 30),
        "vol_ratio": yang_zhang_vol(h, l, c, o, 8) / (yang_zhang_vol(h, l, c, o, 48) + 1e-9),
        "rsi": compute_rsi(c, 14),
        "stoch_rsi": compute_stoch_rsi(c),
        "macd_hist": (macd_line - macd_line.ewm(span = 9, adjust = False).mean()) / (c + 1e-9),
        "bb_pos": (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9),
        "bb_width": 4 * bb_std / (bb_mid + 1e-9),
        "keltner_pos": (c - (ema20 - 2 * atr)) / (4 * atr + 1e-9),
        "cci": compute_cci(h, l, c, 14),
        "williams_r": compute_williams_r(h, l, c, 14),
        "atr_norm": atr / (c + 1e-9),
        "hl_spread": (h - l) / (c + 1e-9),
        "oc_body": (c - o) / (o + 1e-9),
        "vwap_dev": (c - vwap) / (vwap + 1e-9),
        "vol_zscore": vol / (vol.rolling(48).mean() + 1e-9),
        "buy_pres": bp,
        "buy_pres_q": px["taker_buy_quote_volume"] / (px["quote_asset_volume"] + 1e-9),
        "ord_imb": bp * 2 - 1,
        "cmf": compute_cmf(h, l, c, vol, 20),
        "obv_mom": compute_obv_mom(c, vol, 14),
        "trades_log": np.log1p(px["num_trades"]),
        "amihud": amihud_illiq(r, vol, 20),
        "roll_sp": roll_spread(r, 20),
        "kyle_lam": kyle_lambda(r, vol, 20),
        "vpin": (bp * 2 - 1).abs().rolling(50).mean(),
        **ichimoku_features(h, l, c),
    }, window = window)

    valid = get_valid_index(list(feats.values()))
    return feats, valid


def _build_1h_feats(px_1h):
    c, h, l, o = px_1h["close"], px_1h["high"], px_1h["low"], px_1h["open"]
    r = np.log(c / c.shift(1))
    vol = px_1h["volume"]
    bp = px_1h["taker_buy_base_volume"] / (vol + 1e-9)
    adx, di_diff = compute_adx(h, l, c, 14)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    macd_line = c.ewm(span = 12, adjust = False).mean() - c.ewm(span = 26, adjust = False).mean()
    vwap = ((c + h + l) / 3 * vol).rolling(24).sum() / (vol.rolling(24).sum() + 1e-9)
    vol_yz_1h = yang_zhang_vol(h, l, c, o, 30)

    feats = build_features({
        "ret_1": r,
        "ret_4": np.log(c / c.shift(4)),
        "ret_24": np.log(c / c.shift(24)),
        "ret_168": np.log(c / c.shift(168)),
        "vol_yz": vol_yz_1h,
        "vol_ratio": yang_zhang_vol(h, l, c, o, 8) / (yang_zhang_vol(h, l, c, o, 48) + 1e-9),
        "rsi_14": compute_rsi(c, 14),
        "stoch_rsi": compute_stoch_rsi(c),
        "macd_hist": (macd_line - macd_line.ewm(span = 9, adjust = False).mean()) / (c + 1e-9),
        "bb_pos": (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9),
        "bb_width": 4 * bb_std / (bb_mid + 1e-9),
        "adx": adx,
        "di_diff": di_diff,
        "atr_norm": compute_atr(h, l, c, 14) / (c + 1e-9),
        "vwap_dev": (c - vwap) / (vwap + 1e-9),
        "vol_zscore": vol / (vol.rolling(48).mean() + 1e-9),
        "buy_pres": bp,
        "ord_imb": bp * 2 - 1,
        "amihud": amihud_illiq(r, vol, 24),
        "roll_sp": roll_spread(r, 24),
        "kyle_lam": kyle_lambda(r, vol, 24),
        "vpin": (bp * 2 - 1).abs().rolling(50).mean(),
        **ichimoku_features(h, l, c),
    }, window = 168)

    ret_tgt, shr_tgt, dd_tgt = build_targets(c, r, l, horizon = 1)
    valid = get_valid_index(list(feats.values()) + [ret_tgt, shr_tgt, dd_tgt])
    return feats, ret_tgt, shr_tgt, dd_tgt, valid


def process_model2(px_15m, px_30m, px_1h):
    print("Building model2 dataset...")
    assets = px_1h["close"].columns.tolist()
    N = len(assets)

    feats_15m, valid_15m = _build_sub_hourly_feats(px_15m, window = 120)
    feats_30m, valid_30m = _build_sub_hourly_feats(px_30m, window = 96)
    feats_1h, ret_1h, shr_1h, dd_1h, valid_1h = _build_1h_feats(px_1h)

    t0 = valid_1h[0]
    valid_15m = valid_15m[valid_15m >= t0]
    valid_30m = valid_30m[valid_30m >= t0]

    graph = np.load(os.path.join(BASE_DIR, "graph_edges.npz"))
    graph_idx = pd.DatetimeIndex(graph["times"])
    mask_1h = pos_mask(graph_idx, valid_1h)

    in_deg_df = pd.DataFrame(graph["node_in_degree"], index = graph_idx, columns = assets)
    out_deg_df = pd.DataFrame(graph["node_out_degree"], index = graph_idx, columns = assets)
    eigvec_df = pd.DataFrame(graph["node_eigvec"], index = graph_idx, columns = assets)
    clust_df = pd.DataFrame(graph["node_clustering"], index = graph_idx, columns = assets)
    between_df = pd.DataFrame(graph["node_betweenness"], index = graph_idx, columns = assets)
    graph_dens_s = pd.Series(graph["graph_density"], index = graph_idx)
    graph_ent_s = pd.Series(graph["graph_entropy"], index = graph_idx)

    dens_bc = pd.DataFrame(
        np.repeat(
            graph_dens_s.reindex(valid_1h, method = "ffill").fillna(0).values[:, None],
            N, axis = 1),
        index = valid_1h, columns = assets)
    ent_bc = pd.DataFrame(
        np.repeat(
            graph_ent_s.reindex(valid_1h, method = "ffill").fillna(0).values[:, None],
            N, axis = 1),
        index = valid_1h, columns = assets)

    feats_1h.update(build_features({
        "in_degree": in_deg_df.reindex(valid_1h, method = "ffill").fillna(0),
        "out_degree": out_deg_df.reindex(valid_1h, method = "ffill").fillna(0),
        "eigvec_cent": eigvec_df.reindex(valid_1h, method = "ffill").fillna(0),
        "clustering": clust_df.reindex(valid_1h, method = "ffill").fillna(0),
        "betweenness": between_df.reindex(valid_1h, method = "ffill").fillna(0),
        "graph_density": dens_bc,
        "graph_entropy": ent_bc,
    }, window = 168))

    hier_15m = compute_hierarchy_indices(valid_15m, valid_1h)
    hier_30m = compute_hierarchy_indices(valid_30m, valid_1h)

    lookback_15m = 96
    lookback_30m = 48
    end_15m = np.searchsorted(valid_15m.asi8, valid_1h.asi8, side = "right")
    start_15m = np.maximum(end_15m - lookback_15m, 0)
    end_30m = np.searchsorted(valid_30m.asi8, valid_1h.asi8, side = "right")
    start_30m = np.maximum(end_30m - lookback_30m, 0)
    window_idx_15m = np.stack([start_15m, end_15m], axis = 1).astype(np.int32)
    window_idx_30m = np.stack([start_30m, end_30m], axis = 1).astype(np.int32)

    model1_output_placeholder = np.zeros((len(valid_1h), 6), dtype = np.float32)
    sent_scores_1h, sent_missing_1h = sentiment_placeholder(len(valid_1h), N)

    save_npz("model2_dataset",
             features_15m = stack_node_array(feats_15m, valid_15m),
             features_30m = stack_node_array(feats_30m, valid_30m),
             features_1h = stack_node_array(feats_1h, valid_1h),
             targets = np.stack([
                 ret_1h.loc[valid_1h].values,
                 shr_1h.loc[valid_1h].values,
                 dd_1h.loc[valid_1h].values,
             ], axis = 2).astype(np.float32),
             times_15m = valid_15m.asi8,
             times_30m = valid_30m.asi8,
             times_1h = valid_1h.asi8,
             time_enc_15m = compute_time_encoding(valid_15m.asi8, "15m"),
             time_enc_30m = compute_time_encoding(valid_30m.asi8, "30m"),
             time_enc_1h = compute_time_encoding(valid_1h.asi8, "1h"),
             hierarchy_15m_to_1h = hier_15m,
             hierarchy_30m_to_1h = hier_30m,
             window_idx_15m = window_idx_15m,
             window_idx_30m = window_idx_30m,
             model1_outputs = model1_output_placeholder,
             sentiment_scores = sent_scores_1h,
             sentiment_missing = sent_missing_1h)
