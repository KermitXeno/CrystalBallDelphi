from dp_features import *


def process_model1(px_1h, px_4h):
    print("Building model1 dataset...")
    c, h, l, o = px_1h["close"], px_1h["high"], px_1h["low"], px_1h["open"]
    r = np.log(c / c.shift(1))
    vol = px_1h["volume"]
    assets = c.columns.tolist()
    N = len(assets)

    btc_col = next((col for col in c.columns if "BTC" in col), c.columns[0])
    btc_c = c[btc_col]
    btc_ema200 = btc_c.ewm(span = 200, adjust = False).mean()
    btc_ema_slope = btc_ema200.diff()
    btc_rv = r[btc_col].rolling(720, min_periods = 240).std()
    btc_rv_q50 = btc_rv.rolling(2160, min_periods = 720).quantile(0.50)

    vol_yz_1h = yang_zhang_vol(h, l, c, o, 30)
    vol_yz_4h = yang_zhang_vol(
        px_4h["high"], px_4h["low"], px_4h["close"], px_4h["open"], 30
    ).reindex(c.index, method = "ffill")

    adx, di_diff = compute_adx(h, l, c, 14)
    btc_adx_raw = adx[btc_col]
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    macd_line = c.ewm(span = 12, adjust = False).mean() - c.ewm(span = 26, adjust = False).mean()

    node_feats_1h = build_features({
        "ret_1": r,
        "ret_4": np.log(c / c.shift(4)),
        "ret_24": np.log(c / c.shift(24)),
        "ret_168": np.log(c / c.shift(168)),
        "vol_yz_1h": vol_yz_1h,
        "vol_yz_4h": vol_yz_4h,
        "vol_ratio": vol_yz_1h / (vol_yz_4h + 1e-9),
        "vol_pct": r.rolling(720, min_periods = 240).std().expanding(min_periods = 240).rank(pct = True),
        "rsi_14": compute_rsi(c, 14),
        "stoch_rsi": compute_stoch_rsi(c),
        "macd_hist": (macd_line - macd_line.ewm(span = 9, adjust = False).mean()) / (c + 1e-9),
        "bb_pos": (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9),
        "bb_width": 4 * bb_std / (bb_mid + 1e-9),
        "adx": adx,
        "di_diff": di_diff,
        "atr_norm": compute_atr(h, l, c, 14) / (c + 1e-9),
        "vol_zscore": vol / (vol.rolling(48).mean() + 1e-9),
        "amihud": amihud_illiq(r, vol, 24),
        "kyle_lam": kyle_lambda(r, vol, 24),
        **ichimoku_features(h, l, c),
    }, window = 168)

    c4, h4, l4, o4 = px_4h["close"], px_4h["high"], px_4h["low"], px_4h["open"]
    r4 = np.log(c4 / c4.shift(1))
    vol4 = px_4h["volume"]
    adx4, di_diff4 = compute_adx(h4, l4, c4, 14)

    node_feats_4h_raw = {
        "4h_ret_1": r4,
        "4h_ret_6": np.log(c4 / c4.shift(6)),
        "4h_ret_30": np.log(c4 / c4.shift(30)),
        "4h_vol_yz": yang_zhang_vol(h4, l4, c4, o4, 30),
        "4h_rsi_14": compute_rsi(c4, 14),
        "4h_adx": adx4,
        "4h_di_diff": di_diff4,
        "4h_amihud": amihud_illiq(r4, vol4, 24),
    }
    node_feats_4h = {
        name: normalise(df, 90).reindex(c.index, method = "ffill")
        for name, df in node_feats_4h_raw.items()
    }

    graph = np.load(os.path.join(BASE_DIR, "graph_edges.npz"))
    graph_idx = pd.DatetimeIndex(graph["times"])

    in_deg_df = pd.DataFrame(graph["node_in_degree"], index = graph_idx, columns = assets)
    out_deg_df = pd.DataFrame(graph["node_out_degree"], index = graph_idx, columns = assets)
    eigvec_df = pd.DataFrame(graph["node_eigvec"], index = graph_idx, columns = assets)
    clust_df = pd.DataFrame(graph["node_clustering"], index = graph_idx, columns = assets)
    between_df = pd.DataFrame(graph["node_betweenness"], index = graph_idx, columns = assets)

    graph_dens_s = pd.Series(graph["graph_density"], index = graph_idx, name = "graph_density")
    graph_ent_s = pd.Series(graph["graph_entropy"], index = graph_idx, name = "graph_entropy")

    node_feats_graph = build_features({
        "in_degree": in_deg_df.reindex(c.index, method = "ffill").fillna(0),
        "out_degree": out_deg_df.reindex(c.index, method = "ffill").fillna(0),
        "eigvec_cent": eigvec_df.reindex(c.index, method = "ffill").fillna(0),
        "clustering": clust_df.reindex(c.index, method = "ffill").fillna(0),
        "betweenness": between_df.reindex(c.index, method = "ffill").fillna(0),
    }, window = 168)

    node_feats_1h.update(node_feats_4h)
    node_feats_1h.update(node_feats_graph)

    r_clean = r.ffill().fillna(0)
    adj_corr = rolling_corr_matrix(r_clean, window = 120)
    ret_arr = r_clean.values.astype(np.float64)

    cmt_rho_A, cmt_theta, cmt_spec_gap, cmt_hub = compute_cmt(ret_arr, window = 120)
    msvb_ks, msvb_sl, msvb_sr, msvb_sc = compute_msvb(
        ret_arr, window = 120, scales = [1, 2, 4, 8, 16, 24])
    slfi_frust, slfi_fiedler, slfi_gap = compute_slfi(adj_corr)
    pef_pe, pef_deriv, pef_lifetime = compute_pef(adj_corr)

    idx = c.index
    node_feats_1h.update(build_features({
        "cmt_hub": pd.DataFrame(cmt_hub, index = idx, columns = assets),
        "msvb_kstar": pd.DataFrame(np.log1p(msvb_ks), index = idx, columns = assets),
        "msvb_sl": pd.DataFrame(msvb_sl, index = idx, columns = assets),
        "msvb_sr": pd.DataFrame(msvb_sr, index = idx, columns = assets),
        "msvb_sc": pd.DataFrame(msvb_sc, index = idx, columns = assets),
        "slfi_fiedler": pd.DataFrame(slfi_fiedler, index = idx, columns = assets),
        "pef_lifetime": pd.DataFrame(pef_lifetime, index = idx, columns = assets),
    }, window = 168))

    global_feats = normalise(pd.DataFrame({
        "graph_density": graph_dens_s.reindex(c.index, method = "ffill").fillna(0),
        "graph_entropy": graph_ent_s.reindex(c.index, method = "ffill").fillna(0),
        "cmt_rho_A": pd.Series(cmt_rho_A, index = idx),
        "cmt_theta": pd.Series(cmt_theta, index = idx),
        "cmt_spec_gap": pd.Series(cmt_spec_gap, index = idx),
        "slfi_frustration": pd.Series(slfi_frust, index = idx),
        "slfi_gap": pd.Series(slfi_gap, index = idx),
        "pef": pd.Series(pef_pe, index = idx),
        "pef_deriv": pd.Series(pef_deriv, index = idx),
        "breadth_24": (c > c.rolling(24).mean()).mean(axis = 1),
        "breadth_168": (c > c.rolling(168).mean()).mean(axis = 1),
        "ret_dispersion": np.log(c / c.shift(24)).std(axis = 1),
        "corr_regime": r.rolling(24).corr().groupby(level = 0).mean().mean(axis = 1),
        "btc_dominance": r[btc_col].rolling(24).std() / (r.rolling(24).std().mean(axis = 1) + 1e-9),
        "vol_of_vol": r.rolling(24).std().mean(axis = 1).rolling(24).std(),
    }), window = 168)

    regime = compute_regime_labels(px_1h)

    regime_shifted = regime.shift(-4)
    transition_label = (regime_shifted != regime).astype(np.float32)
    transition_label[transition_label.index[-4:]] = np.nan

    valid = get_valid_index(list(node_feats_1h.values()) + [global_feats, regime.to_frame(), transition_label.to_frame()])

    btc_raw = pd.DataFrame({
        "btc_ema_positive": (btc_ema_slope > 0).astype(float).fillna(0),
        "btc_adx_trending": (btc_adx_raw >= 0.20).astype(float).fillna(0),
        "btc_adx_raw": btc_adx_raw.clip(0, 1).fillna(0),
        "btc_rv_vs_q50": (btc_rv / (btc_rv_q50 + 1e-9)).clip(0, 5).fillna(1.0),
    }, index = idx)

    mask = pos_mask(graph_idx, valid)

    sent_scores, sent_missing = sentiment_placeholder(len(valid), N)

    save_npz("model1_dataset",
             node_features = stack_node_array(node_feats_1h, valid),
             global_features = global_feats.loc[valid].values.astype(np.float32),
             btc_raw = btc_raw.loc[valid].values.astype(np.float32),
             regime_labels = regime.loc[valid].values,
             transition_labels = transition_label.loc[valid].values.astype(np.float32),
             times = valid.asi8,
             time_enc = compute_time_encoding(valid.asi8, "1h"),
             sentiment_scores = sent_scores,
             sentiment_missing = sent_missing)
