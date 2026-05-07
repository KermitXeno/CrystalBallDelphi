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
    if len(valid) == 0:
        print(f"  DEBUG empty intersection - per-feature first-valid timestamps:")
        for fname, fdf in feats.items():
            mask_all = fdf.replace([np.inf, -np.inf], np.nan).notna().all(axis = 1)
            n_valid = mask_all.sum()
            first_valid = fdf.index[mask_all].min() if n_valid > 0 else None
            n_per_col = fdf.replace([np.inf, -np.inf], np.nan).notna().sum()
            min_col = n_per_col.idxmin()
            print(f"    {fname}: n_valid={n_valid} first_valid={first_valid} weakest={min_col}({n_per_col[min_col]})")
    return feats, valid

def _xrank(df):
    return df.rank(axis = 1, pct = True).fillna(0.5)

def _rolling_beta_btc(r, btc_r, window = 72, min_periods = 24):
    btc_var = btc_r.rolling(window, min_periods = min_periods).var().clip(lower = 1e-10)
    betas = pd.DataFrame(index = r.index, columns = r.columns, dtype = np.float64)
    for col in r.columns:
        cov_col = r[col].rolling(window, min_periods = min_periods).cov(btc_r)
        betas[col] = (cov_col / btc_var).clip(-5, 5)
    return betas.astype(np.float32)

def _build_4h_feats(px_4h):
    c, h, l, o = px_4h["close"], px_4h["high"], px_4h["low"], px_4h["open"]
    r = np.log(c / c.shift(1))
    vol = px_4h["volume"]
    bp = px_4h["taker_buy_base_volume"] / (vol + 1e-9)
    adx, _ = compute_adx(h, l, c, 14)
    vol_yz = yang_zhang_vol(h, l, c, o, 30)
    ret_4 = np.log(c / c.shift(4))
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    btc_col = next((col for col in c.columns if "BTC" in col), c.columns[0])
    btc_r = r[btc_col]
    feats = build_features({
        "ret_1": r,
        "ret_4": ret_4,
        "stoch_rsi": compute_stoch_rsi(c),
        "adx": adx,
        "tenkan_dev": (c - tenkan) / (c + 1e-9),
    }, window = 60)
    n_zscore = len(feats)
    excess_r1 = r.sub(btc_r, axis = 0)
    beta_btc = _rolling_beta_btc(r, btc_r, window = 30, min_periods = 10)
    idio_ret = r.sub(beta_btc.multiply(btc_r, axis = 0))
    idio_vol = idio_ret.rolling(30, min_periods = 10).std()
    rolling_max = c.rolling(12, min_periods = 1).max()
    vol_mean_30 = vol.rolling(30, min_periods = 8).mean()
    vol_std_30 = vol.rolling(30, min_periods = 8).std().clip(lower = 1e-9)
    vol_rank = vol.rank(axis = 1, pct = True).fillna(0.5)
    rolling_std_24 = r.rolling(24, min_periods = 8).std().clip(lower = 1e-8)
    r3 = r.pow(3)
    r2 = r.pow(2)
    r3_sum = r3.rolling(24, min_periods = 12).sum()
    r2_sum = r2.rolling(24, min_periods = 12).sum()
    amihud_raw = (r.abs() / (vol + 1e-9)).rolling(24, min_periods = 8).mean()
    pos_bars = (r > 0).astype(float)
    var_short = r.rolling(12, min_periods = 6).var().clip(lower = 1e-12)
    ret_4bar = np.log(c / c.shift(4))
    var_long = ret_4bar.rolling(8, min_periods = 4).var().clip(lower = 1e-12)
    down_sv = compute_downside_semivariance(r, 24)
    up_sv = compute_upside_semivariance(r, 24)
    pos_jv, neg_jv = compute_signed_jump_var(r, 24)
    feats["xrank_ret1"] = _xrank(r)
    feats["xrank_ret4"] = _xrank(ret_4)
    feats["xrank_vol"] = _xrank(vol_yz)
    feats["xrank_volume"] = _xrank(vol)
    feats["ret_vs_mean"] = winsorize(r.sub(r.mean(axis = 1), axis = 0)).fillna(0)
    feats["excess_ret1"] = winsorize(excess_r1).fillna(0)
    feats["adx_raw"] = adx.clip(0, 1).fillna(0.2)
    feats["vol_return_interact"] = (np.sign(r) * vol_rank).fillna(0)
    feats["reversal_signal"] = winsorize(r * (1.0 - vol_rank)).fillna(0)
    feats["detrended_volume"] = ((vol - vol_mean_30) / vol_std_30).clip(-5, 5).fillna(0)
    feats["idio_vol"] = normalise(idio_vol.fillna(0), 60)
    feats["jump_flag"] = (r.abs() / rolling_std_24).clip(0, 5).fillna(0)
    feats["realized_skew"] = (r3_sum / (r2_sum.pow(1.5).clip(lower = 1e-12) / 24.0 ** 0.5)).clip(-5, 5).fillna(0)
    feats["amihud_rank"] = _xrank(amihud_raw)
    feats["bp_delta"] = winsorize(bp - bp.shift(4)).fillna(0)
    feats["return_consistency"] = pos_bars.rolling(12, min_periods = 6).mean().fillna(0.5)
    feats["variance_ratio"] = (var_long / (4.0 * var_short)).clip(0.1, 5.0).fillna(1.0)
    feats["high_distance"] = ((c - rolling_max) / (rolling_max + 1e-9)).clip(-1, 0).fillna(0)
    feats["volume_momentum"] = (vol.rolling(4, min_periods = 2).mean() / vol.rolling(24, min_periods = 8).mean().clip(lower = 1e-9)).clip(0, 5).fillna(1.0)
    feats["realized_kurtosis"] = compute_realized_kurtosis(r, 24)
    feats["xrank_down_semivar"] = _xrank(down_sv)
    feats["xrank_up_semivar"] = _xrank(up_sv)
    feats["xrank_pos_jump"] = _xrank(pos_jv)
    feats["xrank_neg_jump"] = _xrank(neg_jv)
    feats["price_acceleration"] = winsorize(compute_price_acceleration(c, 4, 24)).fillna(0)
    feats["relative_strength_24"] = winsorize(compute_relative_strength(r, btc_r, 24)).fillna(0)
    feats["net_flow_persistence"] = compute_net_flow_persistence(bp, 12)
    feats["tail_ratio"] = compute_tail_ratio(r, 24)
    feats["xrank_max_return"] = _xrank(compute_max_return(r, 24))
    xs_median_r = r.median(axis = 1)
    alpha_t = r.sub(xs_median_r, axis = 0)
    past_alpha_6 = alpha_t.rolling(6, min_periods = 3).sum()
    past_alpha_12 = alpha_t.rolling(12, min_periods = 6).sum()
    past_alpha_24 = alpha_t.rolling(24, min_periods = 12).sum()
    feats["xrank_past_alpha_6"] = _xrank(past_alpha_6)
    feats["xrank_past_alpha_12"] = _xrank(past_alpha_12)
    feats["xrank_past_alpha_24"] = _xrank(past_alpha_24)
    n_added = len(feats) - n_zscore
    print(f"  4h features: {len(feats)} total ({n_zscore} z-scored + {n_added} cross-sectional/research)")
    ret_tgt, shr_tgt, dd_tgt = build_targets(c, r, l, horizon = 1)
    valid = get_valid_index(list(feats.values()) + [ret_tgt, shr_tgt, dd_tgt])
    return feats, ret_tgt, shr_tgt, dd_tgt, valid

def process_model2(px_15m, px_1h, px_4h):
    print("Building model2 dataset (4h primary horizon)...")
    assets = px_4h["close"].columns.tolist()
    N = len(assets)
    print(f"  asset count: {N}")
    print(f"  per-asset row counts (close):")
    for col in px_4h["close"].columns:
        n15 = px_15m["close"][col].dropna().shape[0] if col in px_15m["close"].columns else 0
        n1h = px_1h["close"][col].dropna().shape[0] if col in px_1h["close"].columns else 0
        n4h = px_4h["close"][col].dropna().shape[0]
        print(f"    {col}: 15m={n15} 1h={n1h} 4h={n4h}")
    feats_15m, valid_15m = _build_sub_hourly_feats(px_15m, window = 120)
    feats_1h, valid_1h_sub = _build_sub_hourly_feats(px_1h, window = 96)
    feats_4h, ret_4h, shr_4h, dd_4h, valid_4h = _build_4h_feats(px_4h)
    print(f"  raw valid counts: 15m={len(valid_15m)} 1h_sub={len(valid_1h_sub)} 4h={len(valid_4h)}")
    if len(valid_15m) == 0 or len(valid_1h_sub) == 0:
        raise RuntimeError("sub-hourly valid index is empty - one or more assets has insufficient data at this frequency (see per-asset counts above)")
    t0 = valid_4h[0]
    t1 = valid_4h[-1]
    valid_15m = valid_15m[(valid_15m >= t0) & (valid_15m <= t1)]
    valid_1h_sub = valid_1h_sub[(valid_1h_sub >= t0) & (valid_1h_sub <= t1)]
    print(f"  after 4h alignment: 15m={len(valid_15m)} 1h_sub={len(valid_1h_sub)}")
    hier_15m = compute_hierarchy_indices(valid_15m, valid_4h)
    hier_1h_sub = compute_hierarchy_indices(valid_1h_sub, valid_4h)
    end_15m = np.searchsorted(valid_15m.asi8, valid_4h.asi8, side = "right")
    start_15m = np.maximum(end_15m - 96, 0)
    end_1h = np.searchsorted(valid_1h_sub.asi8, valid_4h.asi8, side = "right")
    start_1h = np.maximum(end_1h - 48, 0)
    window_idx_15m = np.stack([start_15m, end_15m], axis = 1).astype(np.int32)
    window_idx_1h = np.stack([start_1h, end_1h], axis = 1).astype(np.int32)
    model1_output_placeholder = np.zeros((len(valid_4h), 6), dtype = np.float32)
    sent_scores_4h, sent_missing_4h = sentiment_placeholder(len(valid_4h), N)
    f4h_arr = stack_node_array(feats_4h, valid_4h)
    print(f"  final 4h feature array: {f4h_arr.shape} ({f4h_arr.shape[2]} features per asset)")
    save_npz("model2_dataset",
             features_15m = stack_node_array(feats_15m, valid_15m),
             features_30m = stack_node_array(feats_1h, valid_1h_sub),
             features_1h = f4h_arr,
             targets = np.stack([
                 ret_4h.loc[valid_4h].values,
                 shr_4h.loc[valid_4h].values,
                 dd_4h.loc[valid_4h].values,
             ], axis = 2).astype(np.float32),
             times_15m = valid_15m.asi8,
             times_30m = valid_1h_sub.asi8,
             times_1h = valid_4h.asi8,
             time_enc_15m = compute_time_encoding(valid_15m.asi8, "15m"),
             time_enc_30m = compute_time_encoding(valid_1h_sub.asi8, "1h"),
             time_enc_1h = compute_time_encoding(valid_4h.asi8, "4h"),
             hierarchy_15m_to_1h = hier_15m,
             hierarchy_30m_to_1h = hier_1h_sub,
             window_idx_15m = window_idx_15m,
             window_idx_30m = window_idx_1h,
             model1_outputs = model1_output_placeholder,
             sentiment_scores = sent_scores_4h,
             sentiment_missing = sent_missing_4h)