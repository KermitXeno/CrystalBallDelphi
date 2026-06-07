import os
import numpy as np
import pandas as pd
from dp_features import (
    winsorize, normalise, get_valid_index, stack_node_array,
    compute_time_encoding, save_npz, sentiment_placeholder,
    compute_hierarchical_indicators, sample_blocks_hierarchical,
    aggregate_blocks_mean_slope, aggregate_ohlcv, yang_zhang_vol,
    compute_adx, compute_rsi, compute_atr,
    HORIZON_CFG, BASE_DIR,
)


def _xrank(df):
    return df.rank(axis = 1, pct = True).fillna(0.5)


def _rolling_beta_btc(r, btc_r, window = 30, min_periods = 10):
    btc_var = btc_r.rolling(window, min_periods = min_periods).var().clip(lower = 1e-10)
    betas = pd.DataFrame(index = r.index, columns = r.columns, dtype = np.float64)
    for col in r.columns:
        cov_col = r[col].rolling(window, min_periods = min_periods).cov(btc_r)
        betas[col] = (cov_col / btc_var).clip(-5, 5)
    return betas.astype(np.float32)


def _build_15m_timing(px_15m, window = 120):
    c = px_15m["close"]
    h = px_15m["high"]
    l = px_15m["low"]
    v = px_15m["volume"]
    r = np.log(c / c.shift(1))
    bp = px_15m["taker_buy_base_volume"] / (v + 1e-9)
    vol_mean = v.rolling(48, min_periods = 8).mean().clip(lower = 1e-9)
    feats = {
        "ret_1": normalise(r.fillna(0), window),
        "ret_4": normalise(np.log(c / c.shift(4)).fillna(0), window),
        "vol_surge": (v / vol_mean).clip(0, 10).fillna(1.0),
        "buy_pressure": bp.fillna(0.5),
        "order_imbalance": (bp * 2 - 1).fillna(0.0),
        "hl_spread": normalise(((h - l) / (c + 1e-9)).fillna(0), window),
    }
    valid = get_valid_index(list(feats.values()))
    return feats, valid


def _build_1h_hierarchical(px_15m):
    indicators = compute_hierarchical_indicators(px_15m, "1h")
    phase_shift = HORIZON_CFG["1h"]["phase_shift"]
    blocks = sample_blocks_hierarchical(indicators, phase_shift, n_blocks = 4)
    feats = aggregate_blocks_mean_slope(blocks)
    valid = get_valid_index(list(feats.values()))
    return feats, valid


def _build_4h_features(px_15m):
    indicators_4h = compute_hierarchical_indicators(px_15m, "4h")
    blocks_4h = sample_blocks_hierarchical(indicators_4h, HORIZON_CFG["4h"]["phase_shift"], n_blocks = 4)
    feats_4h = aggregate_blocks_mean_slope(blocks_4h)
    indicators_16h = compute_hierarchical_indicators(px_15m, "16h")
    blocks_16h = sample_blocks_hierarchical(indicators_16h, HORIZON_CFG["16h"]["phase_shift"], n_blocks = 4)
    feats_16h = aggregate_blocks_mean_slope(blocks_16h)
    for k, v in feats_16h.items():
        feats_4h[f"16h_{k}"] = v
    indicators_64h = compute_hierarchical_indicators(px_15m, "64h")
    blocks_64h = sample_blocks_hierarchical(indicators_64h, HORIZON_CFG["64h"]["phase_shift"], n_blocks = 4)
    feats_64h = aggregate_blocks_mean_slope(blocks_64h)
    for k, v in feats_64h.items():
        feats_4h[f"64h_{k}"] = v
    c = px_15m["close"]
    h = px_15m["high"]
    l = px_15m["low"]
    o = px_15m["open"]
    v = px_15m["volume"]
    r = np.log(c / c.shift(1))
    agg = aggregate_ohlcv(px_15m, 16)
    c4 = agg["close"]
    r4 = np.log(c4 / c4.shift(16))
    btc_col = next((col for col in c.columns if "BTC" in col), c.columns[0])
    btc_r = r[btc_col]
    r4_btc = np.log(c4[btc_col] / c4[btc_col].shift(16))
    excess_r = r4.sub(r4_btc, axis = 0)
    beta_btc = _rolling_beta_btc(r, btc_r, window = 480, min_periods = 160)
    idio_ret = r.sub(beta_btc.multiply(btc_r, axis = 0))
    idio_vol = idio_ret.rolling(480, min_periods = 160).std()
    feats_4h["xrank_ret4"] = _xrank(r4)
    feats_4h["xrank_vol"] = _xrank(yang_zhang_vol(h, l, c, o, 480))
    feats_4h["excess_ret"] = winsorize(excess_r).fillna(0)
    feats_4h["idio_vol"] = normalise(idio_vol.fillna(0), 960)
    feats_4h["beta_btc"] = beta_btc.clip(-3, 3).fillna(1.0)
    vol_rank = v.rank(axis = 1, pct = True).fillna(0.5)
    feats_4h["vol_return_interact"] = (np.sign(r) * vol_rank).fillna(0)
    pos_bars = (r > 0).astype(float)
    feats_4h["return_consistency"] = pos_bars.rolling(192, min_periods = 48).mean().fillna(0.5)
    rolling_max = c.rolling(384, min_periods = 16).max()
    feats_4h["high_distance"] = ((c - rolling_max) / (rolling_max + 1e-9)).clip(-1, 0).fillna(0)
    xs_median_r = r4.median(axis = 1)
    alpha_t = r4.sub(xs_median_r, axis = 0)
    past_alpha_96 = alpha_t.rolling(96, min_periods = 48).sum()
    past_alpha_384 = alpha_t.rolling(384, min_periods = 192).sum()
    feats_4h["xrank_past_alpha_96"] = _xrank(past_alpha_96)
    feats_4h["xrank_past_alpha_384"] = _xrank(past_alpha_384)
    ret_4h = r4.shift(-16).fillna(0)
    sharpe_4h = ret_4h / (r.rolling(384, min_periods = 96).std().clip(lower = 1e-6))
    dd_4h = (c - rolling_max) / (rolling_max + 1e-9)
    n_hier = sum(1 for k in feats_4h if not k.startswith("xrank") and not k.startswith("excess")
                 and not k.startswith("idio") and not k.startswith("beta")
                 and not k.startswith("vol_return") and not k.startswith("return_cons")
                 and not k.startswith("high_dist") and k != "vol_return_interact")
    print(f"  4h features: {len(feats_4h)} total ({n_hier} hierarchical + {len(feats_4h) - n_hier} cross-sectional)")
    valid = get_valid_index(list(feats_4h.values()) + [ret_4h, sharpe_4h, dd_4h])
    return feats_4h, ret_4h, sharpe_4h, dd_4h, valid


def process_model2(px_15m):
    print("Building model2 dataset (hierarchical multi-horizon)...")
    c = px_15m["close"]
    assets = c.columns.tolist()
    N = len(assets)
    print(f"  assets: {N}")

    print("  building 15m timing features...")
    feats_15m, valid_15m = _build_15m_timing(px_15m)
    print(f"  15m: {len(feats_15m)} features, {len(valid_15m)} valid bars")

    print("  building 1h hierarchical features...")
    feats_1h, valid_1h = _build_1h_hierarchical(px_15m)
    print(f"  1h: {len(feats_1h)} features, {len(valid_1h)} valid bars")

    print("  building 4h hierarchical + cross-sectional features...")
    feats_4h, ret_4h, shr_4h, dd_4h, valid_4h = _build_4h_features(px_15m)
    print(f"  4h: {len(feats_4h)} features, {len(valid_4h)} valid bars")

    idx_1h = valid_15m[3::4]
    valid_1h = valid_1h.intersection(idx_1h)
    idx_4h = valid_15m[15::16]
    valid_4h_aligned = valid_4h.intersection(idx_4h)
    if len(valid_4h_aligned) < len(valid_4h):
        print(f"  4h alignment: {len(valid_4h)} -> {len(valid_4h_aligned)}")
        valid_4h = valid_4h_aligned

    t0 = valid_4h[0]
    t1 = valid_4h[-1]
    valid_15m = valid_15m[(valid_15m >= t0) & (valid_15m <= t1)]
    valid_1h = valid_1h[(valid_1h >= t0) & (valid_1h <= t1)]
    print(f"  after alignment: 15m={len(valid_15m)} 1h={len(valid_1h)} 4h={len(valid_4h)}")

    end_15m = np.searchsorted(valid_15m.asi8, valid_4h.asi8, side = "right")
    start_15m = np.maximum(end_15m - 96, 0)
    end_1h = np.searchsorted(valid_1h.asi8, valid_4h.asi8, side = "right")
    start_1h = np.maximum(end_1h - 48, 0)
    window_idx_15m = np.stack([start_15m, end_15m], axis = 1).astype(np.int32)
    window_idx_1h = np.stack([start_1h, end_1h], axis = 1).astype(np.int32)

    m1_path = os.path.join(BASE_DIR, "model1_outputs.npz")
    if os.path.exists(m1_path):
        m1_data = np.load(m1_path)
        if "model1_outputs_15m" in m1_data.files:
            m1_outputs_15m = m1_data["model1_outputs_15m"]
            m1_times_15m = m1_data["times_15m"]
            m1_idx = np.searchsorted(m1_times_15m, valid_4h.asi8)
            valid_mask = m1_idx < len(m1_times_15m)
            model1_outputs = np.zeros((len(valid_4h), m1_outputs_15m.shape[1]), dtype = np.float32)
            model1_outputs[valid_mask] = m1_outputs_15m[m1_idx[valid_mask]]
            print(f"  loaded model1_outputs at 15m cadence: {model1_outputs.shape} (zero staleness)")
        else:
            m1_outputs_1h = m1_data["model1_outputs"] if "model1_outputs" in m1_data.files else m1_data["model1_outputs_1h"]
            m1_times_1h = m1_data["times_1h"]
            m1_idx = np.searchsorted(m1_times_1h, valid_4h.asi8)
            valid_mask = m1_idx < len(m1_times_1h)
            model1_outputs = np.zeros((len(valid_4h), m1_outputs_1h.shape[1]), dtype = np.float32)
            model1_outputs[valid_mask] = m1_outputs_1h[m1_idx[valid_mask]]
            print(f"  loaded model1_outputs at 1h cadence: {model1_outputs.shape} (up to 3h stale)")
    else:
        model1_outputs = np.zeros((len(valid_4h), 5), dtype = np.float32)
        print(f"  WARNING: {m1_path} not found, using zeros")

    sent_scores, sent_missing = sentiment_placeholder(len(valid_4h), N)

    arr_4h = stack_node_array(feats_4h, valid_4h)
    arr_1h = stack_node_array(feats_1h, valid_1h)
    arr_15m = stack_node_array(feats_15m, valid_15m)

    feature_names_4h = np.array(list(feats_4h.keys()), dtype = "U40")
    feature_names_1h = np.array(list(feats_1h.keys()), dtype = "U40")
    feature_names_15m = np.array(list(feats_15m.keys()), dtype = "U40")

    print(f"  arrays: 15m={arr_15m.shape} 1h={arr_1h.shape} 4h={arr_4h.shape}")

    save_npz("model2_dataset",
             features_15m = arr_15m,
             features_1h = arr_1h,
             features_4h = arr_4h,
             targets = np.stack([
                 ret_4h.loc[valid_4h].values,
                 shr_4h.loc[valid_4h].values,
                 dd_4h.loc[valid_4h].values,
             ], axis = 2).astype(np.float32),
             times_15m = valid_15m.asi8,
             times_1h = valid_1h.asi8,
             times_4h = valid_4h.asi8,
             time_enc_15m = compute_time_encoding(valid_15m.asi8, "15m"),
             time_enc_1h = compute_time_encoding(valid_1h.asi8, "1h"),
             time_enc_4h = compute_time_encoding(valid_4h.asi8, "4h"),
             window_idx_15m = window_idx_15m,
             window_idx_1h = window_idx_1h,
             model1_outputs = model1_outputs,
             sentiment_scores = sent_scores,
             sentiment_missing = sent_missing,
             feature_names_4h = feature_names_4h,
             feature_names_1h = feature_names_1h,
             feature_names_15m = feature_names_15m)
