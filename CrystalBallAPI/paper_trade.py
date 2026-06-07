import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "Models")
sys.path.insert(0, MODELS_DIR)

from dp_features import (
    build_price_matrices, compute_hierarchical_indicators,
    sample_blocks_hierarchical, aggregate_blocks_mean_slope,
    aggregate_ohlcv, yang_zhang_vol, compute_adx,
    normalise, winsorize, compute_time_encoding,
    HORIZON_CFG, KEEP_INDICATORS, BASE_DIR,
)
from dp_preproc_model1 import _global_features_15m, _btc_raw_features_15m
from dp_download import SYMBOLS, BASE_URL, COLUMNS, FLOAT_COLS

LOG_DIR = os.path.join(SCRIPT_DIR, "paper_trade_logs")
os.makedirs(LOG_DIR, exist_ok = True)

TAKER_FEE = 0.001
SLIPPAGE_BY_TIER = {
    "large": 0.0003,
    "mid": 0.0005,
    "small": 0.0010,
}
TIER_MAP = {
    "BTCUSDT": "large", "ETHUSDT": "large", "BNBUSDT": "large",
    "SOLUSDT": "mid", "XRPUSDT": "mid", "ADAUSDT": "mid",
    "DOGEUSDT": "mid", "LTCUSDT": "mid", "LINKUSDT": "mid",
    "AVAXUSDT": "mid", "DOTUSDT": "mid", "MATICUSDT": "mid",
    "BCHUSDT": "mid", "ETCUSDT": "mid",
    "AAVEUSDT": "small", "ALGOUSDT": "small", "ATOMUSDT": "small",
    "FILUSDT": "small", "NEARUSDT": "small", "UNIUSDT": "small",
    "XTZUSDT": "small",
}
INITIAL_CAPITAL = 10000.0
UPDATE_INTERVAL_S = 4 * 3600
HISTORY_BARS = 1000


def fetch_recent_candles(symbol, interval = "15m", limit = HISTORY_BARS):
    try:
        resp = requests.get(BASE_URL, params = {
            "symbol": symbol, "interval": interval, "limit": limit,
        }, timeout = 15)
        if resp.status_code != 200:
            print(f"  {symbol} fetch failed: {resp.status_code}")
            return None
        rows = resp.json()
        df = pd.DataFrame(rows, columns = COLUMNS).drop(columns = ["ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit = "ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit = "ms")
        df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)
        df["num_trades"] = df["num_trades"].astype(int)
        df = df.set_index("open_time")
        return df
    except Exception as e:
        print(f"  {symbol} fetch error: {e}")
        return None


def fetch_all_assets(symbols = SYMBOLS, interval = "15m", limit = HISTORY_BARS):
    print(f"Fetching {limit} {interval} candles for {len(symbols)} assets...")
    all_data = {}
    for sym in symbols:
        df = fetch_recent_candles(sym, interval, limit)
        if df is not None and len(df) > 100:
            all_data[sym] = df
            time.sleep(0.12)
        else:
            print(f"  WARNING: {sym} returned insufficient data")
    return all_data


def build_live_price_matrices(all_data):
    common_idx = None
    for sym, df in all_data.items():
        idx = df.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    px = {}
    for field in ["open", "high", "low", "close", "volume",
                   "taker_buy_base_volume", "taker_buy_quote_volume",
                   "quote_asset_volume", "num_trades"]:
        px[field] = pd.DataFrame(
            {sym: df.loc[common_idx, field] for sym, df in all_data.items()},
            index = common_idx)
    return px


def compute_model1_features(px_15m):
    from dp_preproc_model1 import _global_features_15m, _btc_raw_features_15m
    c = px_15m["close"]
    assets = c.columns.tolist()
    all_feats = {}
    for h_name in HORIZON_CFG:
        indicators = compute_hierarchical_indicators(px_15m, h_name)
        phase_shift = HORIZON_CFG[h_name]["phase_shift"]
        blocks = sample_blocks_hierarchical(indicators, phase_shift, n_blocks = 4)
        feats = aggregate_blocks_mean_slope(blocks)
        for k, v in feats.items():
            all_feats[f"{h_name}_{k}"] = v
    node_arr = np.stack([all_feats[k].iloc[-1].values for k in sorted(all_feats.keys())], axis = -1)
    market_features = node_arr.mean(axis = 0)
    glob = _global_features_15m(px_15m)
    global_features = glob.iloc[-1].values.astype(np.float32)
    btc = _btc_raw_features_15m(px_15m)
    btc_raw = btc.iloc[-1].values.astype(np.float32)
    X = np.concatenate([market_features, global_features, btc_raw]).reshape(1, -1).astype(np.float32)
    return X


def run_model1(X):
    from cascade_generate import load_model1, model1_predict
    dir_model, phase_model = load_model1()
    outputs = model1_predict(dir_model, phase_model, X)
    return outputs[0]


def compute_model2_input(px_15m, m1_output):
    import torch
    c = px_15m["close"]
    N = c.shape[1]
    feats_4h = {}
    for h_name in ["4h", "16h", "64h"]:
        indicators = compute_hierarchical_indicators(px_15m, h_name)
        phase_shift = HORIZON_CFG[h_name]["phase_shift"]
        blocks = sample_blocks_hierarchical(indicators, phase_shift, n_blocks = 4)
        feats = aggregate_blocks_mean_slope(blocks)
        prefix = f"{h_name}_" if h_name != "4h" else ""
        for k, v in feats.items():
            feats_4h[f"{prefix}{k}"] = v
    h, l, o, v = px_15m["high"], px_15m["low"], px_15m["open"], px_15m["volume"]
    r = np.log(c / c.shift(1))
    btc_col = next((col for col in c.columns if "BTC" in col), c.columns[0])
    btc_r = r[btc_col]
    agg = aggregate_ohlcv(px_15m, 16)
    c4 = agg["close"]
    r4 = np.log(c4 / c4.shift(16))
    from dp_preproc_model2 import _xrank, _rolling_beta_btc
    excess_r = r4.sub(np.log(c4[btc_col] / c4[btc_col].shift(16)), axis = 0)
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
    feats_4h["xrank_past_alpha_96"] = _xrank(alpha_t.rolling(96, min_periods = 48).sum())
    feats_4h["xrank_past_alpha_384"] = _xrank(alpha_t.rolling(384, min_periods = 192).sum())
    feats_1h = {}
    indicators_1h = compute_hierarchical_indicators(px_15m, "1h")
    phase_shift_1h = HORIZON_CFG["1h"]["phase_shift"]
    blocks_1h = sample_blocks_hierarchical(indicators_1h, phase_shift_1h, n_blocks = 4)
    feats_1h = aggregate_blocks_mean_slope(blocks_1h)
    bp = px_15m["taker_buy_base_volume"] / (v + 1e-9)
    vol_mean = v.rolling(48, min_periods = 8).mean().clip(lower = 1e-9)
    feats_15m = {
        "ret_1": normalise(r.fillna(0), 120),
        "ret_4": normalise(np.log(c / c.shift(4)).fillna(0), 120),
        "vol_surge": (v / vol_mean).clip(0, 10).fillna(1.0),
        "buy_pressure": bp.fillna(0.5),
        "order_imbalance": (bp * 2 - 1).fillna(0.0),
        "hl_spread": normalise(((h - l) / (c + 1e-9)).fillna(0), 120),
    }
    last = c.index[-1]
    f4h_vec = np.stack([feats_4h[k].loc[last].values for k in sorted(feats_4h.keys())], axis = -1)
    f1h_vec = np.stack([feats_1h[k].loc[last].values for k in sorted(feats_1h.keys())], axis = -1)
    f15m_vec = np.stack([feats_15m[k].loc[last].values for k in sorted(feats_15m.keys())], axis = -1)
    te_4h = compute_time_encoding(np.array([last.value]), "4h")[0]
    te_1h = compute_time_encoding(np.array([last.value]), "1h")[0]
    te_15m = compute_time_encoding(np.array([last.value]), "15m")[0]
    seq_4h = 72
    lookback_1h = 48
    lookback_15m = 96
    f4h_t = torch.from_numpy(np.nan_to_num(f4h_vec, nan = 0.0)).float().unsqueeze(0).unsqueeze(0).expand(-1, seq_4h, -1, -1)
    te4h_t = torch.from_numpy(te_4h).float().unsqueeze(0).unsqueeze(0).expand(-1, seq_4h, -1)
    f1h_t = torch.from_numpy(np.nan_to_num(f1h_vec, nan = 0.0)).float().unsqueeze(0).unsqueeze(0).expand(-1, lookback_1h, -1, -1)
    te1h_t = torch.from_numpy(te_1h).float().unsqueeze(0).unsqueeze(0).expand(-1, lookback_1h, -1)
    f15m_t = torch.from_numpy(np.nan_to_num(f15m_vec, nan = 0.0)).float().unsqueeze(0).unsqueeze(0).expand(-1, lookback_15m, -1, -1)
    te15m_t = torch.from_numpy(te_15m).float().unsqueeze(0).unsqueeze(0).expand(-1, lookback_15m, -1)
    m1_t = torch.from_numpy(m1_output).float().unsqueeze(0).unsqueeze(0)
    return {
        "features_4h": f4h_t, "time_enc_4h": te4h_t,
        "features_1h": f1h_t, "time_enc_1h": te1h_t,
        "features_15m": f15m_t, "time_enc_15m": te15m_t,
        "model1_outputs": m1_t,
    }


class PaperTrader:

    def __init__(self, capital = INITIAL_CAPITAL, symbols = SYMBOLS):
        self.symbols = symbols
        self.N = len(symbols)
        self.capital = capital
        self.initial_capital = capital
        self.positions = np.zeros(self.N)
        self.cash = capital
        self.prices = np.zeros(self.N)
        self.nav_history = []
        self.trade_history = []
        self.regime_history = []
        self.model2 = None
        self.model2_cfg = None

    def load_models(self):
        import torch
        from model2_train import load_ckpt, DEFAULT_CFG
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", "model2_best.pt")
        self.model2, ckpt = load_ckpt(ckpt_path, "cpu")
        self.model2.eval()
        self.model2_cfg = ckpt.get("cfg", dict(DEFAULT_CFG))
        print("Models loaded")

    def update_prices(self, px_15m):
        c = px_15m["close"]
        for i, sym in enumerate(self.symbols):
            if sym in c.columns:
                self.prices[i] = float(c[sym].iloc[-1])

    def compute_nav(self):
        position_value = (self.positions * self.prices).sum()
        return self.cash + position_value

    def execute_trades(self, target_weights, gate):
        nav = self.compute_nav()
        target_positions = np.zeros(self.N)
        for i in range(self.N):
            if self.prices[i] > 0:
                target_value = nav * target_weights[i] * gate
                target_positions[i] = target_value / self.prices[i]
        trades = target_positions - self.positions
        total_cost = 0.0
        executed = []
        for i in range(self.N):
            if abs(trades[i]) < 1e-8:
                continue
            trade_value = abs(trades[i] * self.prices[i])
            tier = TIER_MAP.get(self.symbols[i], "small")
            slippage = SLIPPAGE_BY_TIER[tier]
            cost = trade_value * (TAKER_FEE + slippage)
            total_cost += cost
            self.positions[i] = target_positions[i]
            executed.append({
                "symbol": self.symbols[i],
                "side": "BUY" if trades[i] > 0 else "SELL",
                "qty": float(abs(trades[i])),
                "price": float(self.prices[i]),
                "value": float(trade_value),
                "fee": float(trade_value * TAKER_FEE),
                "slippage": float(trade_value * slippage),
                "cost": float(cost),
                "tier": tier,
            })
        position_value = (self.positions * self.prices).sum()
        self.cash = nav - position_value - total_cost
        return executed, total_cost

    def step(self):
        import torch
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"Paper Trade Step: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'=' * 60}")
        all_data = fetch_all_assets(self.symbols)
        if len(all_data) < len(self.symbols) * 0.8:
            print("ERROR: too many assets failed to fetch")
            return
        px_15m = build_live_price_matrices(all_data)
        self.update_prices(px_15m)
        nav_before = self.compute_nav()
        print(f"NAV before: ${nav_before:.2f}")
        print("Computing Model 1 features...")
        X_m1 = compute_model1_features(px_15m)
        m1_output = run_model1(X_m1)
        regime_names = ["direction_prob", "phase_prob", "dir_confidence",
                        "phase_confidence", "transition_intensity"]
        print(f"Regime: " + "  ".join(f"{n}={m1_output[i]:.3f}" for i, n in enumerate(regime_names)))
        dir_p = m1_output[0]
        phase_p = m1_output[1]
        bull = (1 - dir_p) * (1 - phase_p)
        bear = dir_p * (1 - phase_p)
        acc = (1 - dir_p) * phase_p
        dist = dir_p * phase_p
        regime_class = ["BULL", "BEAR", "ACCUMULATING", "DISTRIBUTING"][
            np.argmax([bull, bear, acc, dist])]
        print(f"Regime class: {regime_class} (bull={bull:.3f} bear={bear:.3f} acc={acc:.3f} dist={dist:.3f})")
        print("Computing Model 2 features...")
        m2_input = compute_model2_input(px_15m, m1_output)
        with torch.no_grad():
            out = self.model2(
                f4h = m2_input["features_4h"],
                te4h = m2_input["time_enc_4h"],
                f15m = m2_input["features_15m"],
                te15m = m2_input["time_enc_15m"],
                f1h = m2_input["features_1h"],
                te1h = m2_input["time_enc_1h"],
                m1_out = m2_input["model1_outputs"].squeeze(1))
        pred_ret = out["pred_ret"][0].numpy()
        log_var = out["log_var"][0].numpy()
        print(f"Predicted returns (top 5): " + "  ".join(
            f"{self.symbols[i].replace('USDT','')}:{pred_ret[i]:+.4f}"
            for i in np.argsort(pred_ret)[::-1][:5]))
        print(f"Predicted returns (bottom 5): " + "  ".join(
            f"{self.symbols[i].replace('USDT','')}:{pred_ret[i]:+.4f}"
            for i in np.argsort(pred_ret)[:5]))
        from model2_layers import PortfolioConstructor
        constructor = PortfolioConstructor(cost_rate = 0.0015)
        pred_t = torch.from_numpy(pred_ret).float().unsqueeze(0)
        logvar_t = torch.from_numpy(log_var).float().unsqueeze(0)
        regime_t = torch.from_numpy(m1_output).float().unsqueeze(0)
        prev_w = torch.from_numpy(self.positions * self.prices / max(self.compute_nav(), 1e-9)).float().unsqueeze(0)
        target_w, gate_val = constructor.construct(pred_t, logvar_t, regime_t, prev_w)
        target = target_w[0].numpy()
        gate = float(gate_val[0].item())
        print(f"Gate: {gate:.3f}  Gross target: {target.sum():.3f}")
        print(f"Top targets: " + "  ".join(
            f"{self.symbols[i].replace('USDT','')}:{target[i]:.3f}"
            for i in np.argsort(target)[::-1][:5]))
        executed, total_cost = self.execute_trades(target, gate)
        nav_after = self.compute_nav()
        print(f"Executed {len(executed)} trades, cost ${total_cost:.2f}")
        print(f"NAV after: ${nav_after:.2f}  Return: {(nav_after / nav_before - 1) * 100:+.3f}%")
        print(f"Total return: {(nav_after / self.initial_capital - 1) * 100:+.3f}%")
        gross = (self.positions * self.prices).sum() / max(nav_after, 1e-9)
        print(f"Gross exposure: {gross:.1%}  Cash: ${self.cash:.2f}")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nav": float(nav_after),
            "cash": float(self.cash),
            "gross_exposure": float(gross),
            "gate": gate,
            "regime_class": regime_class,
            "regime": {n: float(m1_output[i]) for i, n in enumerate(regime_names)},
            "predicted_returns": {self.symbols[i]: float(pred_ret[i]) for i in range(self.N)},
            "positions": {self.symbols[i]: float(self.positions[i]) for i in range(self.N)},
            "prices": {self.symbols[i]: float(self.prices[i]) for i in range(self.N)},
            "weights": {self.symbols[i]: float(target[i]) for i in range(self.N)},
            "trades": executed,
            "cost": float(total_cost),
            "elapsed_s": time.time() - t0,
        }
        self.nav_history.append(record)
        self.regime_history.append({
            "timestamp": record["timestamp"],
            "regime_class": regime_class,
            "direction_prob": float(dir_p),
            "phase_prob": float(phase_p),
        })
        log_path = os.path.join(LOG_DIR, "trade_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def run(self, n_steps = None):
        self.load_models()
        step = 0
        while n_steps is None or step < n_steps:
            try:
                self.step()
            except Exception as e:
                print(f"Step error: {e}")
                import traceback
                traceback.print_exc()
            step += 1
            if n_steps is not None and step >= n_steps:
                break
            next_update = UPDATE_INTERVAL_S
            print(f"\nSleeping {next_update // 3600}h until next update...")
            time.sleep(next_update)

    def summary(self):
        if not self.nav_history:
            print("No trades yet")
            return
        navs = [r["nav"] for r in self.nav_history]
        print(f"\n{'=' * 60}")
        print(f"PAPER TRADING SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Steps: {len(navs)}")
        print(f"  Initial: ${self.initial_capital:.2f}")
        print(f"  Current: ${navs[-1]:.2f}")
        print(f"  Return: {(navs[-1] / self.initial_capital - 1) * 100:+.2f}%")
        print(f"  Max NAV: ${max(navs):.2f}")
        print(f"  Min NAV: ${min(navs):.2f}")
        total_cost = sum(r["cost"] for r in self.nav_history)
        print(f"  Total costs: ${total_cost:.2f}")
        regimes = [r["regime_class"] for r in self.nav_history]
        for rc in ["BULL", "BEAR", "ACCUMULATING", "DISTRIBUTING"]:
            pct = regimes.count(rc) / len(regimes) * 100
            print(f"  {rc}: {pct:.1f}%")


if __name__ == "__main__":
    trader = PaperTrader(capital = INITIAL_CAPITAL)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 1, help = "number of steps (0=infinite)")
    parser.add_argument("--capital", type = float, default = INITIAL_CAPITAL)
    args = parser.parse_args()
    trader.capital = args.capital
    trader.cash = args.capital
    trader.initial_capital = args.capital
    n = args.steps if args.steps > 0 else None
    trader.run(n_steps = n)
    trader.summary()
