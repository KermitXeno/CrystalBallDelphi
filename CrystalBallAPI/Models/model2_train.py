import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model2_layers import Model2
from dp_download import BASE_DIR
from ds_struc_wstg import Model2Dataset, ASSETS, BTC_IDX

DEFAULT_CFG = {
    "N": 20,
    "F_1h": None,
    "F_15m": None,
    "F_30m": None,
    "D_time_1h": None,
    "D_time_15m": None,
    "D_time_30m": None,
    "seq_len_1h": 72,
    "seq_k": 8,
    "lookback_15m": 96,
    "lookback_30m": 48,
    "train_stride": 4,
    "d_regime": 6,
    "d_model": 32,
    "d_lstm": 32,
    "d_cross": 48,
    "n_cross_heads": 4,
    "t_recent": 4,
    "dropout": 0.12,
    "embed_drop": 0.5,
    "band_sharpness": 30.0,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 40,
    "grad_clip": 1.0,
    "warmup_epochs": 4,
    "warmup_start_factor": 0.1,
    "scheduler_eta_min": 3e-5,
    "patience": 15,
    "fee_rate": 0.001,
    "slippage_rate": 0.0005,
    "btc_idx": BTC_IDX,
    "conc_coef": 0.25,
    "max_w_coef": 0.5,
    "dd_coef": 0.08,
    "logit_reg_coef": 0.005,
    "max_weight": 0.4,
    "max_hhi": 0.3,
    "band_reg_coef": 0.02,
    "band_target": 0.05,
    "trade_q_coef": 5.0,
    "trade_q_min_delta": 0.005,
    "trade_q_fwd_horizon": 4,
    "trade_q_fwd_decay": 0.7,
    "trade_q_loss_aversion": 1.5,
    "regime_gross_coef": 0.5,
    "bull_gross_target": 0.55,
    "bear_gross_target": 0.10,
    "neutral_gross_target": 0.30,
    "inactivity_coef": 10.0,
    "min_turnover": 0.05,
    "cost_ramp_start_epoch": 5,
    "cost_ramp_end_epoch": 25,
    "cost_start_fraction": 0.15,
    "ir_var_floor": 1e-4,
    "ir_clip": 50.0,
    "skip_nan_batches": True,
    "log_every_n_batches": 25,
}

def _flatten_bk(t):
    return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])

def compute_cost_rate(epoch, cfg):
    full = cfg["fee_rate"] + cfg["slippage_rate"]
    s, e, sf = cfg["cost_ramp_start_epoch"], cfg["cost_ramp_end_epoch"], cfg["cost_start_fraction"]
    if epoch < s:
        frac = sf
    elif epoch >= e:
        frac = 1.0
    else:
        frac = sf + (1.0 - sf) * (epoch - s) / max(e - s, 1)
    return full * frac, frac

def davis_norman_step(prev, target, band, gate, sharpness, max_weight):
    aim = gate.unsqueeze(-1) * target
    deviation = prev - aim
    abs_dev = deviation.abs()
    excess = abs_dev - band
    update_strength = torch.sigmoid(sharpness * excess)
    snap_target = aim + torch.sign(deviation) * band
    snap_target = snap_target.clamp(min = 0.0, max = max_weight)
    new_pos = prev + update_strength * (snap_target - prev)
    return new_pos.clamp(min = 0.0, max = max_weight)

def portfolio_loss(target, gate, band, logits, ret_next, dd_next, m1_out, cfg, cost_rate = None):
    if cost_rate is None:
        cost_rate = cfg["fee_rate"] + cfg["slippage_rate"]
    B, K, N = target.shape
    sharpness = cfg["band_sharpness"]
    max_w = cfg["max_weight"]
    pos_list = []
    prev = target.new_zeros(B, N)
    for t in range(K):
        pos_t = davis_norman_step(prev, target[:, t], band[:, t], gate[:, t], sharpness, max_w)
        pos_list.append(pos_t)
        prev = pos_t
    positions = torch.stack(pos_list, dim = 1)
    prev_pos = torch.cat([target.new_zeros(B, 1, N), positions[:, :-1]], dim = 1)
    delta_pos = positions - prev_pos
    turnover = delta_pos.abs().sum(dim = -1)
    gross_ret = (positions * ret_next).sum(dim = -1)
    port_ret = gross_ret - cost_rate * turnover
    bench_weights = cfg["bench_weights"]
    bench_ret = (ret_next * bench_weights).sum(dim = -1)
    excess = port_ret - bench_ret
    mean_ex = excess.mean()
    std_ex = (excess.var(unbiased = False) + cfg["ir_var_floor"]).sqrt()
    ir = (mean_ex / std_ex).clamp(-cfg["ir_clip"], cfg["ir_clip"])
    btc_ret = ret_next[..., cfg["btc_idx"]]
    trade_q_min = cfg["trade_q_min_delta"]
    traded_mask = (delta_pos.abs() > trade_q_min).float()
    fwd_horizon = cfg["trade_q_fwd_horizon"]
    fwd_returns = ret_next.clone()
    for h in range(1, fwd_horizon):
        shifted = torch.zeros_like(ret_next)
        if K - h > 0:
            shifted[:, :K - h] = ret_next[:, h:]
        fwd_returns = fwd_returns + shifted * (cfg["trade_q_fwd_decay"] ** h)
    is_buy = (delta_pos > 0).float()
    is_sell = (delta_pos < 0).float()
    asym_loss_w = cfg["trade_q_loss_aversion"]
    buy_pnl = (delta_pos * fwd_returns * traded_mask * is_buy).sum(dim = -1)
    sell_pnl_raw = (delta_pos * fwd_returns * traded_mask * is_sell).sum(dim = -1)
    sell_loss_avoided = F.relu(-sell_pnl_raw) * (asym_loss_w - 1.0)
    trade_pnl = buy_pnl + sell_pnl_raw + sell_loss_avoided
    trade_volume = (delta_pos.abs() * traded_mask).sum(dim = -1).clamp(min = 1e-6)
    trade_quality = (trade_pnl / trade_volume).mean()
    gross = positions.sum(dim = -1)
    hhi = (positions ** 2).sum(dim = -1)
    dd_term = (positions * dd_next).sum(dim = -1).mean()
    conc_term = F.relu(hhi - cfg["max_hhi"]).mean()
    maxw_term = F.relu(positions - cfg["max_weight"]).sum(dim = -1).mean()
    band_reg = (band - cfg["band_target"]).pow(2).mean()
    mean_turnover = turnover.mean()
    inactivity_pen = F.relu(cfg["min_turnover"] - mean_turnover)
    bull_target = cfg["bull_gross_target"]
    bear_target = cfg["bear_gross_target"]
    bull_prob = m1_out[..., 0]
    bear_prob = m1_out[..., 1]
    sideways_prob = (1.0 - bull_prob - bear_prob).clamp(min = 0.0, max = 1.0)
    desired_gross = (bull_prob * bull_target
                     + bear_prob * bear_target
                     + sideways_prob * cfg["neutral_gross_target"])
    regime_gross_pen = (gross - desired_gross).pow(2).mean()
    loss = (-ir
            - cfg["trade_q_coef"] * trade_quality
            - cfg["dd_coef"] * dd_term
            + cfg["conc_coef"] * conc_term
            + cfg["max_w_coef"] * maxw_term
            + cfg["logit_reg_coef"] * logits.pow(2).mean()
            + cfg["band_reg_coef"] * band_reg
            + cfg["inactivity_coef"] * inactivity_pen
            + cfg["regime_gross_coef"] * regime_gross_pen)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss ir={ir.item()}")
    per_asset = positions.detach().float().mean(dim = (0, 1)).cpu().numpy()
    n_trades = traded_mask.sum().item() / max(B, 1)
    n_buys = (is_buy * traded_mask).sum().item() / max(B, 1)
    n_sells = (is_sell * traded_mask).sum().item() / max(B, 1)
    return loss, {
        "ir": ir.item(), "trade_q": trade_quality.item(),
        "buy_pnl": buy_pnl.mean().item(), "sell_pnl": sell_pnl_raw.mean().item(),
        "net_ret": port_ret.mean().item(),
        "btc_ret": btc_ret.mean().item(), "bench_ret": bench_ret.mean().item(),
        "excess_ret": mean_ex.item(),
        "excess_vs_btc": (port_ret - btc_ret).mean().item(),
        "cost": (cost_rate * turnover).mean().item(), "turnover": turnover.mean().item(),
        "gross": gross.mean().item(), "cash": (1.0 - gross).mean().item(),
        "desired_gross": desired_gross.mean().item(),
        "gate": gate.mean().item(), "band": band.mean().item(),
        "n_trades": n_trades, "n_buys": n_buys, "n_sells": n_sells,
        "dd": dd_term.item(),
        "max_w": positions.max().item(), "per_asset": per_asset,
    }

def _forward_batch(model, batch, device):
    for k in batch:
        batch[k] = batch[k].to(device, non_blocking = True)
    B, K = batch["features_1h"].shape[:2]
    out = model(
        f1h = _flatten_bk(batch["features_1h"]),
        te1h = _flatten_bk(batch["time_enc_1h"]),
        f15m = _flatten_bk(batch["features_15m"]),
        te15m = _flatten_bk(batch["time_enc_15m"]),
        f30m = _flatten_bk(batch["features_30m"]),
        te30m = _flatten_bk(batch["time_enc_30m"]),
        m1_out = _flatten_bk(batch["model1_outputs"]))
    return (out["target"].view(B, K, -1), out["gate"].view(B, K), out["band"].view(B, K, -1),
            out["logits"].view(B, K, -1), batch["targets"][..., 0], batch["targets"][..., 2],
            batch["model1_outputs"])

def _acc(tot, new):
    for k, v in new.items():
        if k == "per_asset":
            tot[k] = tot.get(k, np.zeros_like(v)) + v
        else:
            tot[k] = tot.get(k, 0.0) + v

def _avg(tot, n):
    d = max(n, 1)
    return {k: v / d for k, v in tot.items()}

def train_epoch(model, loader, optimizer, device, cfg, epoch = None):
    model.train()
    tot_loss, tot_stats, n, n_skip = 0.0, {}, 0, 0
    cost_rate, cost_frac = compute_cost_rate(epoch, cfg)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none = True)
        try:
            tgt, gate, band, logits, ret_n, dd_n, m1 = _forward_batch(model, batch, device)
            loss, stats = portfolio_loss(tgt, gate, band, logits, ret_n, dd_n, m1, cfg, cost_rate)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if not torch.isfinite(gn):
                raise RuntimeError("non-finite grad")
            optimizer.step()
        except RuntimeError as e:
            if cfg["skip_nan_batches"] and "non-finite" in str(e):
                n_skip += 1
                optimizer.zero_grad(set_to_none = True)
                if n_skip > max(20, len(loader) // 4):
                    raise
                continue
            raise
        tot_loss += loss.item()
        _acc(tot_stats, stats)
        n += 1
        lev = cfg["log_every_n_batches"]
        if lev > 0 and (bi + 1) % lev == 0:
            el = time.time() - t0
            its = (bi + 1) / max(el, 1e-6)
            print(f"  Ep {epoch:03d} step {bi+1:03d}/{len(loader)} "
                  f"loss {tot_loss/max(n,1):.4f} ir={tot_stats['ir']/max(n,1):+.3f} "
                  f"tq={tot_stats['trade_q']/max(n,1):+.4f} grs={tot_stats['gross']/max(n,1):+.3f} "
                  f"gate={tot_stats['gate']/max(n,1):.3f} band={tot_stats['band']/max(n,1):.3f} "
                  f"cost={cost_rate*1e4:.1f}bps({cost_frac*100:.0f}%) {its:.1f}it/s", flush = True)
    avg = _avg(tot_stats, n)
    avg["cost_rate"] = cost_rate
    avg["cost_frac"] = cost_frac
    return tot_loss / max(n, 1), avg

@torch.no_grad()
def eval_epoch(model, loader, device, cfg):
    model.eval()
    tot_loss, tot_stats, n = 0.0, {}, 0
    full_cost = cfg["fee_rate"] + cfg["slippage_rate"]
    for batch in loader:
        tgt, gate, band, logits, ret_n, dd_n, m1 = _forward_batch(model, batch, device)
        loss, stats = portfolio_loss(tgt, gate, band, logits, ret_n, dd_n, m1, cfg, full_cost)
        tot_loss += loss.item()
        _acc(tot_stats, stats)
        n += 1
    return tot_loss / max(n, 1), _avg(tot_stats, n)

def _fmt(pa, assets):
    return " ".join(f"{a.replace('USDT','')}:{w:.3f}" for a, w in zip(assets, pa))

def _check_m1(m1):
    nz = np.abs(m1).sum(axis = -1) > 0
    c = nz.sum() / max(len(m1), 1)
    if nz.sum() == 0:
        print("WARNING: model1_outputs all zeros")
        return False
    print(f"m1 populated {c:.1%} range=[{m1[nz].min():+.4f},{m1[nz].max():+.4f}]")
    return True

def save_ckpt(path, model, opt, sched, epoch, best_vl, cfg):
    cfg_save = {k: v for k, v in cfg.items()}
    if "bench_weights" in cfg_save and torch.is_tensor(cfg_save["bench_weights"]):
        cfg_save["bench_weights"] = cfg_save["bench_weights"].cpu().tolist()
    torch.save({
        "cfg": cfg_save, "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(), "epoch": epoch, "best_val_loss": best_vl,
    }, path)

def _mk_model(cfg, device, asset_vol):
    model = Model2(
        F_1h = cfg["F_1h"], F_15m = cfg["F_15m"], F_30m = cfg["F_30m"],
        D_time_1h = cfg["D_time_1h"], D_time_15m = cfg["D_time_15m"], D_time_30m = cfg["D_time_30m"],
        N_assets = cfg["N"], d_regime = cfg["d_regime"],
        d_model = cfg["d_model"], d_lstm = cfg["d_lstm"], d_cross = cfg["d_cross"],
        n_cross_heads = cfg["n_cross_heads"], dropout = cfg["dropout"],
        embed_drop = cfg["embed_drop"], band_sharpness = cfg["band_sharpness"],
        t_recent = cfg["t_recent"]).to(device)
    model.alloc.set_inv_vol_prior(asset_vol.to(device))
    return model

def load_ckpt(path, device = "cpu"):
    ckpt = torch.load(path, map_location = device, weights_only = False)
    c = ckpt["cfg"]
    if "bench_weights" in c and isinstance(c["bench_weights"], list):
        c["bench_weights"] = torch.tensor(c["bench_weights"], device = device, dtype = torch.float32)
    model = Model2(
        F_1h = c["F_1h"], F_15m = c["F_15m"], F_30m = c["F_30m"],
        D_time_1h = c["D_time_1h"], D_time_15m = c["D_time_15m"], D_time_30m = c["D_time_30m"],
        N_assets = c["N"], d_regime = c.get("d_regime", 6),
        d_model = c.get("d_model", 32), d_lstm = c.get("d_lstm", 32), d_cross = c.get("d_cross", 48),
        n_cross_heads = c.get("n_cross_heads", 4), dropout = c.get("dropout", 0.12),
        embed_drop = c.get("embed_drop", 0.5), band_sharpness = c.get("band_sharpness", 30.0),
        t_recent = c.get("t_recent", 4)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt

def _mk_optim(model, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr = cfg["lr"], weight_decay = cfg["weight_decay"])
    ct = cfg["epochs"] - cfg["warmup_epochs"]
    w = torch.optim.lr_scheduler.LinearLR(opt, start_factor = cfg["warmup_start_factor"],
                                          end_factor = 1.0, total_iters = cfg["warmup_epochs"])
    c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = max(ct, 1), eta_min = cfg["scheduler_eta_min"])
    s = torch.optim.lr_scheduler.SequentialLR(opt, schedulers = [w, c], milestones = [cfg["warmup_epochs"]])
    return opt, s

def _ep_print(ep, tr, va, model, tl, vl, el, cf):
    temp = float(model.alloc.log_temp.exp().item())
    print(f"  Ep {ep:03d} "
          f"tr ir={tr['ir']:+.3f} tq={tr['trade_q']:+.4f} bp={tr['buy_pnl']:+.4f} "
          f"sp={tr['sell_pnl']:+.4f} grs={tr['gross']:.3f}/{tr['desired_gross']:.3f} "
          f"gate={tr['gate']:.3f} band={tr['band']:.3f} "
          f"nb={tr['n_buys']:.1f} ns={tr['n_sells']:.1f} "
          f"va ir={va['ir']:+.3f} tq={va['trade_q']:+.4f} grs={va['gross']:.3f} "
          f"temp={temp:.3f} loss={tl:.4f}/{vl:.4f} "
          f"cost={tr.get('cost_rate',0)*1e4:.1f}bps({cf*100:.0f}%) {el:.0f}s", flush = True)
    print(f"    tr {_fmt(tr['per_asset'], ASSETS)}", flush = True)
    print(f"    va {_fmt(va['per_asset'], ASSETS)}", flush = True)

def build_trade_decision(out, prev_position, cfg, deadband = 0.005, min_notional = 0.005):
    target = out["target"][0].cpu().numpy()
    gate = float(out["gate"][0].cpu().numpy())
    band = out["band"][0].cpu().numpy()
    sharpness = cfg.get("band_sharpness", 30.0)
    max_w = cfg.get("max_weight", 0.4)
    aim = gate * target
    deviation = prev_position - aim
    abs_dev = np.abs(deviation)
    snap_target = aim + np.sign(deviation) * band
    snap_target = np.clip(snap_target, 0.0, max_w)
    excess = abs_dev - band
    strength = 1.0 / (1.0 + np.exp(-sharpness * excess))
    new_pos = prev_position + strength * (snap_target - prev_position)
    new_pos = np.clip(new_pos, 0.0, max_w)
    delta = new_pos - prev_position
    dc = np.where(np.abs(delta) < deadband, 0.0, delta)
    trades = [{"asset_idx": int(i), "side": "buy" if d > 0 else "sell",
               "weight_delta": float(d), "target_weight": float(new_pos[i])}
              for i, d in enumerate(dc) if abs(d) >= min_notional]
    return {"target_position": new_pos.tolist(), "trades": trades, "gross": float(new_pos.sum()),
            "cash_weight": 1.0 - float(new_pos.sum()), "gate": gate, "band": band.tolist(),
            "action": "hold" if not trades else "rebalance"}

def _mk_dataset(npz_path, cfg, split):
    return Model2Dataset(npz_path, seq_len_1h = cfg["seq_len_1h"], seq_k = cfg["seq_k"],
                         split = split, stride = cfg["train_stride"],
                         lookback_15m = cfg["lookback_15m"], lookback_30m = cfg["lookback_30m"])

def _populate_cfg(ds, cfg):
    cfg["N"] = ds.N
    cfg["F_1h"] = ds.F_1h
    cfg["F_15m"] = ds.F_15m
    cfg["F_30m"] = ds.F_30m
    cfg["D_time_1h"] = ds.D_time_1h
    cfg["D_time_15m"] = ds.D_time_15m
    cfg["D_time_30m"] = ds.D_time_30m
    cfg["btc_idx"] = ds.assets.index("BTCUSDT")

def train(cfg = None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    npz_path = os.path.join(BASE_DIR, "model2_dataset.npz")
    tr_ds = _mk_dataset(npz_path, cfg, "train")
    va_ds = _mk_dataset(npz_path, cfg, "val")
    te_ds = _mk_dataset(npz_path, cfg, "test")
    print(f"Samples: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}")
    _populate_cfg(tr_ds, cfg)
    _check_m1(tr_ds.model1_outputs)
    asset_vol = torch.from_numpy(tr_ds.asset_vol).float()
    inv_vol = 1.0 / asset_vol.clamp(min = 1e-6)
    bench_w = inv_vol / inv_vol.sum()
    cfg["bench_weights"] = bench_w.to(device)
    bench_str = " ".join(f"{a.replace('USDT','')}:{w:.3f}" for a, w in zip(ASSETS, bench_w.numpy()))
    print(f"Benchmark (inv-vol): {bench_str}")
    model = _mk_model(cfg, device, asset_vol)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_p:,} samples:params={len(tr_ds)/max(n_p,1):.2f}:1")
    print(f"F_1h={cfg['F_1h']} F_15m={cfg['F_15m']} F_30m={cfg['F_30m']}")
    print(f"Model: d_model={cfg['d_model']} d_lstm={cfg['d_lstm']} d_cross={cfg['d_cross']} "
          f"heads={cfg['n_cross_heads']} dropout={cfg['dropout']}")
    print(f"Davis-Norman: band_sharpness={cfg['band_sharpness']} band_target={cfg['band_target']} "
          f"band_reg_coef={cfg['band_reg_coef']} max_weight={cfg['max_weight']}")
    print(f"Trade quality: coef={cfg['trade_q_coef']} min_delta={cfg['trade_q_min_delta']} "
          f"fwd_horizon={cfg['trade_q_fwd_horizon']} fwd_decay={cfg['trade_q_fwd_decay']} "
          f"loss_aversion={cfg['trade_q_loss_aversion']}")
    print(f"Regime sizing: bull={cfg['bull_gross_target']} bear={cfg['bear_gross_target']} "
          f"neutral={cfg['neutral_gross_target']} coef={cfg['regime_gross_coef']}")
    opt, sched = _mk_optim(model, cfg)
    pin = device.type == "cuda"
    tr_ld = DataLoader(tr_ds, batch_size = cfg["batch_size"], shuffle = True,
                       num_workers = 0, pin_memory = pin, drop_last = True)
    va_ld = DataLoader(va_ds, batch_size = cfg["batch_size"] * 2, shuffle = False,
                       num_workers = 0, pin_memory = pin)
    te_ld = DataLoader(te_ds, batch_size = cfg["batch_size"] * 2, shuffle = False,
                       num_workers = 0, pin_memory = pin)
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok = True)
    bp = os.path.join(ckpt_dir, "model2_best.pt")
    best_vl, pat = float("inf"), 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tl, ts = train_epoch(model, tr_ld, opt, device, cfg, epoch = ep)
        vl, vs = eval_epoch(model, va_ld, device, cfg)
        sched.step()
        _ep_print(ep, ts, vs, model, tl, vl, time.time() - t0, ts.get("cost_frac", 0))
        if vl < best_vl:
            best_vl, pat = vl, 0
            save_ckpt(bp, model, opt, sched, ep, best_vl, cfg)
            print(f"  => saved best val_loss={best_vl:.4f}")
        else:
            pat += 1
            if pat >= cfg["patience"]:
                print(f"  Early stop ep {ep}")
                break
    print("\nLoading best for test")
    model, _ = load_ckpt(bp, device)
    tl, ts = eval_epoch(model, te_ld, device, cfg)
    print(f"Test loss={tl:.4f} ir={ts['ir']:+.3f} tq={ts['trade_q']:+.4f} "
          f"net_ret={ts['net_ret']:+.5f} bench_ret={ts['bench_ret']:+.5f} btc_ret={ts['btc_ret']:+.5f} "
          f"excess_vs_bench={ts['excess_ret']:+.5f} excess_vs_btc={ts['excess_vs_btc']:+.5f} "
          f"grs={ts['gross']:.3f} gate={ts['gate']:.3f} band={ts['band']:.3f} "
          f"ntr={ts['n_trades']:.1f} max_w={ts['max_w']:.3f}")
    print(f"Test {_fmt(ts['per_asset'], ASSETS)}")
    return model, ts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    train()
