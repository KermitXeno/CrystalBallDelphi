import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model2_layers import Model2, PortfolioConstructor
from ds_struc_wstg import Model2Dataset, ASSETS, BTC_IDX, BASE_DIR

DEFAULT_CFG = {
    "N": 20,
    "F_4h": None,
    "F_15m": None,
    "F_1h": None,
    "D_time_4h": None,
    "D_time_15m": None,
    "D_time_1h": None,
    "seq_len_4h": 72,
    "seq_k": 8,
    "lookback_15m": 96,
    "lookback_1h": 48,
    "train_stride": 1,
    "d_regime": 5,
    "d_model": 32,
    "d_lstm": 32,
    "d_cross": 48,
    "n_cross_heads": 4,
    "t_recent": 4,
    "dropout": 0.12,
    "embed_drop": 0.5,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 40,
    "grad_clip": 1.0,
    "warmup_epochs": 3,
    "warmup_start_factor": 0.1,
    "scheduler_eta_min": 3e-5,
    "patience": 15,
    "temporal_ic_coef": 1.5,
    "xs_ic_coef": 0.3,
    "nll_coef": 0.5,
    "directional_coef": 0.3,
    "cost_rate": 0.0015,
    "btc_idx": BTC_IDX,
    "skip_nan_batches": True,
    "log_every_n_batches": 25,
}


def temporal_ic_loss(pred, actual):
    pred_dm = pred - pred.mean(dim = 1, keepdim = True)
    actual_dm = actual - actual.mean(dim = 1, keepdim = True)
    cov = (pred_dm * actual_dm).sum(dim = 1)
    std_p = (pred_dm.pow(2).sum(dim = 1) + 1e-8).sqrt()
    std_a = (actual_dm.pow(2).sum(dim = 1) + 1e-8).sqrt()
    ic_per_asset = cov / (std_p * std_a + 1e-8)
    return -ic_per_asset.mean(), ic_per_asset.mean().item()


def xs_ic_loss(pred, actual):
    pred_dm = pred - pred.mean(dim = -1, keepdim = True)
    actual_dm = actual - actual.mean(dim = -1, keepdim = True)
    cov = (pred_dm * actual_dm).sum(dim = -1)
    std_p = (pred_dm.pow(2).sum(dim = -1) + 1e-8).sqrt()
    std_a = (actual_dm.pow(2).sum(dim = -1) + 1e-8).sqrt()
    ic = cov / (std_p * std_a + 1e-8)
    return -ic.mean(), ic.mean().item()


def gaussian_nll_loss(pred, actual, log_var):
    var = log_var.exp().clamp(min = 1e-6, max = 10.0)
    nll = 0.5 * (log_var + (pred - actual).pow(2) / var)
    return nll.mean()


def directional_loss(pred, actual):
    correct = (pred.sign() == actual.sign()).float()
    confidence = pred.abs()
    wrong_confident = (1.0 - correct) * confidence
    return wrong_confident.mean()


def compute_prediction_loss(pred_ret, log_var, actual_ret, cfg):
    t_ic_l, t_ic_val = temporal_ic_loss(pred_ret, actual_ret)
    xs_ic_l, xs_ic_val = xs_ic_loss(
        pred_ret.reshape(-1, pred_ret.shape[-1]),
        actual_ret.reshape(-1, actual_ret.shape[-1]))
    nll_l = gaussian_nll_loss(
        pred_ret.reshape(-1, pred_ret.shape[-1]),
        actual_ret.reshape(-1, actual_ret.shape[-1]),
        log_var.reshape(-1, log_var.shape[-1]))
    dir_l = directional_loss(
        pred_ret.reshape(-1, pred_ret.shape[-1]),
        actual_ret.reshape(-1, actual_ret.shape[-1]))
    loss = (cfg["temporal_ic_coef"] * t_ic_l
            + cfg["xs_ic_coef"] * xs_ic_l
            + cfg["nll_coef"] * nll_l
            + cfg["directional_coef"] * dir_l)
    flat_pred = pred_ret.reshape(-1, pred_ret.shape[-1])
    flat_actual = actual_ret.reshape(-1, actual_ret.shape[-1])
    dir_acc = (flat_pred.sign() == flat_actual.sign()).float().mean().item()
    return loss, {
        "t_ic": t_ic_val,
        "xs_ic": xs_ic_val,
        "nll": nll_l.item(),
        "dir_acc": dir_acc,
        "pred_std": pred_ret.std().item(),
        "log_var_mean": log_var.mean().item(),
    }


def portfolio_metrics(pred_ret, log_var, actual_ret, regime_ctx, cfg):
    constructor = PortfolioConstructor(cost_rate = cfg["cost_rate"])
    with torch.no_grad():
        weights, gate = constructor.construct(pred_ret, log_var, regime_ctx)
        port_ret = (weights * actual_ret).sum(dim = -1)
        gross = weights.sum(dim = -1)
        btc_ret = actual_ret[..., cfg["btc_idx"]]
        inv_vol = torch.ones(cfg["N"], device = pred_ret.device) / cfg["N"]
        bench_ret = (actual_ret * inv_vol).sum(dim = -1)
        max_w = weights.max().item()
        concentration = (weights ** 2).sum(dim = -1).mean().item()
        mean_pred = pred_ret.mean(dim = 0).cpu().numpy()
        n_bullish = (pred_ret > 0).float().mean(dim = 0).cpu().numpy()
        pred_correct = ((pred_ret > 0) == (actual_ret > 0)).float().mean().item()
    return {
        "port_ret": port_ret.mean().item(),
        "bench_ret": bench_ret.mean().item(),
        "btc_ret": btc_ret.mean().item(),
        "excess_ret": (port_ret - bench_ret).mean().item(),
        "gross": gross.mean().item(),
        "gate_mean": gate.mean().item(),
        "gate_min": gate.min().item(),
        "gate_max": gate.max().item(),
        "max_w": max_w,
        "hhi": concentration,
        "weights": weights.detach().float().mean(dim = 0).cpu().numpy(),
        "mean_pred": mean_pred,
        "n_bullish": n_bullish,
        "pred_correct": pred_correct,
    }


def _flatten_bk(x):
    if x.dim() == 5:
        B, K, T, N, F = x.shape
        return x.reshape(B * K, T, N, F)
    if x.dim() == 4:
        B, K, T, D = x.shape
        return x.reshape(B * K, T, D)
    if x.dim() == 3:
        B, K, D = x.shape
        return x.reshape(B * K, D)
    return x


def _run_batch(model, batch, cfg):
    B = batch["features_4h"].shape[0]
    K = batch["features_4h"].shape[1]
    out = model(
        f4h = _flatten_bk(batch["features_4h"]),
        te4h = _flatten_bk(batch["time_enc_4h"]),
        f15m = _flatten_bk(batch["features_15m"]),
        te15m = _flatten_bk(batch["time_enc_15m"]),
        f1h = _flatten_bk(batch["features_1h"]),
        te1h = _flatten_bk(batch["time_enc_1h"]),
        m1_out = _flatten_bk(batch["model1_outputs"]))
    pred_ret = out["pred_ret"].view(B, K, -1)
    log_var = out["log_var"].view(B, K, -1)
    actual_ret = batch["targets"][..., 0]
    loss, stats = compute_prediction_loss(pred_ret, log_var, actual_ret, cfg)
    m1_flat = _flatten_bk(batch["model1_outputs"])
    pred_flat = pred_ret.reshape(B * K, -1)
    logvar_flat = log_var.reshape(B * K, -1)
    actual_flat = actual_ret.reshape(B * K, -1)
    port = portfolio_metrics(pred_flat, logvar_flat, actual_flat, m1_flat, cfg)
    stats.update(port)
    return loss, stats



def _fmt_top(arr, names, n = 5):
    top = np.argsort(arr)[::-1][:n]
    return " ".join(f"{names[i].replace('USDT','')}:{arr[i]:.1%}" for i in top if arr[i] > 0.001)


def _fmt_signals(mean_pred, names, n = 3):
    order = np.argsort(mean_pred)
    buys = [i for i in order[::-1][:n] if mean_pred[i] > 0]
    sells = [i for i in order[:n] if mean_pred[i] < 0]
    buy_str = " ".join(f"{names[i].replace('USDT','')} {mean_pred[i]:+.3%}" for i in buys)
    sell_str = " ".join(f"{names[i].replace('USDT','')} {mean_pred[i]:+.3%}" for i in sells)
    return buy_str or "none", sell_str or "none"


def train_epoch(model, loader, optimizer, cfg, epoch, device):
    model.train()
    tot_loss = 0.0
    tot_stats = {}
    n = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        try:
            loss, stats = _run_batch(model, batch, cfg)
        except RuntimeError as e:
            if cfg["skip_nan_batches"] and "non-finite" in str(e):
                continue
            raise
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        tot_loss += loss.item()
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                tot_stats[k] = tot_stats.get(k, 0) + v
        n += 1
        if (bi + 1) % cfg["log_every_n_batches"] == 0:
            its = (bi + 1) / (time.time() - t0)
            print(f"    step {bi+1:03d}/{len(loader)}"
                  f"  loss={tot_loss/max(n,1):.3f}"
                  f"  timing={tot_stats['t_ic']/max(n,1):+.3f}"
                  f"  direction={tot_stats['dir_acc']/max(n,1):.1%}"
                  f"  exposure={tot_stats['gross']/max(n,1):.1%}"
                  f"  {its:.1f}it/s", flush = True)
    avg = {k: v / max(n, 1) for k, v in tot_stats.items() if isinstance(v, (int, float))}
    return tot_loss / max(n, 1), avg


@torch.no_grad()
def eval_epoch(model, loader, cfg, device):
    model.eval()
    tot_loss = 0.0
    tot_stats = {}
    gate_mins, gate_maxs = [], []
    all_weights, all_preds, all_bullish = [], [], []
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        loss, stats = _run_batch(model, batch, cfg)
        if not torch.isfinite(loss):
            continue
        tot_loss += loss.item()
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                tot_stats[k] = tot_stats.get(k, 0) + v
            elif k == "weights":
                all_weights.append(v)
            elif k == "mean_pred":
                all_preds.append(v)
            elif k == "n_bullish":
                all_bullish.append(v)
        if "gate_min" in stats:
            gate_mins.append(stats["gate_min"])
        if "gate_max" in stats:
            gate_maxs.append(stats["gate_max"])
        n += 1
    avg = {k: v / max(n, 1) for k, v in tot_stats.items() if isinstance(v, (int, float))}
    avg["per_asset"] = np.mean(all_weights, axis = 0) if all_weights else np.zeros(cfg["N"])
    avg["mean_pred"] = np.mean(all_preds, axis = 0) if all_preds else np.zeros(cfg["N"])
    avg["n_bullish"] = np.mean(all_bullish, axis = 0) if all_bullish else np.zeros(cfg["N"])
    avg["gate_min"] = min(gate_mins) if gate_mins else 0.0
    avg["gate_max"] = max(gate_maxs) if gate_maxs else 1.0
    return tot_loss / max(n, 1), avg


def _ep_print(ep, tr, va, tl, vl, el):
    buys, sells = _fmt_signals(va.get("mean_pred", np.zeros(20)), ASSETS)
    n_bull = int((va.get("n_bullish", np.zeros(20)) > 0.5).sum())
    n_bear = 20 - n_bull
    g_mean = va.get("gate_mean", va.get("gate", 0.5))
    g_min = va.get("gate_min", 0.0)
    g_max = va.get("gate_max", 1.0)
    print(f"\n  {'=' * 68}")
    print(f"  Epoch {ep}")
    print(f"  {'=' * 68}")
    print(f"  Prediction    Temporal IC: {tr['t_ic']:+.3f} / {va['t_ic']:+.3f}"
          f"    Asset IC: {tr['xs_ic']:+.3f} / {va['xs_ic']:+.3f}")
    print(f"                Direction: {tr['dir_acc']:.1%} / {va['dir_acc']:.1%}"
          f"          NLL: {tr['nll']:.3f}")
    print(f"  Portfolio     Return: {tr['port_ret']:+.4%} / {va['port_ret']:+.4%}"
          f"       Bench: {tr['bench_ret']:+.4%}")
    print(f"                Exposure: {tr['gross']:.1%} / {va['gross']:.1%}"
          f"         Gate: {g_mean:.2f} [{g_min:.2f} to {g_max:.2f}]")
    print(f"  Signals       Bullish: {n_bull}/20    Buy:  {buys}")
    if sells != "none":
        print(f"                Bearish: {n_bear}/20    Sell: {sells}")
    print(f"  Weights       {_fmt_top(va['per_asset'], ASSETS)}")
    print(f"  Loss: {tl:.4f} / {vl:.4f}  {el:.0f}s", flush = True)

def _mk_dataset(npz_path, cfg, split):
    return Model2Dataset(npz_path, seq_len_4h = cfg["seq_len_4h"], seq_k = cfg["seq_k"],
                         split = split, stride = cfg["train_stride"],
                         lookback_15m = cfg["lookback_15m"], lookback_1h = cfg["lookback_1h"])


def _populate_cfg(ds, cfg):
    cfg["N"] = ds.N
    cfg["F_4h"] = ds.F_4h
    cfg["F_15m"] = ds.F_15m
    cfg["F_1h"] = ds.F_1h
    cfg["D_time_4h"] = ds.D_time_4h
    cfg["D_time_15m"] = ds.D_time_15m
    cfg["D_time_1h"] = ds.D_time_1h
    cfg["btc_idx"] = ds.assets.index("BTCUSDT")


def _mk_model(cfg, device):
    model = Model2(
        F_4h = cfg["F_4h"], F_15m = cfg["F_15m"], F_1h = cfg["F_1h"],
        D_time_4h = cfg["D_time_4h"], D_time_15m = cfg["D_time_15m"], D_time_1h = cfg["D_time_1h"],
        N_assets = cfg["N"], d_regime = cfg["d_regime"],
        d_model = cfg["d_model"], d_lstm = cfg["d_lstm"], d_cross = cfg["d_cross"],
        n_cross_heads = cfg["n_cross_heads"], dropout = cfg["dropout"],
        embed_drop = cfg["embed_drop"], t_recent = cfg["t_recent"]).to(device)
    return model


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location = device, weights_only = False)
    c = ckpt["cfg"]
    model = _mk_model(c, device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def train():
    cfg = dict(DEFAULT_CFG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    npz_path = os.path.join(BASE_DIR, "model2_dataset.npz")
    tr_ds = _mk_dataset(npz_path, cfg, "train")
    va_ds = _mk_dataset(npz_path, cfg, "val")
    te_ds = _mk_dataset(npz_path, cfg, "test")
    _populate_cfg(tr_ds, cfg)
    print(f"Samples: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}")
    m1_pop = (tr_ds.model1_outputs.sum(axis = -1) != 0).mean()
    m1_range = f"[{tr_ds.model1_outputs.min():+.4f},{tr_ds.model1_outputs.max():+.4f}]"
    print(f"m1 populated {m1_pop:.1%} range={m1_range}")
    model = _mk_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = len(tr_ds) / n_params if n_params > 0 else 0
    print(f"Params: {n_params:,} samples:params={ratio:.2f}:1")
    print(f"F_4h={cfg['F_4h']} F_1h={cfg['F_1h']} F_15m={cfg['F_15m']}")
    print(f"Model: d_model={cfg['d_model']} d_lstm={cfg['d_lstm']} d_cross={cfg['d_cross']} heads={cfg['n_cross_heads']} dropout={cfg['dropout']}")
    print(f"Loss: temporal_ic={cfg['temporal_ic_coef']} xs_ic={cfg['xs_ic_coef']} nll={cfg['nll_coef']} directional={cfg['directional_coef']}")
    tr_loader = DataLoader(tr_ds, batch_size = cfg["batch_size"], shuffle = True, drop_last = True, num_workers = 0)
    va_loader = DataLoader(va_ds, batch_size = cfg["batch_size"], shuffle = False, drop_last = False, num_workers = 0)
    te_loader = DataLoader(te_ds, batch_size = cfg["batch_size"], shuffle = False, drop_last = False, num_workers = 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg["lr"], weight_decay = cfg["weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = cfg["warmup_start_factor"], total_iters = cfg["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg["epochs"] - cfg["warmup_epochs"], eta_min = cfg["scheduler_eta_min"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones = [cfg["warmup_epochs"]])
    best_val = float("inf")
    wait = 0
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok = True)
    ckpt_path = os.path.join(ckpt_dir, "model2_best.pt")
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tl, tr_s = train_epoch(model, tr_loader, optimizer, cfg, ep, device)
        vl, va_s = eval_epoch(model, va_loader, cfg, device)
        scheduler.step()
        el = time.time() - t0
        _ep_print(ep, tr_s, va_s, tl, vl, el)
        if vl < best_val:
            best_val = vl
            wait = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                         "val_loss": vl, "val_stats": va_s}, ckpt_path)
            print(f"  => saved best val_loss={vl:.4f}")
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  Early stop ep {ep}")
                break
    print(f"\nLoading best for test")
    model, ckpt = load_ckpt(ckpt_path, device)
    tl, ts = eval_epoch(model, te_loader, cfg, device)
    buys, sells = _fmt_signals(ts.get("mean_pred", np.zeros(20)), ASSETS)
    g_mean = ts.get("gate_mean", ts.get("gate", 0.5))
    print(f"\n  {'=' * 68}")
    print(f"  TEST RESULTS")
    print(f"  {'=' * 68}")
    print(f"  Prediction    Temporal IC: {ts['t_ic']:+.3f}    Asset IC: {ts['xs_ic']:+.3f}    Direction: {ts['dir_acc']:.1%}")
    print(f"  Portfolio     Return: {ts['port_ret']:+.4%}    Bench: {ts['bench_ret']:+.4%}    BTC: {ts['btc_ret']:+.4%}")
    print(f"                Excess: {ts['excess_ret']:+.4%}    Exposure: {ts['gross']:.1%}    Gate: {g_mean:.2f} [{ts.get('gate_min',0):.2f} to {ts.get('gate_max',1):.2f}]")
    print(f"  Signals       Buy:  {buys}")
    if sells != "none":
        print(f"                Sell: {sells}")
    print(f"  Weights       {_fmt_top(ts['per_asset'], ASSETS)}")
    print(f"  Max weight: {ts['max_w']:.1%}    Concentration: {ts.get('hhi',0):.3f}")
    return model


if __name__ == "__main__":
    train()
