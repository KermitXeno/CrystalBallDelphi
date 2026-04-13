import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model1_layers import (
    GrangerGraphEncoder, TemporalAssetEncoder, SentimentEncoder,
    GRN, SupConRegularizer, EMAModel,
)
from ds_struc_wstg import Model1Dataset, BASE_DIR


DEFAULT_CFG = {
    "N": 10,
    "F_node": 42,
    "F_global": 15,
    "D_time": 7,
    "F_sentiment": 5,
    "window": 96,
    "n_regime_classes": 3,

    "d_model": 64,
    "d_lstm": 96,
    "d_embed": 64,
    "n_heads": 4,
    "n_graph_layers": 2,
    "n_lstm_layers": 1,
    "F_edge": 4,
    "d_edge": 16,
    "graph_window": 8,
    "F_btc_raw": 4,
    "d_btc": 32,
    "d_sentiment": 24,
    "pool_stride": 4,
    "dropout": 0.20,
    "btc_drop": 0.20,
    "stoch_depth": 0.15,
    "edge_drop": 0.15,

    "label_smoothing": 0.08,
    "lambda_transition": 0.10,
    "lambda_aux": 0.15,
    "lambda_supcon": 0.05,
    "lr": 1.5e-4,
    "transition_lr_scale": 1.0,
    "weight_decay": 2e-3,
    "batch_size": 64,
    "grad_accum_steps": 1,
    "epochs": 200,
    "grad_clip": 1.0,
    "warmup_epochs": 10,
    "warmup_start_factor": 0.1,
    "train_stride": 3,
    "scheduler_eta_min": 5e-5,
    "patience": 35,
    "use_amp": True,
    "mixup_alpha": 0.0,
    "ema_decay": 0.999,
    "ema_eval_after": 10,
    "transition_pos_weight_cap": 4.0,
    "temporal_mask_prob": 0.15,
    "temporal_block_size": 8,
    "feature_mask_prob": 0.10,
    "feature_noise_std": 0.03,
}


class Model1(nn.Module):

    def __init__(self, cfg = None):
        super().__init__()
        cfg = {**DEFAULT_CFG, **(cfg or {})}

        F_node = cfg["F_node"]
        F_global = cfg["F_global"]
        D_time = cfg["D_time"]
        N = cfg["N"]
        F_sentiment = cfg["F_sentiment"]
        d_model = cfg["d_model"]
        d_lstm = cfg["d_lstm"]
        d_embed = cfg["d_embed"]
        n_heads = cfg["n_heads"]
        n_graph_layers = cfg["n_graph_layers"]
        n_lstm_layers = cfg["n_lstm_layers"]
        d_sentiment = cfg["d_sentiment"]
        dropout = cfg["dropout"]
        n_cls = cfg["n_regime_classes"]
        F_btc_raw = cfg.get("F_btc_raw", 4)
        d_btc = cfg.get("d_btc", 32)

        self.graph_window = cfg["graph_window"]
        self.F_btc_raw = F_btc_raw

        self.graph_encoder = GrangerGraphEncoder(
            F_node, d_embed,
            n_heads = n_heads, n_layers = n_graph_layers,
            F_edge = cfg["F_edge"], d_edge = cfg["d_edge"],
            dropout = dropout,
            stoch_depth = cfg.get("stoch_depth", 0.1),
            edge_drop = cfg.get("edge_drop", 0.1),
            k_graph = cfg["graph_window"])

        self.temporal_encoder = TemporalAssetEncoder(
            F_node, F_global, D_time,
            d_model = d_model, d_lstm = d_lstm,
            n_lstm_layers = n_lstm_layers,
            pool_stride = cfg.get("pool_stride", 4),
            dropout = dropout)

        self.sentiment_encoder = SentimentEncoder(
            N, F_sentiment, d_sentiment, dropout = dropout)

        self.asset_embed = nn.Embedding(N, d_embed)
        nn.init.normal_(self.asset_embed.weight, std = 0.02)
        self.asset_proj_temporal = (
            nn.Linear(d_embed, d_model, bias = False) if d_embed != d_model
            else None)

        d_graph = d_embed * 2
        self.temporal_gate = GRN(d_lstm, d_lstm, d_ctx = d_graph, dropout = dropout)
        self.graph_gate = GRN(d_graph, d_graph, d_ctx = d_lstm, dropout = dropout)

        d_ctx = d_lstm + d_graph + d_sentiment + 1

        self.btc_proj = nn.Linear(F_btc_raw, d_btc)
        self.btc_drop = nn.Dropout(cfg.get("btc_drop", 0.20))
        self.btc_regime_head = nn.Linear(d_btc, n_cls)

        d_fused = d_ctx + d_btc
        self.regime_head = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused // 2, n_cls),
        )

        self.trending_head = nn.Linear(d_ctx, 1)
        self.adx_head = nn.Linear(d_ctx, 1)

        self.d_combined = d_fused

        self.transition_head = nn.Sequential(
            nn.Linear(d_fused, d_fused // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused // 2, 1),
        )

        self.supcon = None
        if cfg.get("lambda_supcon", 0) > 0:
            self.supcon = SupConRegularizer(d_fused, d_proj = 64, temperature = 0.1)

        self.cfg = cfg

    def _asset_emb(self):
        a = self.asset_embed.weight
        if self.asset_proj_temporal is not None:
            return a, self.asset_proj_temporal(a)
        return a, a

    def encode(self, batch):
        node_features = batch["node_features"]
        global_features = batch["global_features"]
        time_enc = batch["time_enc"]
        edge_features = batch["edge_features"]
        sentiment_scores = batch["sentiment_scores"]
        sentiment_missing = batch["sentiment_missing"]

        a_graph, a_temporal = self._asset_emb()

        temporal_vec = self.temporal_encoder(
            node_features, global_features, time_enc, asset_embed = a_temporal)

        gw = self.graph_window
        node_feats_graph = node_features[:, -gw :, :, :]
        node_emb = self.graph_encoder(
            node_feats_graph, edge_features, asset_embed = a_graph)
        graph_mean = node_emb.mean(dim = 1)
        graph_max = node_emb.max(dim = 1).values
        graph_vec = torch.cat([graph_mean, graph_max], dim = -1)

        temporal_fused = self.temporal_gate(temporal_vec, ctx = graph_vec)
        graph_fused = self.graph_gate(graph_vec, ctx = temporal_vec)

        sent_vec = self.sentiment_encoder(sentiment_scores, sentiment_missing)

        ctx_vec = torch.cat([temporal_fused, graph_fused, sent_vec], dim = -1)

        btc_raw = batch["btc_raw"]
        if self.training:
            btc_raw = self.btc_drop(btc_raw)
        btc_hidden = F.gelu(self.btc_proj(btc_raw))

        return ctx_vec, btc_hidden

    def forward(self, batch):
        ctx_vec, btc_hidden = self.encode(batch)

        btc_logits = self.btc_regime_head(btc_hidden)
        fused = torch.cat([ctx_vec, btc_hidden], dim = -1)
        ctx_logits = self.regime_head(fused)
        regime_logits = btc_logits + ctx_logits

        transition_logit = self.transition_head(fused).squeeze(-1)

        aux = {
            "trending_logit": self.trending_head(ctx_vec).squeeze(-1),
            "adx_pred": self.adx_head(ctx_vec).squeeze(-1),
        }

        return regime_logits, transition_logit, fused, aux


class Model1Loss(nn.Module):

    def __init__(self, class_weights, transition_pos_weight,
                 lambda_transition = 0.10, lambda_aux = 0.15,
                 label_smoothing = 0.08):
        super().__init__()
        self.regime_loss = nn.CrossEntropyLoss(
            weight = class_weights, label_smoothing = label_smoothing)
        self.transition_loss = nn.BCEWithLogitsLoss(pos_weight = transition_pos_weight)
        self.trending_loss = nn.BCEWithLogitsLoss()
        self.aux_reg = nn.SmoothL1Loss()
        self.lambda_transition = lambda_transition
        self.lambda_aux = lambda_aux

    def forward(self, regime_logits, transition_logit, regime_labels,
                transition_labels, aux = None, adx_target = None):
        r_loss = self.regime_loss(regime_logits, regime_labels)
        t_loss = self.transition_loss(transition_logit, transition_labels)
        total = r_loss + self.lambda_transition * t_loss

        if aux is not None:
            trending_target = (regime_labels < 2).float()
            total = total + self.lambda_aux * self.trending_loss(
                aux["trending_logit"], trending_target)
            if adx_target is not None:
                total = total + self.lambda_aux * self.aux_reg(
                    aux["adx_pred"], adx_target)

        return total, r_loss.detach(), t_loss.detach()


def compute_class_weights(dataset, n_classes = 3, power = 1.0):
    labels = dataset.regime_labels
    counts = np.bincount(labels.astype(np.int64), minlength = n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv_freq = counts.sum() / (n_classes * counts)
    raw = np.power(inv_freq, power)
    weights = raw / raw.sum() * n_classes
    return torch.from_numpy(weights)


def compute_transition_pos_weight(dataset, cap = 4.0):
    labels = dataset.transition_labels
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    pos_weight = min(n_neg / max(n_pos, 1.0), cap)
    return torch.tensor([pos_weight])


def augment_batch(batch, cfg):
    node = batch["node_features"]
    glob = batch["global_features"]
    tenc = batch["time_enc"]
    B, T = node.shape[:2]
    device = node.device

    mask_prob = cfg.get("temporal_mask_prob", 0.0)
    block_size = cfg.get("temporal_block_size", 8)
    if mask_prob > 0:
        protect_tail = 2
        maskable_T = T - protect_tail
        n_blocks = (maskable_T + block_size - 1) // block_size
        block_keep = torch.bernoulli(
            torch.full((B, n_blocks), 1.0 - mask_prob, device = device))
        time_mask = block_keep.repeat_interleave(block_size, dim = 1)[:, :maskable_T]
        time_mask = torch.cat(
            [time_mask, torch.ones(B, protect_tail, device = device)], dim = 1)
        node = node * time_mask[:, :, None, None]
        glob = glob * time_mask[:, :, None]
        tenc = tenc * time_mask[:, :, None]

    feat_mask_prob = cfg.get("feature_mask_prob", 0.0)
    if feat_mask_prob > 0:
        F_node = node.shape[-1]
        F_glob = glob.shape[-1]
        node_fmask = torch.bernoulli(
            torch.full((B, 1, 1, F_node), 1.0 - feat_mask_prob, device = device))
        glob_fmask = torch.bernoulli(
            torch.full((B, 1, F_glob), 1.0 - feat_mask_prob, device = device))
        node = node * node_fmask
        glob = glob * glob_fmask

    noise_std = cfg.get("feature_noise_std", 0.0)
    if noise_std > 0:
        node = node + torch.randn_like(node) * noise_std
        glob = glob + torch.randn_like(glob) * noise_std

    batch["node_features"] = node
    batch["global_features"] = glob
    batch["time_enc"] = tenc
    return batch


def _to_device(batch, device):
    return {k: v.to(device, non_blocking = True) for k, v in batch.items()}


def _compute_metrics(regime_pred, regime_true, trans_pred, trans_true, n_classes = 3):
    acc = float((regime_pred == regime_true).mean())

    f1_per_class = []
    for c in range(n_classes):
        tp = ((regime_pred == c) & (regime_true == c)).sum()
        fp = ((regime_pred == c) & (regime_true != c)).sum()
        fn = ((regime_pred != c) & (regime_true == c)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1_per_class.append(2 * prec * rec / (prec + rec + 1e-9))
    f1_macro = float(np.mean(f1_per_class))

    trans_binary = (trans_pred >= 0.5).astype(np.int32)
    tp_t = ((trans_binary == 1) & (trans_true == 1)).sum()
    fn_t = ((trans_binary == 0) & (trans_true == 1)).sum()
    trans_recall = float(tp_t / (tp_t + fn_t + 1e-9))

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_per_class": [float(x) for x in f1_per_class],
        "trans_recall": trans_recall,
    }


def train_epoch(model, loader, optimizer, loss_fn, scaler, device, grad_clip,
                cfg = None, ema = None):
    model.train()
    total_loss = total_r = total_t = 0.0
    n = 0
    lambda_supcon = cfg.get("lambda_supcon", 0.0) if cfg else 0.0
    grad_accum = cfg.get("grad_accum_steps", 1) if cfg else 1

    optimizer.zero_grad(set_to_none = True)

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        regime_labels = batch.pop("regime_label")
        transition_labels = batch.pop("transition_label")
        adx_target = batch["btc_raw"][:, 2]

        batch = augment_batch(batch, cfg)

        with torch.amp.autocast(device_type = device.type, enabled = scaler is not None):
            regime_logits, transition_logit, combined, aux = model(batch)
            loss, r_loss, t_loss = loss_fn(
                regime_logits, transition_logit, regime_labels, transition_labels,
                aux = aux, adx_target = adx_target)

            if lambda_supcon > 0 and model.supcon is not None:
                loss = loss + lambda_supcon * model.supcon(combined, regime_labels)

            loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            if ema is not None:
                ema.update(model)

        bs = regime_labels.size(0)
        total_loss += loss.item() * grad_accum * bs
        total_r += r_loss.item() * bs
        total_t += t_loss.item() * bs
        n += bs

    return total_loss / n, total_r / n, total_t / n


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_r = total_t = 0.0
    n = 0
    all_regime_pred, all_regime_true = [], []
    all_trans_pred, all_trans_true = [], []

    for batch in loader:
        batch = _to_device(batch, device)
        regime_labels = batch.pop("regime_label")
        transition_labels = batch.pop("transition_label")

        regime_logits, transition_logit, _, _ = model(batch)
        loss, r_loss, t_loss = loss_fn(
            regime_logits, transition_logit, regime_labels, transition_labels)

        bs = regime_labels.size(0)
        total_loss += loss.item() * bs
        total_r += r_loss.item() * bs
        total_t += t_loss.item() * bs
        n += bs

        all_regime_pred.append(regime_logits.argmax(dim = -1).cpu().numpy())
        all_regime_true.append(regime_labels.cpu().numpy())
        all_trans_pred.append(torch.sigmoid(transition_logit).cpu().numpy())
        all_trans_true.append(transition_labels.cpu().numpy())

    metrics = _compute_metrics(
        np.concatenate(all_regime_pred),
        np.concatenate(all_regime_true),
        np.concatenate(all_trans_pred),
        np.concatenate(all_trans_true),
    )
    return total_loss / n, total_r / n, total_t / n, metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, ema = None):
    state = {
        "cfg": model.cfg,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


def load_checkpoint(path, device = "cpu"):
    ckpt = torch.load(path, map_location = device)
    model = Model1(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def train(cfg = None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    npz_path = os.path.join(BASE_DIR, "model1_dataset.npz")
    dataset = Model1Dataset(npz_path, window = cfg["window"],
                            graph_window = cfg["graph_window"])
    print(f"Dataset: T={dataset.T}  F_node={dataset.F_node}  "
          f"F_global={dataset.F_global}  samples={len(dataset)}")

    cfg["F_node"] = dataset.F_node
    cfg["F_global"] = dataset.F_global
    cfg["D_time"] = dataset.D_time
    cfg["F_sentiment"] = dataset.F_sentiment
    cfg["F_edge"] = dataset.F_edge
    cfg["F_btc_raw"] = getattr(dataset, "F_btc_raw", 4)

    train_sub, val_sub, test_sub = dataset.chronological_split(
        train = 0.70, val = 0.15)
    if cfg["train_stride"] > 1:
        from torch.utils.data import Subset as _Subset
        strided = list(range(0, len(train_sub), cfg["train_stride"]))
        train_sub = _Subset(train_sub.dataset, [train_sub.indices[i] for i in strided])
    print(f"Split: train={len(train_sub)}  val={len(val_sub)}  test={len(test_sub)}")

    num_workers = 0
    train_loader = DataLoader(
        train_sub, batch_size = cfg["batch_size"], shuffle = True,
        num_workers = num_workers, pin_memory = device.type == "cuda", drop_last = True)
    val_loader = DataLoader(
        val_sub, batch_size = cfg["batch_size"] * 2, shuffle = False,
        num_workers = num_workers, pin_memory = device.type == "cuda")
    test_loader = DataLoader(
        test_sub, batch_size = cfg["batch_size"] * 2, shuffle = False,
        num_workers = num_workers, pin_memory = device.type == "cuda")

    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    n_cls = cfg["n_regime_classes"]
    class_weights = compute_class_weights(base_ds, n_classes = n_cls).to(device)
    trans_pos_weight = compute_transition_pos_weight(
        base_ds, cap = cfg["transition_pos_weight_cap"]).to(device)
    labels = base_ds.regime_labels
    counts = np.bincount(labels.astype(np.int64), minlength = n_cls)
    print(f"Class counts: {counts.tolist()}  weights: {[round(w, 3) for w in class_weights.tolist()]}")
    n_pos = (base_ds.transition_labels == 1).sum()
    print(f"Transition labels: {n_pos}/{len(base_ds.transition_labels)} positive  pos_weight={trans_pos_weight.item():.2f}")

    model = Model1(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  ({n_params * 4 / 1e6:.1f} MB float32)")

    loss_fn = Model1Loss(
        class_weights,
        trans_pos_weight,
        lambda_transition = cfg["lambda_transition"],
        lambda_aux = cfg.get("lambda_aux", 0.15),
        label_smoothing = cfg["label_smoothing"])

    trans_ids = set(id(p) for p in model.transition_head.parameters())
    supcon_ids = set(id(p) for p in model.supcon.parameters()) if model.supcon else set()
    aux_head_params = list(model.trending_head.parameters()) + list(model.adx_head.parameters())
    aux_ids = trans_ids | supcon_ids | set(id(p) for p in aux_head_params)
    main_params = [p for p in model.parameters() if id(p) not in aux_ids and p.requires_grad]
    trans_params = [p for p in model.transition_head.parameters() if p.requires_grad]
    param_groups = [
        {"params": main_params, "lr": cfg["lr"]},
        {"params": trans_params, "lr": cfg["lr"] * cfg["transition_lr_scale"]},
        {"params": aux_head_params, "lr": cfg["lr"]},
    ]
    if model.supcon is not None:
        supcon_params = [p for p in model.supcon.parameters() if p.requires_grad]
        param_groups.append({"params": supcon_params, "lr": cfg["lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay = cfg["weight_decay"])

    cosine_T = cfg["epochs"] - cfg["warmup_epochs"]
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = cfg["warmup_start_factor"],
        end_factor = 1.0,
        total_iters = cfg["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = cosine_T,
        eta_min = cfg["scheduler_eta_min"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [cfg["warmup_epochs"]])

    print(f"Scheduler: warmup={cfg['warmup_epochs']}  cosine_T={cosine_T}  "
          f"eta_min={cfg['scheduler_eta_min']}  total_epochs={cfg['epochs']}")

    use_amp = cfg["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    ema = EMAModel(model, decay = cfg["ema_decay"])
    ema_eval_after = cfg.get("ema_eval_after", 10)

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok = True)
    best_path = os.path.join(ckpt_dir, "model1_best.pt")
    last_path = os.path.join(ckpt_dir, "model1_last.pt")

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_r, train_t = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            cfg["grad_clip"], cfg = cfg, ema = ema)

        use_ema_eval = epoch > ema_eval_after
        if use_ema_eval:
            ema.apply_shadow(model)
        val_loss, val_r, val_t, val_metrics = eval_epoch(
            model, val_loader, loss_fn, device)
        if use_ema_eval:
            ema.restore(model)

        scheduler.step()
        elapsed = time.time() - t0

        f1_str = "  ".join(f"C{i}={v:.3f}" for i, v in enumerate(val_metrics["f1_per_class"]))
        ema_tag = " [ema]" if use_ema_eval else ""
        print(
            f"Ep {epoch:03d}  "
            f"train {train_loss:.4f} (r={train_r:.4f} t={train_t:.4f})  "
            f"val {val_loss:.4f} (r={val_r:.4f} t={val_t:.4f})  "
            f"acc={val_metrics['acc']:.3f}  f1={val_metrics['f1_macro']:.3f}  "
            f"t_rec={val_metrics['trans_recall']:.3f}  "
            f"{f1_str}  "
            f"{elapsed:.1f}s{ema_tag}"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_loss, ema = ema)

        val_f1 = val_metrics["f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_count = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_loss, ema = ema)
            print(f"  => saved best  val_f1={best_val_f1:.4f}  val_loss={best_val_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"Early stop at epoch {epoch} (patience={cfg['patience']})")
                break

    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(best_path, map_location = device)
    model = Model1(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt:
        best_ema = EMAModel(model)
        best_ema.load_state_dict(ckpt["ema"])
        best_ema.apply_shadow(model)

    test_loss, test_r, test_t, test_metrics = eval_epoch(model, test_loader, loss_fn, device)
    f1_str = "  ".join(f"C{i}={v:.3f}" for i, v in enumerate(test_metrics["f1_per_class"]))
    print(
        f"Test  loss={test_loss:.4f}  acc={test_metrics['acc']:.3f}  "
        f"f1_macro={test_metrics['f1_macro']:.3f}  "
        f"trans_recall={test_metrics['trans_recall']:.3f}  "
        f"{f1_str}"
    )

    return model, test_metrics


if __name__ == "__main__":
    train()
