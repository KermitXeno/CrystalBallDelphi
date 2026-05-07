import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_download import BASE_DIR
from model1_train import Model1, DEFAULT_CFG
from ds_struc_wstg import Model1Dataset, set_model1_outputs


def derive_m1_out_dim(n_cls):
    return n_cls + 3


def load_model_from_ckpt(ckpt_path, device, use_ema):
    ckpt = torch.load(ckpt_path, map_location = device, weights_only = False)
    cfg = {**DEFAULT_CFG, **ckpt.get("cfg", {})}
    model = Model1(cfg).to(device)
    if use_ema and "ema" in ckpt:
        print("loading ema shadow weights")
        ema_state = ckpt["ema"]
        if isinstance(ema_state, dict) and "shadow" in ema_state:
            model.load_state_dict(ema_state["shadow"])
        else:
            model.load_state_dict(ema_state)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def collect_logits(model, loader, device):
    all_regime, all_trans, all_trend = [], [], []
    for batch in loader:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        batch.pop("regime_label", None)
        batch.pop("transition_label", None)
        regime_logits, transition_logit, _, aux = model(batch)
        all_regime.append(regime_logits.cpu().numpy())
        all_trans.append(transition_logit.cpu().numpy())
        all_trend.append(aux["trending_logit"].cpu().numpy())
    return (np.concatenate(all_regime), np.concatenate(all_trans), np.concatenate(all_trend))


def pack_m1_out(regime_logits, trans_logits, trend_logits, n_cls, target_dim):
    x = regime_logits - regime_logits.max(axis = -1, keepdims = True)
    e = np.exp(x)
    probs = e / e.sum(axis = -1, keepdims = True)
    conf = probs.max(axis = -1, keepdims = True)
    trans_prob = (1.0 / (1.0 + np.exp(-trans_logits)))[:, None]
    trend_prob = (1.0 / (1.0 + np.exp(-trend_logits)))[:, None]
    base = np.concatenate([probs, trans_prob, conf, trend_prob], axis = -1).astype(np.float32)
    if base.shape[1] > target_dim:
        raise ValueError(
            f"packed m1 dim {base.shape[1]} exceeds m2 model1_outputs slot dim {target_dim}. "
            f"increase the model1_outputs placeholder in dp_preproc_model2.py to at least "
            f"shape (T_1h, {base.shape[1]}) and rerun preprocessing.")
    if base.shape[1] < target_dim:
        pad = np.zeros((base.shape[0], target_dim - base.shape[1]), dtype = np.float32)
        base = np.concatenate([base, pad], axis = -1)
    return base


def align_to_target_times(source_times, source_m1, target_times):
    src_sorted_idx = np.argsort(source_times)
    src_t = source_times[src_sorted_idx]
    src_v = source_m1[src_sorted_idx]
    n_src = len(src_t)
    pos = np.searchsorted(src_t, target_times)
    pos_safe = np.minimum(pos, max(n_src - 1, 0))
    matched = (n_src > 0) & (src_t[pos_safe] == target_times)
    out = np.zeros((len(target_times), source_m1.shape[1]), dtype = np.float32)
    out[matched] = src_v[pos_safe[matched]]
    return out, int(matched.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default = os.path.join(BASE_DIR, "checkpoints", "model1_best.pt"))
    ap.add_argument("--m1_npz", default = os.path.join(BASE_DIR, "model1_dataset.npz"))
    ap.add_argument("--m2_npz", default = os.path.join(BASE_DIR, "model2_dataset.npz"))
    ap.add_argument("--batch_size", type = int, default = 128)
    ap.add_argument("--use_ema", action = "store_true")
    ap.add_argument("--min_coverage", type = float, default = 0.95, help = "minimum fraction of m2 timestamps that must be matched (default 0.95)")
    ap.add_argument("--allow_low_coverage", action = "store_true",                                e is below --min_coverage")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"loading checkpoint {args.ckpt}")
    model, cfg, ckpt = load_model_from_ckpt(args.ckpt, device, args.use_ema)

    n_cls = cfg["n_regime_classes"]
    expected_dim = derive_m1_out_dim(n_cls)
    print(f"n_regime_classes = {n_cls}  packed m1 vector size = {expected_dim} "
          f"(= {n_cls} regime probs + 1 trans prob + 1 max conf + 1 trend prob)")

    print(f"reading m2 dataset metadata from {args.m2_npz}")
    m2_meta = np.load(args.m2_npz)
    target_times = m2_meta["times_1h"].astype(np.int64)
    target_shape = m2_meta["model1_outputs"].shape
    target_dim = target_shape[1]
    print(f"m2 timeline length = {len(target_times)}  m1 slot shape = {target_shape}")

    if target_dim < expected_dim:
        raise ValueError(
            f"m2 model1_outputs dim {target_dim} too small for n_cls = {n_cls} "
            f"(needs at least {expected_dim}). update the placeholder in dp_preproc_model2.py "
            f"to shape ({target_shape[0]}, {expected_dim}) and rerun preprocessing.")

    print(f"loading m1 dataset {args.m1_npz}")
    dataset = Model1Dataset(args.m1_npz, window = cfg["window"], graph_window = cfg["graph_window"])
    print(f"m1 T = {dataset.T}  samples = {len(dataset)}")

    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False,
                        num_workers = 0, pin_memory = device.type == "cuda")

    print("running inference over full m1 dataset")
    regime_logits, trans_logits, trend_logits = collect_logits(model, loader, device)
    n_pred = len(regime_logits)
    print(f"collected {n_pred} predictions")

    if regime_logits.shape[1] != n_cls:
        raise ValueError(
            f"checkpoint produced {regime_logits.shape[1]} regime classes but cfg says "
            f"n_regime_classes = {n_cls}. checkpoint and cfg are out of sync.")

    m1_packed = pack_m1_out(regime_logits, trans_logits, trend_logits, n_cls, target_dim)

    m1_full = np.load(args.m1_npz)
    times_full = m1_full["times"].astype(np.int64)
    sample_to_raw_idx = np.arange(n_pred) + cfg["window"]
    if sample_to_raw_idx.max() >= len(times_full):
        raise ValueError(
            f"sample-to-raw index max {sample_to_raw_idx.max()} >= len(times_full) "
            f"{len(times_full)}. m1 dataset and inference output are inconsistent.")
    decision_times = times_full[sample_to_raw_idx]

    aligned, matched = align_to_target_times(decision_times, m1_packed, target_times)
    coverage = matched / max(1, len(target_times))
    print(f"alignment matched {matched}/{len(target_times)} ({coverage:.2%})")

    if coverage < args.min_coverage:
        msg = (f"coverage {coverage:.2%} below threshold {args.min_coverage:.2%}. "
               f"unmatched rows will be zero-vector m1 inputs at inference.")
        if not args.allow_low_coverage:
            raise RuntimeError(msg + "  pass --allow_low_coverage to proceed anyway.")
        print(f"WARNING {msg}")

    nz_rows = np.abs(aligned).sum(axis = -1) > 0
    if nz_rows.any():
        print("per-slot mean over matched rows:")
        slot_names = [f"regime_p{i}" for i in range(n_cls)] + ["trans_p", "max_conf", "trend_p"]
        means = aligned[nz_rows].mean(axis = 0)
        for i in range(target_dim):
            label = slot_names[i] if i < len(slot_names) else f"pad{i - len(slot_names)}"
            print(f"  slot[{i:02d}] {label:>12}  {means[i]:+.4f}")

    print(f"writing back to {args.m2_npz}")
    set_model1_outputs(args.m2_npz, aligned)

    verify = np.load(args.m2_npz)
    saved = verify["model1_outputs"]
    if not np.allclose(saved, aligned, atol = 1e-6):
        max_diff = float(np.abs(saved - aligned).max())
        raise RuntimeError(
            f"roundtrip verification failed: written m1_outputs differ from intended "
            f"(max abs diff = {max_diff:.2e})")
    nz_saved = int((np.abs(saved).sum(axis = -1) > 0).sum())
    print(f"roundtrip verified  saved nonzero rows = {nz_saved}/{len(saved)}")
    print("done")


if __name__ == "__main__":
    main()
