import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds_struc_wstg import BASE_DIR, ASSETS


def load_model1():
    from xgboost import XGBRegressor
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    dir_model = XGBRegressor()
    dir_model.load_model(os.path.join(ckpt_dir, "model1_direction.json"))
    phase_model = XGBRegressor()
    phase_model.load_model(os.path.join(ckpt_dir, "model1_phase.json"))
    return dir_model, phase_model


def model1_predict(dir_model, phase_model, X):
    dir_prob = dir_model.predict(X).clip(0.0, 1.0)
    phase_prob = phase_model.predict(X).clip(0.0, 1.0)
    dir_conf = np.abs(2.0 * dir_prob - 1.0)
    phase_conf = np.abs(2.0 * phase_prob - 1.0)
    transition_intensity = (1.0 - dir_conf) * (1.0 - phase_conf)
    return np.stack([dir_prob, phase_prob, dir_conf, phase_conf, transition_intensity], axis = -1).astype(np.float32)


def generate_training_outputs():
    from model1_train import load_features
    dir_model, phase_model = load_model1()
    npz_path = os.path.join(BASE_DIR, "model1_dataset.npz")
    X, _, _, times, _ = load_features(npz_path)
    X = np.nan_to_num(X, nan = 0.0)
    outputs_15m = model1_predict(dir_model, phase_model, X)
    outputs_1h = outputs_15m[3::4]
    times_1h = times[3::4]
    print(f"Model1 outputs: {outputs_15m.shape} (15m) -> {outputs_1h.shape} (1h)")
    out_path = os.path.join(BASE_DIR, "model1_outputs.npz")
    np.savez_compressed(out_path,
                        model1_outputs_15m = outputs_15m, times_15m = times,
                        model1_outputs_1h = outputs_1h, times_1h = times_1h)
    print(f"Saved {out_path}")
    return outputs_15m, times, outputs_1h, times_1h


def load_model2(device = "cpu"):
    import torch
    from model2_train import load_ckpt
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "model2_best.pt")
    model, ckpt = load_ckpt(ckpt_path, device)
    model.eval()
    return model, ckpt


def live_inference(model2, dir_model, phase_model, m2_dataset, cfg, prev_position = None):
    import torch
    from model2_train import build_trade_decision
    device = next(model2.parameters()).device
    sample = m2_dataset.get_current_sample()
    m1_features = sample.get("model1_outputs", None)
    if m1_features is None:
        raise ValueError("model1_outputs not in Model2Dataset sample")
    m1_out = torch.from_numpy(m1_features).to(device) if isinstance(m1_features, np.ndarray) else m1_features.to(device)
    with torch.no_grad():
        out = model2(
            f4h = sample["features_4h"].to(device),
            te4h = sample.get("time_enc_4h", torch.zeros(1, 72, 8)).to(device),
            f15m = sample.get("features_15m", torch.zeros(1, 96, len(ASSETS), 1)).to(device),
            te15m = sample.get("time_enc_15m", torch.zeros(1, 96, 8)).to(device),
            f1h = sample.get("features_1h", torch.zeros(1, 48, len(ASSETS), 1)).to(device),
            te1h = sample.get("time_enc_1h", torch.zeros(1, 48, 8)).to(device),
            m1_out = m1_out)
    if prev_position is None:
        prev_position = np.zeros(len(ASSETS))
    decision = build_trade_decision(out, prev_position, cfg)
    m1_vals = m1_out[0].cpu().numpy() if torch.is_tensor(m1_out) else m1_out[0]
    decision["regime"] = {
        "direction_prob": float(m1_vals[0]),
        "phase_prob": float(m1_vals[1]),
        "direction_confidence": float(m1_vals[2]),
        "phase_confidence": float(m1_vals[3]),
        "transition_intensity": float(m1_vals[4]),
    }
    bull_prob = (1 - m1_vals[0]) * (1 - m1_vals[1])
    bear_prob = m1_vals[0] * (1 - m1_vals[1])
    acc_prob = (1 - m1_vals[0]) * m1_vals[1]
    dist_prob = m1_vals[0] * m1_vals[1]
    decision["regime_4class"] = {
        "bull": float(bull_prob),
        "bear": float(bear_prob),
        "accumulating": float(acc_prob),
        "distributing": float(dist_prob),
    }
    return decision


if __name__ == "__main__":
    outputs_15m, times_15m, outputs_1h, times_1h = generate_training_outputs()
    print(f"\nModel1 output stats (15m cadence):")
    names = ["direction_prob", "phase_prob", "dir_confidence", "phase_confidence", "transition_intensity"]
    for i, name in enumerate(names):
        col = outputs_15m[:, i]
        print(f"  {name}: mean={col.mean():.3f}  std={col.std():.3f}  min={col.min():.3f}  max={col.max():.3f}")
