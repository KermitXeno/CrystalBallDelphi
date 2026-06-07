import os
import sys
import json
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
import xgboost as xgb
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds_struc_wstg import BASE_DIR

DEFAULT_CFG = {
    "n_estimators": 1500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 25,
    "early_stopping_rounds": 50,
    "train_frac": 0.70,
    "val_frac": 0.15,
}

REGIME_NAMES = ["bull", "bear", "accumulating", "distributing"]


def load_features(npz_path):
    d = np.load(npz_path, allow_pickle = True)
    market = np.array(d["market_features"], dtype = np.float32)
    glob = np.array(d["global_features"], dtype = np.float32)
    btc = np.array(d["btc_raw"], dtype = np.float32)
    X = np.concatenate([market, glob, btc], axis = 1)
    regime_labels = np.array(d["regime_labels"], dtype = np.int64)
    regime_scores = np.array(d["regime_scores"], dtype = np.float32)
    times = np.array(d["times"])
    feature_names = [str(s) for s in d["feature_names"]] if "feature_names" in d.files else None
    return X, regime_labels, regime_scores, times, feature_names


def make_soft_targets(regime_scores):
    y_dir = (regime_scores[:, 1] + regime_scores[:, 3]).astype(np.float32)
    y_phase = (regime_scores[:, 2] + regime_scores[:, 3]).astype(np.float32)
    return y_dir, y_phase


def chronological_split(X, y_dir, y_phase, regime_labels, train_frac = 0.70, val_frac = 0.15):
    n = len(X)
    n_tr = int(n * train_frac)
    n_va = int(n * (train_frac + val_frac))
    return {
        "X_tr": X[:n_tr], "X_val": X[n_tr:n_va], "X_te": X[n_va:],
        "dir_tr": y_dir[:n_tr], "dir_val": y_dir[n_tr:n_va], "dir_te": y_dir[n_va:],
        "phase_tr": y_phase[:n_tr], "phase_val": y_phase[n_tr:n_va], "phase_te": y_phase[n_va:],
        "regime_tr": regime_labels[:n_tr], "regime_val": regime_labels[n_tr:n_va], "regime_te": regime_labels[n_va:],
    }


def reconstruct_4class(dir_prob, phase_prob):
    probs = np.stack([
        (1 - dir_prob) * (1 - phase_prob),
        dir_prob * (1 - phase_prob),
        (1 - dir_prob) * phase_prob,
        dir_prob * phase_prob,
    ], axis = -1)
    return probs


def evaluate(dir_model, phase_model, X, regime_labels, split_name = "test"):
    dir_prob = dir_model.predict(X).clip(0.0, 1.0)
    phase_prob = phase_model.predict(X).clip(0.0, 1.0)
    probs_4c = reconstruct_4class(dir_prob, phase_prob)
    pred_4c = probs_4c.argmax(axis = -1)
    acc = accuracy_score(regime_labels, pred_4c)
    f1_macro = f1_score(regime_labels, pred_4c, average = "macro", zero_division = 0)
    f1_per = f1_score(regime_labels, pred_4c, average = None, labels = [0, 1, 2, 3], zero_division = 0)
    dir_true = ((regime_labels == 1) | (regime_labels == 3)).astype(int)
    phase_true = (regime_labels >= 2).astype(int)
    dir_acc = accuracy_score(dir_true, (dir_prob > 0.5).astype(int))
    phase_acc = accuracy_score(phase_true, (phase_prob > 0.5).astype(int))
    print(f"\n  {split_name} results:")
    print(f"    4-class acc={acc:.4f}  f1_macro={f1_macro:.4f}")
    print(f"    per-class: " + "  ".join(f"{REGIME_NAMES[i]}={f1_per[i]:.3f}" for i in range(4)))
    print(f"    direction acc={dir_acc:.4f}  phase acc={phase_acc:.4f}")
    print(f"\n{classification_report(regime_labels, pred_4c, target_names = REGIME_NAMES, zero_division = 0)}")
    return {"acc": acc, "f1_macro": f1_macro, "f1_per_class": f1_per.tolist(),
            "dir_acc": dir_acc, "phase_acc": phase_acc}


def print_feature_importance(model, feature_names, name, top_k = 20):
    imp = model.feature_importances_
    ranked = np.argsort(imp)[::-1][:top_k]
    print(f"\n  {name} top-{top_k} features (gain):")
    for i, idx in enumerate(ranked):
        fname = feature_names[idx] if feature_names else f"f{idx}"
        print(f"    {i + 1:>2}. {fname:<35} {imp[idx]:.4f}")


def calibration_analysis(dir_model, phase_model, X, regime_labels, split_name = "val"):
    dir_pred = dir_model.predict(X).clip(0.0, 1.0)
    phase_pred = phase_model.predict(X).clip(0.0, 1.0)
    dir_true = ((regime_labels == 1) | (regime_labels == 3)).astype(int)
    phase_true = (regime_labels >= 2).astype(int)
    probs_4c = reconstruct_4class(dir_pred, phase_pred)
    pred_4c = probs_4c.argmax(axis = -1)
    confidence = probs_4c.max(axis = -1)
    correct = (pred_4c == regime_labels).astype(int)

    print(f"\n{'=' * 60}")
    print(f"CALIBRATION ANALYSIS ({split_name})")
    print(f"{'=' * 60}")

    edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    print(f"\n  4-class confidence vs accuracy:")
    print(f"  {'confidence':<15} {'count':>8} {'accuracy':>10} {'pct_of_data':>12}")
    print(f"  {'-' * 48}")
    for i in range(len(edges) - 1):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        acc = correct[mask].mean()
        pct = n / len(confidence) * 100
        label = f"{edges[i]:.1f}-{edges[i + 1]:.1f}"
        tag = " *** OVERCONFIDENT" if acc < edges[i] else ""
        print(f"  {label:<15} {n:>8} {acc:>10.3f} {pct:>11.1f}%{tag}")

    for name, pred, true in [("Direction", dir_pred, dir_true), ("Phase", phase_pred, phase_true)]:
        print(f"\n  {name} calibration (pred vs actual):")
        print(f"  {'pred_range':<15} {'count':>8} {'pred_mean':>10} {'actual_mean':>12} {'gap':>8}")
        print(f"  {'-' * 56}")
        for i in range(10):
            lo, hi = i * 0.1, (i + 1) * 0.1
            mask = (pred >= lo) & (pred < hi)
            n = mask.sum()
            if n < 10:
                continue
            pm = pred[mask].mean()
            am = true[mask].mean()
            gap = pm - am
            tag = " ***" if abs(gap) > 0.15 else ""
            print(f"  {lo:.1f}-{hi:.1f}        {n:>8} {pm:>10.3f} {am:>12.3f} {gap:>+8.3f}{tag}")

    wrong = ~correct.astype(bool)
    print(f"\n  When WRONG (n={wrong.sum()}):")
    print(f"    mean confidence: {confidence[wrong].mean():.3f}")
    print(f"    median confidence: {np.median(confidence[wrong]):.3f}")
    print(f"  When RIGHT (n={correct.sum()}):")
    print(f"    mean confidence: {confidence[correct.astype(bool)].mean():.3f}")
    print(f"    median confidence: {np.median(confidence[correct.astype(bool)]):.3f}")

    really_wrong = wrong & (confidence > 0.7)
    print(f"\n  Confidently wrong (conf>0.7 AND wrong): {really_wrong.sum()} ({really_wrong.sum() / len(confidence) * 100:.1f}%)")
    if really_wrong.sum() > 0:
        for c in range(4):
            mask_c = really_wrong & (regime_labels == c)
            if mask_c.sum() > 0:
                pred_dist = np.bincount(pred_4c[mask_c], minlength = 4)
                print(f"    true={REGIME_NAMES[c]}: {mask_c.sum()} cases, predicted as {dict(zip(REGIME_NAMES, pred_dist))}")


def train(cfg = None):
    if cfg is None:
        cfg = dict(DEFAULT_CFG)
    npz_path = os.path.join(BASE_DIR, "model1_dataset.npz")
    print(f"Loading {npz_path}")
    X, regime_labels, regime_scores, times, feature_names = load_features(npz_path)
    y_dir, y_phase = make_soft_targets(regime_scores)
    print(f"Features: {X.shape}  Labels: {regime_labels.shape}")
    print(f"Direction target: mean={y_dir.mean():.3f}  std={y_dir.std():.3f}")
    print(f"Phase target: mean={y_phase.mean():.3f}  std={y_phase.std():.3f}")

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in features, filling with 0")
        X = np.nan_to_num(X, nan = 0.0)

    s = chronological_split(X, y_dir, y_phase, regime_labels, cfg["train_frac"], cfg["val_frac"])
    print(f"Split: train={len(s['X_tr'])}  val={len(s['X_val'])}  test={len(s['X_te'])}")

    xgb_params = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "learning_rate": cfg["learning_rate"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "reg_alpha": cfg["reg_alpha"],
        "reg_lambda": cfg["reg_lambda"],
        "min_child_weight": cfg["min_child_weight"],
        "early_stopping_rounds": cfg["early_stopping_rounds"],
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }

    print("\n" + "=" * 60)
    print("Training DIRECTION regressor (bear-camp strength)")
    print("=" * 60)
    t0 = time.time()
    dir_model = XGBRegressor(**xgb_params)
    dir_model.fit(s["X_tr"], s["dir_tr"],
                  eval_set = [(s["X_tr"], s["dir_tr"]), (s["X_val"], s["dir_val"])],
                  verbose = 50)
    dir_time = time.time() - t0
    best_dir = getattr(dir_model, "best_iteration", cfg["n_estimators"])
    print(f"  trained in {dir_time:.1f}s  best_iteration={best_dir}")

    print("\n" + "=" * 60)
    print("Training PHASE regressor (disagreement strength)")
    print("=" * 60)
    t0 = time.time()
    phase_model = XGBRegressor(**xgb_params)
    phase_model.fit(s["X_tr"], s["phase_tr"],
                    eval_set = [(s["X_tr"], s["phase_tr"]), (s["X_val"], s["phase_val"])],
                    verbose = 50)
    phase_time = time.time() - t0
    best_phase = getattr(phase_model, "best_iteration", cfg["n_estimators"])
    print(f"  trained in {phase_time:.1f}s  best_iteration={best_phase}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    train_metrics = evaluate(dir_model, phase_model, s["X_tr"], s["regime_tr"], "train")
    val_metrics = evaluate(dir_model, phase_model, s["X_val"], s["regime_val"], "val")
    test_metrics = evaluate(dir_model, phase_model, s["X_te"], s["regime_te"], "test")

    if feature_names:
        print_feature_importance(dir_model, feature_names, "Direction", top_k = 20)
        print_feature_importance(phase_model, feature_names, "Phase", top_k = 20)

    calibration_analysis(dir_model, phase_model, s["X_val"], s["regime_val"], "val")
    calibration_analysis(dir_model, phase_model, s["X_te"], s["regime_te"], "test")

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok = True)
    dir_path = os.path.join(ckpt_dir, "model1_direction.json")
    phase_path = os.path.join(ckpt_dir, "model1_phase.json")
    dir_model.save_model(dir_path)
    phase_model.save_model(phase_path)
    print(f"\nSaved direction model: {dir_path}")
    print(f"Saved phase model: {phase_path}")

    meta = {"cfg": cfg, "train": train_metrics, "val": val_metrics, "test": test_metrics,
            "dir_best_iter": int(getattr(dir_model, "best_iteration", cfg["n_estimators"])),
            "phase_best_iter": int(getattr(phase_model, "best_iteration", cfg["n_estimators"]))}
    meta_path = os.path.join(ckpt_dir, "model1_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent = 2)
    print(f"Saved meta: {meta_path}")

    return dir_model, phase_model, val_metrics


def generate_model1_outputs(dir_model = None, phase_model = None):
    npz_path = os.path.join(BASE_DIR, "model1_dataset.npz")
    X, regime_labels, regime_scores, times, _ = load_features(npz_path)
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        X = np.nan_to_num(X, nan = 0.0)

    if dir_model is None:
        ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
        dir_model = XGBRegressor()
        dir_model.load_model(os.path.join(ckpt_dir, "model1_direction.json"))
        phase_model = XGBRegressor()
        phase_model.load_model(os.path.join(ckpt_dir, "model1_phase.json"))

    dir_prob = dir_model.predict(X).clip(0.0, 1.0)
    phase_prob = phase_model.predict(X).clip(0.0, 1.0)
    dir_conf = np.abs(2.0 * dir_prob - 1.0)
    phase_conf = np.abs(2.0 * phase_prob - 1.0)
    transition_intensity = (1.0 - dir_conf) * (1.0 - phase_conf)
    outputs_15m = np.stack([dir_prob, phase_prob, dir_conf, phase_conf, transition_intensity], axis = -1).astype(np.float32)
    print(f"Generated model1_outputs: {outputs_15m.shape} at 15m cadence")
    print(f"  channels: direction_prob, phase_prob, dir_confidence, phase_confidence, transition_intensity")

    outputs_1h = outputs_15m[3::4]
    times_1h = times[3::4]
    print(f"Resampled to 1h: {outputs_1h.shape}")

    return outputs_15m, times, outputs_1h, times_1h


if __name__ == "__main__":
    dir_model, phase_model, _ = train()
    outputs_15m, times_15m, outputs_1h, times_1h = generate_model1_outputs(dir_model, phase_model)

    out_path = os.path.join(BASE_DIR, "model1_outputs.npz")
    np.savez_compressed(out_path,
                        model1_outputs_15m = outputs_15m, times_15m = times_15m,
                        model1_outputs_1h = outputs_1h, times_1h = times_1h)
    print(f"Saved {out_path}")
