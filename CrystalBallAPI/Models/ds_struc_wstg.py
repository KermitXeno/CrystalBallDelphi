import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os

ASSETS = [
    "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ATOMUSDT", "AVAXUSDT",
    "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOTUSDT", "ETCUSDT",
    "ETHUSDT", "FILUSDT", "LINKUSDT", "LTCUSDT",
    "MATICUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT",
    "XRPUSDT", "XTZUSDT",
]

BTC_IDX = ASSETS.index("BTCUSDT")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class Model1Dataset(Dataset):
    def __init__(self, npz_path, window = 672, assets = ASSETS):
        self.window = window
        self.assets = assets
        self.N = len(assets)

        d = np.load(npz_path, allow_pickle = False)
        self.node_features = np.array(d["node_features"], dtype = np.float32)
        self.global_features = np.array(d["global_features"], dtype = np.float32)
        self.btc_raw = np.array(d["btc_raw"], dtype = np.float32)
        self.regime_scores = np.array(d["regime_scores"], dtype = np.float32)
        self.regime_labels = np.array(d["regime_labels"], dtype = np.int64)
        self.transition_labels = np.array(d["transition_labels"], dtype = np.float32)
        self.times = np.array(d["times"])
        self.time_enc = np.array(d["time_enc"], dtype = np.float32)
        self.sentiment_scores = np.array(d["sentiment_scores"], dtype = np.float32)
        self.sentiment_missing = np.array(d["sentiment_missing"], dtype = np.float32)
        if "horizon_names" in d.files:
            self.horizon_names = [str(s) for s in d["horizon_names"]]
        else:
            self.horizon_names = ["1h", "4h", "16h", "64h"]

        self.T = self.node_features.shape[0]
        self.F_node = self.node_features.shape[2]
        self.F_global = self.global_features.shape[1]
        self.F_btc_raw = self.btc_raw.shape[1]
        self.D_time = self.time_enc.shape[1]
        self.F_sentiment = self.sentiment_scores.shape[2]

        assert self.node_features.shape == (self.T, self.N, self.F_node)
        assert self.global_features.shape == (self.T, self.F_global)
        assert self.btc_raw.shape == (self.T, self.F_btc_raw)
        assert self.regime_scores.shape == (self.T, 3)
        assert self.regime_labels.shape == (self.T,)
        assert self.transition_labels.shape == (self.T,)
        assert self.time_enc.shape == (self.T, self.D_time)
        assert self.sentiment_scores.shape == (self.T, self.N, self.F_sentiment)
        assert self.sentiment_missing.shape == (self.T,)
        assert self.T > self.window, "fewer bars than lookback window"

    def __len__(self):
        return self.T - self.window

    def __getitem__(self, idx):
        i = idx + self.window
        return {
            "node_features": torch.from_numpy(np.array(self.node_features[i - self.window : i])),
            "global_features": torch.from_numpy(np.array(self.global_features[i - self.window : i])),
            "time_enc": torch.from_numpy(np.array(self.time_enc[i - self.window : i])),
            "btc_raw": torch.from_numpy(np.array(self.btc_raw[i])),
            "sentiment_scores": torch.from_numpy(np.array(self.sentiment_scores[i])),
            "sentiment_missing": torch.tensor(float(self.sentiment_missing[i])),
            "regime_label": torch.tensor(int(self.regime_labels[i]), dtype = torch.long),
            "regime_score": torch.from_numpy(np.array(self.regime_scores[i])),
            "transition_label": torch.tensor(float(self.transition_labels[i])),
        }

    def get_current_sample(self):
        i = self.T
        return {
            "node_features": torch.from_numpy(np.array(self.node_features[i - self.window : i])).unsqueeze(0),
            "global_features": torch.from_numpy(np.array(self.global_features[i - self.window : i])).unsqueeze(0),
            "time_enc": torch.from_numpy(np.array(self.time_enc[i - self.window : i])).unsqueeze(0),
            "btc_raw": torch.from_numpy(np.array(self.btc_raw[i - 1])).unsqueeze(0),
            "sentiment_scores": torch.from_numpy(np.array(self.sentiment_scores[i - 1])).unsqueeze(0),
            "sentiment_missing": torch.tensor([float(self.sentiment_missing[i - 1])]),
        }

    def update_sentiment(self, scores, missing):
        assert scores.shape == self.sentiment_scores.shape, (
            f"scores shape mismatch: got {scores.shape}, expected {self.sentiment_scores.shape}")
        assert missing.shape == self.sentiment_missing.shape, (
            f"missing shape mismatch: got {missing.shape}, expected {self.sentiment_missing.shape}")
        self.sentiment_scores = scores.astype(np.float32)
        self.sentiment_missing = missing.astype(np.float32)

    def chronological_split(self, train = 0.70, val = 0.15):
        n = len(self)
        t_end = int(n * train)
        v_end = int(n * (train + val))
        return (
            Subset(self, range(0, t_end)),
            Subset(self, range(t_end, v_end)),
            Subset(self, range(v_end, n)),
        )

    def to_dataloader(self, batch_size = 32, shuffle = True, num_workers = 0):
        return DataLoader(
            self, batch_size = batch_size, shuffle = shuffle,
            num_workers = num_workers, pin_memory = torch.cuda.is_available(),
            drop_last = False,
        )


class Model2Dataset(Dataset):
    def __init__(self, npz_path, seq_len_1h = 72, seq_k = 8, split = "train", train_frac = 0.70, val_frac = 0.15, stride = 1, lookback_15m = 96, lookback_30m = 48, assets = ASSETS, bar_range = None):
        self.seq_len_1h = seq_len_1h
        self.seq_k = seq_k
        self.lookback_15m = lookback_15m
        self.lookback_30m = lookback_30m
        self.assets = assets
        self.N = len(assets)
        d = np.load(npz_path, allow_pickle = False)
        self.features_15m = np.array(d["features_15m"], dtype = np.float32)
        self.features_30m = np.array(d["features_30m"], dtype = np.float32)
        self.features_1h = np.array(d["features_1h"], dtype = np.float32)
        self.targets = np.array(d["targets"], dtype = np.float32)
        self.times_15m = np.array(d["times_15m"])
        self.times_30m = np.array(d["times_30m"])
        self.times_1h = np.array(d["times_1h"])
        self.time_enc_15m = np.array(d["time_enc_15m"], dtype = np.float32)
        self.time_enc_30m = np.array(d["time_enc_30m"], dtype = np.float32)
        self.time_enc_1h = np.array(d["time_enc_1h"], dtype = np.float32)
        self.window_idx_15m = np.array(d["window_idx_15m"], dtype = np.int32)
        self.window_idx_30m = np.array(d["window_idx_30m"], dtype = np.int32)
        self.model1_outputs = np.array(d["model1_outputs"], dtype = np.float32)
        self.sentiment_scores = np.array(d["sentiment_scores"], dtype = np.float32)
        self.sentiment_missing = np.array(d["sentiment_missing"], dtype = np.float32)
        self.T_1h = self.features_1h.shape[0]
        self.T_15m = self.features_15m.shape[0]
        self.T_30m = self.features_30m.shape[0]
        self.F_1h = self.features_1h.shape[2]
        self.F_15m = self.features_15m.shape[2]
        self.F_30m = self.features_30m.shape[2]
        self.D_time_1h = self.time_enc_1h.shape[1]
        self.D_time_15m = self.time_enc_15m.shape[1]
        self.D_time_30m = self.time_enc_30m.shape[1]
        self.F_model1 = self.model1_outputs.shape[1]
        self.F_sentiment = self.sentiment_scores.shape[2]
        assert self.features_1h.shape == (self.T_1h, self.N, self.F_1h)
        assert self.targets.shape == (self.T_1h, self.N, 3)
        assert self.model1_outputs.shape == (self.T_1h, self.F_model1)
        assert self.T_1h > seq_len_1h + seq_k, "fewer 1h bars than seq_len + seq_k"
        self.asset_vol = self.targets[:, :, 0].std(axis = 0).astype(np.float32)
        all_starts = np.arange(seq_len_1h, self.T_1h - seq_k - 1, stride)
        if bar_range is not None:
            lo, hi = bar_range
            self.starts = all_starts[(all_starts >= lo) & (all_starts + seq_k <= hi)]
            self.asset_vol = self.targets[lo:hi, :, 0].std(axis = 0).astype(np.float32)
        else:
            n = len(all_starts)
            n_tr = int(n * train_frac)
            n_va = int(n * val_frac)
            if split == "train":
                self.starts = all_starts[:n_tr]
            elif split == "val":
                self.starts = all_starts[n_tr : n_tr + n_va]
            elif split == "test":
                self.starts = all_starts[n_tr + n_va :]
            elif split == "all":
                self.starts = all_starts
            else:
                raise ValueError(f"unknown split: {split}")

    def __len__(self):
        return len(self.starts)

    def _pad_to(self, arr, start, end, target_len):
        window = np.array(arr[start:end], dtype = np.float32)
        actual = window.shape[0]
        if actual == target_len:
            return window
        if actual > target_len:
            return window[-target_len:]
        pad = np.zeros((target_len - actual,) + window.shape[1:], dtype = np.float32)
        return np.concatenate([pad, window], axis = 0)

    def __getitem__(self, idx):
        t0 = int(self.starts[idx])
        L = self.seq_len_1h
        K = self.seq_k
        ts = np.arange(t0, t0 + K)
        f1h = np.stack([self.features_1h[t - L + 1 : t + 1] for t in ts])
        te1h = np.stack([self.time_enc_1h[t - L + 1 : t + 1] for t in ts])
        f15m_list, te15m_list, f30m_list, te30m_list = [], [], [], []
        for t in ts:
            s15, e15 = int(self.window_idx_15m[t, 0]), int(self.window_idx_15m[t, 1])
            f15m_list.append(self._pad_to(self.features_15m, s15, e15, self.lookback_15m))
            te15m_list.append(self._pad_to(self.time_enc_15m, s15, e15, self.lookback_15m))
            s30, e30 = int(self.window_idx_30m[t, 0]), int(self.window_idx_30m[t, 1])
            f30m_list.append(self._pad_to(self.features_30m, s30, e30, self.lookback_30m))
            te30m_list.append(self._pad_to(self.time_enc_30m, s30, e30, self.lookback_30m))
        return {
            "features_1h": torch.from_numpy(f1h),
            "time_enc_1h": torch.from_numpy(te1h),
            "features_15m": torch.from_numpy(np.stack(f15m_list)),
            "time_enc_15m": torch.from_numpy(np.stack(te15m_list)),
            "features_30m": torch.from_numpy(np.stack(f30m_list)),
            "time_enc_30m": torch.from_numpy(np.stack(te30m_list)),
            "model1_outputs": torch.from_numpy(self.model1_outputs[t0 : t0 + K].astype(np.float32)),
            "targets": torch.from_numpy(self.targets[t0 : t0 + K].astype(np.float32)),
        }

    def get_current_sample(self):
        i = self.T_1h - 1
        L = self.seq_len_1h
        return {
            "features_1h": torch.from_numpy(np.array(self.features_1h[i - L + 1 : i + 1])).unsqueeze(0),
            "model1_outputs": torch.from_numpy(np.array(self.model1_outputs[i])).unsqueeze(0),
        }

    def update_sentiment(self, scores, missing):
        assert scores.shape == self.sentiment_scores.shape
        assert missing.shape == self.sentiment_missing.shape
        self.sentiment_scores = scores.astype(np.float32)
        self.sentiment_missing = missing.astype(np.float32)


def set_model1_outputs(npz_path, m1_outputs):
    d = dict(np.load(npz_path))
    expected_shape = d["model1_outputs"].shape
    assert m1_outputs.shape == expected_shape, (
        f"shape mismatch {m1_outputs.shape} vs {expected_shape}")
    d["model1_outputs"] = m1_outputs.astype(np.float32)
    np.savez_compressed(npz_path, **d)


if __name__ == "__main__":
    m1 = Model1Dataset(os.path.join(BASE_DIR, "model1_dataset.npz"), window = 672)
    print(f"Model1  T = {m1.T}  F_node = {m1.F_node}  F_global = {m1.F_global}  F_btc_raw = {m1.F_btc_raw}  D_time = {m1.D_time}")
    print(f"Model1  samples = {len(m1)}")
    sample = m1[0]
    print(f"Model1  sample keys: {list(sample.keys())}")
    print(f"Model1  node_features shape: {sample['node_features'].shape}")
    print(f"Model1  regime_label: {sample['regime_label']}  regime_score: {sample['regime_score'].numpy()}")
    train_sub, val_sub, test_sub = m1.chronological_split()
    print(f"Model1  split sizes: train = {len(train_sub)}  val = {len(val_sub)}  test = {len(test_sub)}")

    print()

    m2_train = Model2Dataset(os.path.join(BASE_DIR, "model2_dataset.npz"), seq_len_1h = 72, seq_k = 8, split = "train", stride = 4, lookback_15m = 96, lookback_30m = 48)
    m2_val = Model2Dataset(os.path.join(BASE_DIR, "model2_dataset.npz"), seq_len_1h = 72, seq_k = 8, split = "val", stride = 4, lookback_15m = 96, lookback_30m = 48)
    m2_test = Model2Dataset(os.path.join(BASE_DIR, "model2_dataset.npz"), seq_len_1h = 72, seq_k = 8, split = "test", stride = 4, lookback_15m = 96, lookback_30m = 48)
    print(f"Model2  T_1h = {m2_train.T_1h}  T_30m = {m2_train.T_30m}  T_15m = {m2_train.T_15m}")
    print(f"Model2  F_1h = {m2_train.F_1h}  F_30m = {m2_train.F_30m}  F_15m = {m2_train.F_15m}")
    print(f"Model2  F_model1 = {m2_train.F_model1}")
    print(f"Model2  split sizes: train = {len(m2_train)}  val = {len(m2_val)}  test = {len(m2_test)}")
    sample2 = m2_train[0]
    print(f"Model2  sample keys: {list(sample2.keys())}")
    print(f"Model2  features_1h shape: {sample2['features_1h'].shape}")
    print(f"Model2  targets shape: {sample2['targets'].shape}")
    print(f"Model2  model1_outputs shape: {sample2['model1_outputs'].shape}")
    live2 = m2_train.get_current_sample()
    print(f"Model2  live features_1h shape: {live2['features_1h'].shape}")
