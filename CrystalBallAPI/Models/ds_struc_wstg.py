import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os

ASSETS = [
    "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT",
    "LINKUSDT", "LTCUSDT", "MATICUSDT", "SOLUSDT", "XRPUSDT",
]

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class Model1Dataset(Dataset):
    def __init__(self, npz_path, window = 168, graph_window = 8, assets = ASSETS):
        self.window = window
        self.graph_window = graph_window
        self.assets = assets
        self.N = len(assets)

        d = np.load(npz_path, allow_pickle = False)
        self.node_features = np.array(d["node_features"], dtype = np.float32)
        self.global_features = np.array(d["global_features"], dtype = np.float32)
        self.btc_raw = np.array(d["btc_raw"], dtype = np.float32)
        self.regime_labels = np.array(d["regime_labels"], dtype = np.int64)
        self.transition_labels = np.array(d["transition_labels"], dtype = np.float32)
        
        self.times = np.array(d["times"])
        self.time_enc = np.array(d["time_enc"], dtype = np.float32)
        self.sentiment_scores = np.array(d["sentiment_scores"], dtype = np.float32)
        self.sentiment_missing = np.array(d["sentiment_missing"], dtype = np.float32)

        self.T = self.node_features.shape[0]
        self.F_node = self.node_features.shape[2]
        self.F_global = self.global_features.shape[1]
        self.F_btc_raw = self.btc_raw.shape[1]
        self.D_time = self.time_enc.shape[1]
        self.F_sentiment = self.sentiment_scores.shape[2]

        assert self.node_features.shape == (self.T, self.N, self.F_node)
        assert self.global_features.shape == (self.T, self.F_global)
        assert self.btc_raw.shape == (self.T, self.F_btc_raw)
        assert self.regime_labels.shape == (self.T,)
        assert self.transition_labels.shape == (self.T,)
        graph = np.load(os.path.join(BASE_DIR, "graph_edges.npz"))
        graph_times = graph["times"]
        self.edge_features = graph["edge_features"].astype(np.float32)
        self.F_edge = self.edge_features.shape[3]
        self.graph_idx = np.searchsorted(graph_times, self.times)
        assert np.all(graph_times[self.graph_idx] == self.times), (
            "graph_edges.npz timestamps do not align with model1_dataset")
        
        assert self.time_enc.shape == (self.T, self.D_time)
        assert self.sentiment_scores.shape == (self.T, self.N, self.F_sentiment)
        assert self.sentiment_missing.shape == (self.T,)
        assert self.T > self.window, "fewer bars than lookback window"

    def __len__(self):
        return self.T - self.window

    def _get_edge_window(self, i):
        gi = self.graph_idx[i]
        start = max(gi - self.graph_window + 1, 0)
        end = gi + 1
        window = np.array(self.edge_features[start:end], dtype = np.float32)
        actual = window.shape[0]
        if actual < self.graph_window:
            pad_shape = (self.graph_window - actual,) + window.shape[1:]
            window = np.concatenate([np.zeros(pad_shape, dtype = np.float32), window], axis = 0)
        return torch.from_numpy(window)

    def __getitem__(self, idx):
        i = idx + self.window
        return {
            "node_features": torch.from_numpy(np.array(self.node_features[i - self.window : i])),
            "global_features": torch.from_numpy(np.array(self.global_features[i - self.window : i])),
            "time_enc": torch.from_numpy(np.array(self.time_enc[i - self.window : i])),
            "edge_features": self._get_edge_window(i),
            "btc_raw": torch.from_numpy(np.array(self.btc_raw[i])),
            "sentiment_scores": torch.from_numpy(np.array(self.sentiment_scores[i])),
            "sentiment_missing": torch.tensor(float(self.sentiment_missing[i])),
            "regime_label": torch.tensor(int(self.regime_labels[i]), dtype = torch.long),
            "transition_label": torch.tensor(float(self.transition_labels[i])),
        }

    def get_current_sample(self):
        i = self.T
        return {
            "node_features": torch.from_numpy(np.array(self.node_features[i - self.window : i])).unsqueeze(0),
            "global_features": torch.from_numpy(np.array(self.global_features[i - self.window : i])).unsqueeze(0),
            "time_enc": torch.from_numpy(np.array(self.time_enc[i - self.window : i])).unsqueeze(0),
            "edge_features": self._get_edge_window(i - 1).unsqueeze(0),
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
    def __init__(self, npz_path, window_1h = 96, assets = ASSETS):
        self.window_1h = window_1h
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

        self.T_15m = self.features_15m.shape[0]
        self.T_30m = self.features_30m.shape[0]
        self.T_1h = self.features_1h.shape[0]
        self.F_15m = self.features_15m.shape[2]
        self.F_30m = self.features_30m.shape[2]
        self.F_1h = self.features_1h.shape[2]
        self.D_time_15m = self.time_enc_15m.shape[1]
        self.D_time_30m = self.time_enc_30m.shape[1]
        self.D_time_1h = self.time_enc_1h.shape[1]
        self.F_model1 = self.model1_outputs.shape[1]
        self.F_sentiment = self.sentiment_scores.shape[2]

        sizes_15m = self.window_idx_15m[:, 1] - self.window_idx_15m[:, 0]
        sizes_30m = self.window_idx_30m[:, 1] - self.window_idx_30m[:, 0]
        self.lookback_15m = int(sizes_15m.max())
        self.lookback_30m = int(sizes_30m.max())

        self.start_idx = self.window_1h

        assert self.features_1h.shape == (self.T_1h, self.N, self.F_1h)
        assert self.targets.shape == (self.T_1h, self.N, 3)
        assert self.window_idx_15m.shape == (self.T_1h, 2)
        assert self.window_idx_30m.shape == (self.T_1h, 2)
        graph = np.load(os.path.join(BASE_DIR, "graph_edges.npz"))
        graph_times = graph["times"]
        self.edge_features = graph["edge_features"].astype(np.float32)
        self.F_edge = self.edge_features.shape[3]
        self.graph_idx = np.searchsorted(graph_times, self.times_1h)
        assert np.all(graph_times[self.graph_idx] == self.times_1h), (
            "graph_edges.npz timestamps do not align with model2_dataset")
        
        assert self.model1_outputs.shape == (self.T_1h, self.F_model1)
        assert self.sentiment_scores.shape == (self.T_1h, self.N, self.F_sentiment)
        assert self.sentiment_missing.shape == (self.T_1h,)
        assert self.window_idx_15m[:, 0].min() >= 0
        assert self.window_idx_15m[:, 1].max() <= self.T_15m
        assert self.window_idx_30m[:, 0].min() >= 0
        assert self.window_idx_30m[:, 1].max() <= self.T_30m
        assert self.T_1h > self.start_idx, "fewer 1h bars than lookback window"

    def __len__(self):
        return self.T_1h - self.start_idx

    def _pad_to(self, arr, start, end, target_len):
        window = np.array(arr[start:end], dtype = np.float32)
        actual = window.shape[0]
        if actual == target_len:
            return window
        pad = np.zeros((target_len - actual,) + window.shape[1:], dtype = np.float32)
        return np.concatenate([pad, window], axis = 0)

    def __getitem__(self, idx):
        i = idx + self.start_idx
        s15 = int(self.window_idx_15m[i, 0])
        e15 = int(self.window_idx_15m[i, 1])
        s30 = int(self.window_idx_30m[i, 0])
        e30 = int(self.window_idx_30m[i, 1])

        feat_15m = self._pad_to(self.features_15m, s15, e15, self.lookback_15m)
        enc_15m = self._pad_to(self.time_enc_15m, s15, e15, self.lookback_15m)
        feat_30m = self._pad_to(self.features_30m, s30, e30, self.lookback_30m)
        enc_30m = self._pad_to(self.time_enc_30m, s30, e30, self.lookback_30m)

        return {
            "features_15m": torch.from_numpy(feat_15m),
            "features_30m": torch.from_numpy(feat_30m),
            "features_1h": torch.from_numpy(np.array(self.features_1h[i - self.window_1h : i])),
            "time_enc_15m": torch.from_numpy(enc_15m),
            "time_enc_30m": torch.from_numpy(enc_30m),
            "time_enc_1h": torch.from_numpy(np.array(self.time_enc_1h[i - self.window_1h : i])),
            "edge_features": torch.from_numpy(np.array(self.edge_features[self.graph_idx[i]])),
            "model1_outputs": torch.from_numpy(np.array(self.model1_outputs[i])),
            "sentiment_scores": torch.from_numpy(np.array(self.sentiment_scores[i])),
            "sentiment_missing": torch.tensor(float(self.sentiment_missing[i])),
            "targets": torch.from_numpy(np.array(self.targets[i])),
        }

    def get_current_sample(self):
        i = self.T_1h
        s15 = int(self.window_idx_15m[i - 1, 0])
        e15 = int(self.window_idx_15m[i - 1, 1])
        s30 = int(self.window_idx_30m[i - 1, 0])
        e30 = int(self.window_idx_30m[i - 1, 1])

        feat_15m = self._pad_to(self.features_15m, s15, e15, self.lookback_15m)
        enc_15m = self._pad_to(self.time_enc_15m, s15, e15, self.lookback_15m)
        feat_30m = self._pad_to(self.features_30m, s30, e30, self.lookback_30m)
        enc_30m = self._pad_to(self.time_enc_30m, s30, e30, self.lookback_30m)

        return {
            "features_15m": torch.from_numpy(feat_15m).unsqueeze(0),
            "features_30m": torch.from_numpy(feat_30m).unsqueeze(0),
            "features_1h": torch.from_numpy(np.array(self.features_1h[i - self.window_1h : i])).unsqueeze(0),
            "time_enc_15m": torch.from_numpy(enc_15m).unsqueeze(0),
            "time_enc_30m": torch.from_numpy(enc_30m).unsqueeze(0),
            "time_enc_1h": torch.from_numpy(np.array(self.time_enc_1h[i - self.window_1h : i])).unsqueeze(0),
            "edge_features": torch.from_numpy(np.array(self.edge_features[self.graph_idx[i - 1]])).unsqueeze(0),
            "model1_outputs": torch.from_numpy(np.array(self.model1_outputs[i - 1])).unsqueeze(0),
            "sentiment_scores": torch.from_numpy(np.array(self.sentiment_scores[i - 1])).unsqueeze(0),
            "sentiment_missing": torch.tensor([float(self.sentiment_missing[i - 1])]),
        }

    def set_model1_outputs(self, outputs):
        assert outputs.shape == self.model1_outputs.shape, (
            f"shape mismatch: got {outputs.shape}, expected {self.model1_outputs.shape}")
        self.model1_outputs = outputs.astype(np.float32)

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


if __name__ == "__main__":
    m1 = Model1Dataset(os.path.join(BASE_DIR, "model1_dataset.npz"), window = 72, graph_window = 8)
    print(f"Model1  T={m1.T}  F_node={m1.F_node}  F_global={m1.F_global}  F_btc_raw={m1.F_btc_raw}  D_time={m1.D_time}")
    print(f"Model1  samples={len(m1)}")

    sample = m1[0]
    print(f"Model1  sample keys: {list(sample.keys())}")
    print(f"Model1  node_features shape: {sample['node_features'].shape}")
    print(f"Model1  edge_features shape: {sample['edge_features'].shape}")
    print(f"Model1  regime_label: {sample['regime_label']}")
    print(f"Model1  transition_label: {sample['transition_label']}")

    train_sub, val_sub, test_sub = m1.chronological_split()
    print(f"Model1  split sizes: train={len(train_sub)}  val={len(val_sub)}  test={len(test_sub)}")

    live = m1.get_current_sample()
    print(f"Model1  live node_features shape: {live['node_features'].shape}")

    print()

    m2 = Model2Dataset(os.path.join(BASE_DIR, "model2_dataset.npz"), window_1h = 96)
    print(f"Model2  T_1h={m2.T_1h}  T_30m={m2.T_30m}  T_15m={m2.T_15m}")
    print(f"Model2  F_1h={m2.F_1h}  F_30m={m2.F_30m}  F_15m={m2.F_15m}")
    print(f"Model2  lookback_15m={m2.lookback_15m}  lookback_30m={m2.lookback_30m}")
    print(f"Model2  F_model1={m2.F_model1}  samples={len(m2)}")

    sample2 = m2[0]
    print(f"Model2  sample keys: {list(sample2.keys())}")
    print(f"Model2  features_15m shape: {sample2['features_15m'].shape}")
    print(f"Model2  features_30m shape: {sample2['features_30m'].shape}")
    print(f"Model2  features_1h shape: {sample2['features_1h'].shape}")
    print(f"Model2  targets shape: {sample2['targets'].shape}")
    print(f"Model2  model1_outputs shape: {sample2['model1_outputs'].shape}")

    train_sub2, val_sub2, test_sub2 = m2.chronological_split()
    print(f"Model2  split sizes: train={len(train_sub2)}  val={len(val_sub2)}  test={len(test_sub2)}")

    live2 = m2.get_current_sample()
    print(f"Model2  live features_1h shape: {live2['features_1h'].shape}")