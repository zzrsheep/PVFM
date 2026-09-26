from pathlib import Path
from types import SimpleNamespace
import numpy as np


def _task(name, resolution, seq, pred):
    return SimpleNamespace(
        task_name=name,
        seq_len=seq,
        pred_len=pred,
        task_spec=SimpleNamespace(resolution=resolution),
        min_history_covariate_valid_ratio=1.0,
        min_future_covariate_valid_ratio=1.0,
        min_history_covariate_std=0.0,
        min_future_covariate_std=0.0,
    )


class _Child:
    def __init__(self, root, task, resolution, seq, pred, n=600):
        self.binary_dataset_dir = str(root)
        self.window_index_root = ""
        self.task_datasets = [_task(task, resolution, seq, pred)]
        self.station_records = [
            {"station_dir": "s0", "station_key": "s0", "region": "A"}
        ]
        self.task_datasets[0].station_records = self.station_records
        step = {"1h": 3600, "15min": 900}[resolution]
        timestamps = (
            np.datetime64("2020-01-01", "s")
            + np.arange(n, dtype="timedelta64[s]") * step
        )
        self._arrays = {
            0: {
                "target": np.ones((n, 1), np.float32),
                "target_mask": np.ones((n, 1), np.float32),
                "history_covariates": np.zeros((n, 0), np.float32),
                "history_covariate_mask": np.ones((n, 0), np.float32),
                "future_covariates": np.zeros((n, 0), np.float32),
                "future_covariate_mask": np.ones((n, 0), np.float32),
                "time_features": np.zeros((n, 4), np.float32),
                "timestamps": timestamps,
                "site_features": np.ones(5, np.float32),
            }
        }
        self.index = [(0, i, 0, i) for i in range(20, 80)]

    def _station_arrays(self, _resolution, _station):
        return self._arrays[0]

    def _station_dir(self, _station):
        return self.station_records[_station]["station_dir"]

    def _station_key(self, _station):
        return self.station_records[_station]["station_key"]


class _Parent:
    def __init__(self, root, n=600):
        self.children = [
            _Child(Path(root) / "route", "pv_6h_ahead", "1h", 24, 6, n=n),
            _Child(Path(root) / "origin", "pv_24h_ahead", "1h", 72, 24, n=n),
            _Child(Path(root) / "native", "pv_15min_24h_ahead", "15min", 192, 96, n=n),
        ]
        self.task_datasets = [c.task_datasets[0] for c in self.children]
        self.index = []
        self.region_keys = []
        for child, item in enumerate(self.children):
            for local, (_x, _y, station, start) in enumerate(item.index):
                self.index.append((child, local, station, start))
                self.region_keys.append("A")
        self.source_cache_names_by_child = ["cache", "cache", "cache15"]
        self.source_cache_names = ["cache", "cache15"]
        self.source_cache_weights = {"cache": 1.0, "cache15": 1.0}
        self.cache_sampling_mode = "weighted"
        self.split = "train"
        self.model_dims = {
            "target_dim": 1,
            "historical_covariate_dim": 0,
            "future_covariate_dim": 0,
            "past_time_dim": 4,
            "future_time_dim": 4,
            "static_dim": 4,
            "output_dim": 1,
        }
        self.context_max_hours = 672

    def _source_dataset(self, index):
        return self.children[int(index)]
