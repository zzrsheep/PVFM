"""Deterministic synthetic station arrays for sampler equivalence checks."""

from pathlib import Path

import numpy as np

from tests.fixtures import _Child, _Parent


class AuditChild(_Child):
    def __init__(self, root, task, resolution, seq, pred):
        super().__init__(root, task, resolution, seq, pred, n=2048)
        self.station_records = []
        self._arrays = {}
        self._allowed_station_indices = list(range(16))
        for station in range(16):
            region = f"R{station // 2:02d}"
            name = f"source/{region}/station_{station:02d}"
            self.station_records.append(
                {"station_dir": name, "station_key": name, "region": region}
            )
            n = 2048 + 17 * station
            x = np.arange(n, dtype=np.float32)
            step = 900 if resolution == "15min" else 3600
            stamps = np.datetime64("2020-01-01", "s") + np.arange(n) * np.timedelta64(
                step, "s"
            )
            # A cadence break and target/weather gaps exercise the quality index.
            stamps[1100:] += np.timedelta64(step, "s")
            mask = np.ones((n, 1), dtype=np.float32)
            mask[1010:1013] = 0
            weather_mask = np.ones((n, 6), dtype=np.float32)
            weather_mask[1040:1042] = 0
            weather = np.stack([np.sin(x / (13 + j)) for j in range(6)], axis=1)
            self._arrays[station] = {
                "target": (np.sin(x / 23) ** 2)[:, None],
                "target_mask": mask,
                "history_covariates": weather,
                "history_covariate_mask": weather_mask,
                "future_covariates": weather * np.float32(0.75),
                "future_covariate_mask": weather_mask.copy(),
                "time_features": np.tile(
                    np.array([0, 12, 0, 1, 1], np.float32), (n, 1)
                ),
                "timestamps": stamps,
                "site_features": np.array([-30, 150, 0, 1, 10], np.float32),
            }
        self.task_datasets[0].station_records = self.station_records

    def _station_arrays(self, _resolution, station):
        return self._arrays[int(station)]


class AuditParent(_Parent):
    def __init__(self, root):
        super().__init__(root)
        self.children = [
            AuditChild(Path(root) / "route", "pv_6h_ahead", "1h", 24, 6),
            AuditChild(Path(root) / "origin", "pv_24h_ahead", "1h", 72, 24),
            AuditChild(Path(root) / "native", "pv_15min_24h_ahead", "15min", 192, 96),
        ]
        self.task_datasets = [c.task_datasets[0] for c in self.children]
        self.model_dims.update(
            historical_covariate_dim=6,
            future_covariate_dim=6,
            past_time_dim=5,
            future_time_dim=5,
        )
        self.time_feature_schema = "minute_hour_weekday_day_dayofyear_v1"
