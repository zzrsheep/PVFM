# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any

import numpy as np

import pandas as pd

def make_fev_window(item: dict[str, Any], window_id: str):
    """Create an FEV window with PV labels hidden after the frozen cutoff."""
    import datasets
    from fev.task import EvaluationWindow

    metadata = item["metadata"]
    history_columns = tuple(metadata["history_covariate_cols"])
    future_columns = tuple(metadata["future_covariate_cols"])
    if not history_columns or history_columns != future_columns:
        raise ValueError("Expected matching non-empty historical ERA5 and future NWP columns.")

    past_pv = np.asarray(item["past_target"], dtype=np.float32).reshape(-1)
    future_pv = np.asarray(item["future_target"], dtype=np.float32).reshape(-1)
    history = np.asarray(item["historical_covariates"], dtype=np.float32)
    future = np.asarray(item["future_covariates"], dtype=np.float32)
    past_mask = np.asarray(item["past_observed_mask"], dtype=np.float32).reshape(-1)
    future_mask = np.asarray(item["future_observed_mask"], dtype=np.float32).reshape(-1)
    history_mask = np.asarray(item["historical_covariates_mask"], dtype=np.float32)
    future_mask_cov = np.asarray(item["future_covariates_mask"], dtype=np.float32)
    timestamps = np.asarray(metadata["timestamps"])

    if history.shape != (len(past_pv), len(history_columns)) or future.shape != (len(future_pv), len(history_columns)):
        raise ValueError("Frozen PVFM cache tensor shapes do not match the task contract.")
    if len(timestamps) != len(past_pv) + len(future_pv):
        raise ValueError("Frozen PVFM timestamps do not cover the exact C+H window.")
    if not (
        np.all(past_mask > 0)
        and np.all(future_mask > 0)
        and np.all(history_mask > 0)
        and np.all(future_mask_cov > 0)
        and np.isfinite(past_pv).all()
        and np.isfinite(future_pv).all()
        and np.isfinite(history).all()
        and np.isfinite(future).all()
    ):
        raise ValueError("Strict frozen-window QC failed: an input, label, or mask is invalid.")

    # The full dataset contains labels only so FEV can score them. EvaluationWindow
    # exposes target values only before cutoff to the model.
    full_covariates = np.concatenate([history, future], axis=0)
    data: dict[str, list] = {
        "id": [window_id],
        "timestamp": [pd.to_datetime(timestamps).to_pydatetime().tolist()],
        "target": [np.concatenate([past_pv, future_pv]).tolist()],
    }
    for column_index, column in enumerate(history_columns):
        data[column] = [full_covariates[:, column_index].tolist()]
    return EvaluationWindow(
        full_dataset=datasets.Dataset.from_dict(data),
        cutoff=len(past_pv),
        horizon=len(future_pv),
        min_context_length=len(past_pv),
        max_context_length=len(past_pv),
        id_column="id",
        timestamp_column="timestamp",
        target_columns=["target"],
        known_dynamic_columns=list(history_columns),
        past_dynamic_columns=[],
        static_columns=[],
    )


class FrozenWindowTask:
    """Minimal FEV task facade for a homogeneous list of frozen PVFM windows."""

    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    target_columns = ["target"]
    past_dynamic_columns: list[str] = []

    def __init__(self, windows: list[Any]) -> None:
        if not windows:
            raise ValueError("FrozenWindowTask requires at least one window.")
        self._windows = windows
        self.horizon = int(windows[0].horizon)
        self.known_dynamic_columns = list(windows[0].known_dynamic_columns)
        if any(int(window.horizon) != self.horizon for window in windows):
            raise ValueError("One FrozenWindowTask must contain a single horizon.")

    def iter_windows(self):
        yield from self._windows

    def get_window(self, index: int):
        return self._windows[index]

