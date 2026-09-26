# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import pandas as pd

def make_batched_fev_window(
    items: Sequence[dict[str, Any]],
    window_ids: Sequence[str],
    *,
    include_covariates: bool,
):
    """Pack homogeneous frozen items into one vectorized FEV window."""
    import datasets
    from fev.task import EvaluationWindow

    if not items or len(items) != len(window_ids):
        raise ValueError("items and window_ids must have the same positive length.")

    first = items[0]
    context_length = int(first["seq_len"])
    horizon = int(first["pred_len"])
    first_metadata = first["metadata"]
    history_columns = tuple(first_metadata["history_covariate_cols"])
    future_columns = tuple(first_metadata["future_covariate_cols"])
    if include_covariates and (not history_columns or history_columns != future_columns):
        raise ValueError("Expected matching non-empty historical ERA5 and future NWP columns.")

    data: dict[str, list[Any]] = {
        "id": [],
        "timestamp": [],
        "target": [],
    }
    if include_covariates:
        data.update({column: [] for column in history_columns})

    for item, window_id in zip(items, window_ids, strict=True):
        if int(item["seq_len"]) != context_length or int(item["pred_len"]) != horizon:
            raise ValueError("One batched FEV window must use a single context/horizon shape.")
        metadata = item["metadata"]
        if tuple(metadata["history_covariate_cols"]) != history_columns:
            raise ValueError("Historical covariate schema differs inside one batch.")
        if tuple(metadata["future_covariate_cols"]) != future_columns:
            raise ValueError("Future covariate schema differs inside one batch.")

        past_pv = np.asarray(item["past_target"], dtype=np.float32).reshape(-1)
        future_pv = np.asarray(item["future_target"], dtype=np.float32).reshape(-1)
        past_mask = np.asarray(item["past_observed_mask"], dtype=np.float32).reshape(-1)
        future_mask = np.asarray(item["future_observed_mask"], dtype=np.float32).reshape(-1)
        timestamps = np.asarray(metadata["timestamps"])
        if len(past_pv) != context_length or len(future_pv) != horizon:
            raise ValueError("Frozen target tensors do not match the task contract.")
        if len(timestamps) != context_length + horizon:
            raise ValueError("Frozen timestamps do not cover the exact C+H window.")
        if not (
            np.all(past_mask > 0)
            and np.all(future_mask > 0)
            and np.isfinite(past_pv).all()
            and np.isfinite(future_pv).all()
        ):
            raise ValueError("Strict target QC failed for a frozen window.")

        data["id"].append(str(window_id))
        data["timestamp"].append(pd.to_datetime(timestamps).to_pydatetime().tolist())
        data["target"].append(np.concatenate([past_pv, future_pv]).tolist())

        if include_covariates:
            history = np.asarray(item["historical_covariates"], dtype=np.float32)
            future = np.asarray(item["future_covariates"], dtype=np.float32)
            history_mask = np.asarray(item["historical_covariates_mask"], dtype=np.float32)
            future_mask_cov = np.asarray(item["future_covariates_mask"], dtype=np.float32)
            expected_history = (context_length, len(history_columns))
            expected_future = (horizon, len(history_columns))
            if history.shape != expected_history or future.shape != expected_future:
                raise ValueError("Frozen covariate tensors do not match the task contract.")
            if not (
                np.all(history_mask > 0)
                and np.all(future_mask_cov > 0)
                and np.isfinite(history).all()
                and np.isfinite(future).all()
            ):
                raise ValueError("Strict covariate QC failed for a frozen window.")
            full_covariates = np.concatenate([history, future], axis=0)
            for column_index, column in enumerate(history_columns):
                data[column].append(full_covariates[:, column_index].tolist())

    return EvaluationWindow(
        full_dataset=datasets.Dataset.from_dict(data),
        cutoff=context_length,
        horizon=horizon,
        min_context_length=context_length,
        max_context_length=context_length,
        id_column="id",
        timestamp_column="timestamp",
        target_columns=["target"],
        known_dynamic_columns=list(history_columns) if include_covariates else [],
        past_dynamic_columns=[],
        static_columns=[],
    )

