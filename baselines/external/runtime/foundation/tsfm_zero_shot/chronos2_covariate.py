# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any

import numpy as np

from foundation.tsfm_zero_shot.probabilistic_outputs import quantile_index, validate_quantile_levels

def build_chronos2_input(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one exact cache window to Chronos-2 target/covariate inputs.

    ``future_covariates`` is intentionally populated only from PVFM NWP values
    starting immediately after the cutoff.  It never contains labels.
    """
    metadata = item["metadata"]
    history_cols = tuple(metadata["history_covariate_cols"])
    future_cols = tuple(metadata["future_covariate_cols"])
    if not history_cols or history_cols != future_cols:
        raise ValueError(
            "Chronos-2 B-setting requires matching non-empty history ERA5 and "
            f"future NWP column names; got history={history_cols}, future={future_cols}."
        )

    target = np.asarray(item["past_target"], dtype=np.float32).reshape(-1)
    history = np.asarray(item["historical_covariates"], dtype=np.float32)
    future = np.asarray(item["future_covariates"], dtype=np.float32)
    target_mask = np.asarray(item["past_observed_mask"], dtype=np.float32).reshape(-1)
    history_mask = np.asarray(item["historical_covariates_mask"], dtype=np.float32)
    future_mask = np.asarray(item["future_covariates_mask"], dtype=np.float32)

    if target.ndim != 1 or history.ndim != 2 or future.ndim != 2:
        raise ValueError("PVFM cache item has unexpected target/covariate rank.")
    if len(target) != history.shape[0] or history.shape[1] != len(history_cols):
        raise ValueError("History target/covariate dimensions do not match cache metadata.")
    if future.shape != (int(item["pred_len"]), len(future_cols)):
        raise ValueError("Future NWP shape does not match this task horizon and cache metadata.")
    if not (
        np.all(target_mask > 0)
        and np.all(history_mask > 0)
        and np.all(future_mask > 0)
        and np.isfinite(target).all()
        and np.isfinite(history).all()
        and np.isfinite(future).all()
    ):
        raise ValueError("Strict PVFM QC invariant failed: Chronos input contains an invalid value or mask.")

    past_covariates = {name: history[:, idx].copy() for idx, name in enumerate(history_cols)}
    future_covariates = {name: future[:, idx].copy() for idx, name in enumerate(future_cols)}
    return {
        "target": target.copy(),
        "past_covariates": past_covariates,
        "future_covariates": future_covariates,
    }


def quantile_forecasts(
    predictions: list[Any],
    available_quantiles: list[float],
    requested_quantiles: list[float] | tuple[float, ...],
    prediction_length: int,
) -> np.ndarray:
    """Extract requested Chronos-2 quantiles as ``[batch, horizon, quantile]``."""
    requested = validate_quantile_levels(requested_quantiles)
    available = tuple(float(level) for level in available_quantiles)
    indices = [quantile_index(available, level) for level in requested]
    rows = []
    for prediction in predictions:
        array = prediction.detach().float().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != len(available):
            raise ValueError(f"Unexpected Chronos-2 prediction shape {array.shape}; expected (1, n_quantiles, H).")
        row = np.asarray(array[0, indices, :], dtype=np.float64).T
        if row.shape != (prediction_length, len(requested)) or not np.isfinite(row).all():
            raise ValueError("Chronos-2 returned an invalid quantile forecast.")
        rows.append(row)
    return np.stack(rows, axis=0)

