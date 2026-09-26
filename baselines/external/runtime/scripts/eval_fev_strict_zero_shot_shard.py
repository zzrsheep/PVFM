# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any

import numpy as np

from foundation.tsfm_zero_shot.fev_frozen_windows import FrozenWindowTask

QUANTILE_LEVELS = tuple(float(q) for q in FrozenWindowTask.quantile_levels)


def _extract_forecast_column(
    target_predictions: Any,
    key: str,
    batch_size: int,
    horizon: int,
) -> np.ndarray:
    try:
        values = target_predictions[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"FEV wrapper output is missing forecast column {key!r}.") from error
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (horizon,) and batch_size == 1:
        array = array.reshape(1, horizon)
    if array.shape != (batch_size, horizon):
        raise ValueError(
            f"Unexpected FEV forecast shape for {key!r}: {array.shape}; "
            f"expected {(batch_size, horizon)}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"FEV wrapper returned a non-finite forecast for {key!r}.")
    return array


def extract_forecasts(
    prediction_dict: Any,
    batch_size: int,
    horizon: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract point and all official FEV quantile forecasts.

    FEV returns a DatasetDict keyed by target.  For the single PV target, the
    target dataset contains ``predictions`` plus one column per requested
    quantile.  We deliberately retain the complete probabilistic output even
    though the primary station metrics currently use the point forecast.
    """
    try:
        target_predictions = prediction_dict["target"]
    except (KeyError, TypeError) as error:
        raise ValueError("FEV wrapper output is missing the target prediction dataset.") from error

    point = _extract_forecast_column(target_predictions, "predictions", batch_size, horizon)
    quantiles = {
        str(q): _extract_forecast_column(target_predictions, str(q), batch_size, horizon)
        for q in QUANTILE_LEVELS
    }
    return point, quantiles

