# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any

import numpy as np

from foundation.tsfm_zero_shot.fev_frozen_windows import FrozenWindowTask

from foundation.tsfm_zero_shot.probabilistic_outputs import quantile_key, validate_quantile_levels

QUANTILE_LEVELS = validate_quantile_levels(FrozenWindowTask.quantile_levels)


def extract_prediction(predictions: Any, horizon: int) -> np.ndarray:
    array = np.asarray(predictions["target"]["predictions"], dtype=np.float64)
    if array.shape == (horizon,):
        return array
    if array.shape == (1, horizon):
        return array[0]
    raise ValueError(f"Unexpected FEV TabPFN prediction shape {array.shape}; expected ({horizon},).")


def extract_forecasts(predictions: Any, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    point = extract_prediction(predictions, horizon)
    target = predictions["target"]
    columns = []
    for level in QUANTILE_LEVELS:
        array = np.asarray(target[quantile_key(level)], dtype=np.float64)
        if array.shape == (1, horizon):
            array = array[0]
        if array.shape != (horizon,) or not np.isfinite(array).all():
            raise ValueError(f"Unexpected FEV TabPFN q{level:g} shape/value: {array.shape}.")
        columns.append(array)
    quantiles = np.stack(columns, axis=-1)
    if not np.isfinite(point).all():
        raise ValueError("FEV TabPFN point forecast contains a non-finite value.")
    return point, quantiles

