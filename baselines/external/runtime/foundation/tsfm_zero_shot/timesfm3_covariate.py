# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from foundation.tsfm_zero_shot.probabilistic_outputs import DEFAULT_QUANTILE_LEVELS, quantile_index, validate_quantile_levels

TIMESFM3_DEFAULT_QUANTILE_LEVELS = DEFAULT_QUANTILE_LEVELS


def _as_array(value: Any, *, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _output_quantiles(output: Any) -> np.ndarray:
    """Return one TimesFM-3 output's quantiles as ``[target, H, Q]``."""
    values = getattr(output, "quantiles", None)
    if values is None and isinstance(output, dict):
        values = output.get("quantiles")
    if values is None:
        raise ValueError("TimesFM-3 output does not contain quantile forecasts.")
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Unexpected TimesFM-3 quantile output shape={array.shape}.")
    return array


def build_timesfm3_input(
    item: dict[str, Any],
    *,
    covariate_mode: str = "era5_nwp",
) -> dict[str, np.ndarray | None]:
    """Map one frozen PVFM cache item to the TimesFM-3 public input shapes.

    ``era5_nwp`` creates one past-future dynamic covariate channel per weather
    variable.  Its context portion is historical weather and its future portion
    is NWP, so the model receives exactly the same information available to the
    other covariate-aware zero-shot evaluators.  ``none`` is a target-only
    diagnostic mode and is not the main comparison protocol.

    No values are imputed here.  In particular, NaNs are preserved for the
    official TimesFM-3 implementation to handle according to its native API.
    """
    if covariate_mode not in {"era5_nwp", "none"}:
        raise ValueError(f"Unsupported TimesFM-3 covariate mode: {covariate_mode!r}.")

    metadata = item["metadata"]
    history_cols = tuple(metadata["history_covariate_cols"])
    future_cols = tuple(metadata["future_covariate_cols"])
    if covariate_mode == "era5_nwp" and (not history_cols or history_cols != future_cols):
        raise ValueError(
            "TimesFM-3 era5_nwp mode requires matching non-empty history and "
            f"future covariate columns; got history={history_cols}, future={future_cols}."
        )

    target = _as_array(item["past_target"]).reshape(-1)
    history = _as_array(item["historical_covariates"])
    future = _as_array(item["future_covariates"])
    past_mask = _as_array(item["past_observed_mask"])
    history_mask = _as_array(item["historical_covariates_mask"])
    future_mask = _as_array(item["future_covariates_mask"])

    if target.ndim != 1 or history.ndim != 2 or future.ndim != 2:
        raise ValueError("PVFM cache item has unexpected target/covariate rank.")
    if history.shape != (len(target), len(history_cols)):
        raise ValueError(
            "Historical covariate shape does not match target and metadata: "
            f"target={target.shape}, history={history.shape}, columns={len(history_cols)}."
        )
    if future.shape != (int(item["pred_len"]), len(future_cols)):
        raise ValueError(
            "Future covariate shape does not match task horizon and metadata: "
            f"future={future.shape}, pred_len={item['pred_len']}, columns={len(future_cols)}."
        )
    if past_mask.reshape(-1).shape != target.shape:
        raise ValueError("Past target mask shape does not match past target.")
    if history_mask.shape != history.shape or future_mask.shape != future.shape:
        raise ValueError("Covariate mask shapes do not match their source arrays.")

    past_future_covariates: np.ndarray | None = None
    if covariate_mode == "era5_nwp":
        # TimesFM-3 expects [covariate_channel, context + horizon].
        past_future_covariates = np.concatenate((history.T, future.T), axis=1)

    return {
        "context": target[None, :].copy(),
        "past_only_covariates": None,
        "past_future_covariates": past_future_covariates,
    }


def quantile_forecasts(
    outputs: Iterable[Any],
    available_quantiles: Iterable[float],
    requested_quantiles: Iterable[float],
    prediction_length: int,
) -> np.ndarray:
    """Extract TimesFM-3 quantiles as ``[batch, horizon, quantile]``.

    The public TimesFM-3 output is either ``[H, Q]`` for a single target or
    ``[target, H, Q]`` for a multivariate query.  The PVFM adapter always asks
    for one target and therefore selects target index zero when needed.
    """
    available = validate_quantile_levels(available_quantiles)
    requested = validate_quantile_levels(requested_quantiles)
    indices = [quantile_index(available, level) for level in requested]
    rows: list[np.ndarray] = []
    for output in outputs:
        array = _output_quantiles(output)
        if array.shape[0] != 1 or array.shape[1] != int(prediction_length):
            raise ValueError(
                "TimesFM-3 returned an unexpected target/horizon shape: "
                f"{array.shape}; expected (1, {prediction_length}, {len(available)})."
            )
        # Slice the horizon first, then apply the quantile index list.  Using
        # two advanced indices in one expression would transpose the axes.
        row = np.asarray(array[0][:, indices], dtype=np.float64)
        if row.shape != (int(prediction_length), len(requested)):
            raise ValueError(f"Invalid TimesFM-3 quantile shape after selection: {row.shape}.")
        rows.append(row)
    if not rows:
        return np.empty((0, int(prediction_length), len(requested)), dtype=np.float64)
    return np.stack(rows, axis=0)


def model_quantile_levels(model: Any) -> tuple[float, ...]:
    """Discover the loaded model's quantile grid without importing TimesFM-3."""
    candidates = [
        getattr(model, "quantiles", None),
        getattr(getattr(model, "config", None), "quantiles", None),
        getattr(getattr(model, "model", None), "quantiles", None),
        getattr(getattr(getattr(model, "model", None), "config", None), "quantiles", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return validate_quantile_levels(candidate)
    return TIMESFM3_DEFAULT_QUANTILE_LEVELS

