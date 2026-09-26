# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from foundation.tsfm_zero_shot.probabilistic_outputs import DEFAULT_QUANTILE_LEVELS, quantile_index, validate_quantile_levels

CITRAS_DEFAULT_QUANTILE_LEVELS = DEFAULT_QUANTILE_LEVELS


def _as_array(value: Any, *, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def build_citras_input(
    item: dict[str, Any],
    *,
    covariate_mode: str = "era5_nwp",
) -> dict[str, np.ndarray | None]:
    """Map one frozen PVFM cache item to CITRAS tensor-like arrays.

    ``era5_nwp`` uses the six matching weather channels as known covariates:
    historical ERA5 occupies the first ``L`` rows and future NWP occupies the
    following ``H`` rows.  No target values from the forecast interval are
    included.  ``none`` is retained as an explicit target-only diagnostic.
    """
    mode = str(covariate_mode or "era5_nwp").strip().lower()
    if mode not in {"era5_nwp", "none"}:
        raise ValueError(f"Unsupported CITRAS covariate mode: {covariate_mode!r}.")

    metadata = item["metadata"]
    history_cols = tuple(metadata.get("history_covariate_cols", ()))
    future_cols = tuple(metadata.get("future_covariate_cols", ()))
    if mode == "era5_nwp" and (not history_cols or history_cols != future_cols):
        raise ValueError(
            "CITRAS era5_nwp mode requires matching non-empty history and "
            f"future covariate columns; got history={history_cols}, future={future_cols}."
        )

    target = _as_array(item["past_target"]).reshape(-1, 1)
    history = _as_array(item["historical_covariates"])
    future = _as_array(item["future_covariates"])
    past_mask = _as_array(item["past_observed_mask"]).reshape(-1)
    history_mask = _as_array(item["historical_covariates_mask"])
    future_mask = _as_array(item["future_covariates_mask"])

    if target.ndim != 2 or history.ndim != 2 or future.ndim != 2:
        raise ValueError("PVFM cache item has unexpected target/covariate rank.")
    if history.shape != (target.shape[0], len(history_cols)):
        raise ValueError(
            "Historical covariate shape does not match target and metadata: "
            f"target={target.shape}, history={history.shape}, columns={len(history_cols)}."
        )
    if future.shape != (int(item["pred_len"]), len(future_cols)):
        raise ValueError(
            "Future covariate shape does not match task horizon and metadata: "
            f"future={future.shape}, pred_len={item['pred_len']}, columns={len(future_cols)}."
        )
    if past_mask.shape != (target.shape[0],):
        raise ValueError("Past target mask shape does not match past target.")
    if history_mask.shape != history.shape or future_mask.shape != future.shape:
        raise ValueError("Covariate mask shapes do not match their source arrays.")
    if not (
        np.all(past_mask > 0)
        and np.all(history_mask > 0)
        and np.all(future_mask > 0)
        and np.isfinite(target).all()
        and np.isfinite(history).all()
        and np.isfinite(future).all()
    ):
        raise ValueError(
            "Strict PVFM QC invariant failed: CITRAS input contains an invalid "
            "value or mask."
        )

    known_covariates: np.ndarray | None = None
    if mode == "era5_nwp":
        known_covariates = np.concatenate((history, future), axis=0)

    return {
        "target": target.copy(),
        "observed_cov": None,
        "known_cov": None if known_covariates is None else known_covariates.copy(),
    }


def quantile_forecasts(
    values: Any,
    available_quantiles: Iterable[float],
    requested_quantiles: Iterable[float],
    prediction_length: int,
) -> np.ndarray:
    """Extract CITRAS quantiles as ``[batch, horizon, quantile]``.

    The outer PVFM adapter stacks public per-window outputs into ``[B, H, Ct, Q]``.
    PVFM is a scalar-target benchmark, so ``Ct=1`` is required and the target
    axis is removed here.
    """
    available = validate_quantile_levels(available_quantiles)
    requested = validate_quantile_levels(requested_quantiles)
    indices = [quantile_index(available, level) for level in requested]
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or array.shape[2] != 1:
        raise ValueError(
            "CITRAS quantiles must have shape [B,H,1,Q]; "
            f"got {array.shape}."
        )
    if array.shape[1] != int(prediction_length) or array.shape[3] != len(available):
        raise ValueError(
            "CITRAS returned an unexpected horizon/quantile shape: "
            f"{array.shape}; expected [B,{prediction_length},1,{len(available)}]."
        )
    selected = np.asarray(array[:, :, 0, :][:, :, indices], dtype=np.float64)
    if selected.shape != (array.shape[0], int(prediction_length), len(requested)):
        raise ValueError(f"Invalid CITRAS quantile shape after selection: {selected.shape}.")
    if not np.isfinite(selected).all():
        raise ValueError("CITRAS returned non-finite quantile forecasts.")
    return selected


def model_quantile_levels(model: Any) -> tuple[float, ...]:
    """Discover the quantile grid exposed by a loaded CITRAS model."""
    candidates = [
        getattr(model, "quantiles", None),
        getattr(getattr(model, "model", None), "quantiles", None),
        getattr(getattr(model, "config", None), "quantiles", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return validate_quantile_levels(candidate)
    return CITRAS_DEFAULT_QUANTILE_LEVELS

