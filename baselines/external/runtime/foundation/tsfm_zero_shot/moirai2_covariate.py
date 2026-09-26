# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Any

import numpy as np

import pandas as pd

from foundation.tsfm_zero_shot.probabilistic_outputs import validate_quantile_levels

ITEM_ID_COLUMN = "item_id"


TIMESTAMP_COLUMN = "timestamp"


TARGET_COLUMN = "target"


def _validated_arrays(item: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact PVFM arrays after enforcing the frozen-window QC contract."""
    metadata = item["metadata"]
    history_cols = tuple(metadata["history_covariate_cols"])
    future_cols = tuple(metadata["future_covariate_cols"])
    if not history_cols or history_cols != future_cols:
        raise ValueError(
            "Moirai-2 B-setting requires matching non-empty history ERA5 and "
            f"future NWP column names; got history={history_cols}, future={future_cols}."
        )

    target = np.asarray(item["past_target"], dtype=np.float32).reshape(-1)
    history = np.asarray(item["historical_covariates"], dtype=np.float32)
    future = np.asarray(item["future_covariates"], dtype=np.float32)
    target_mask = np.asarray(item["past_observed_mask"], dtype=np.float32).reshape(-1)
    history_mask = np.asarray(item["historical_covariates_mask"], dtype=np.float32)
    future_mask = np.asarray(item["future_covariates_mask"], dtype=np.float32)
    timestamps = np.asarray(metadata["timestamps"])

    if target.ndim != 1 or history.ndim != 2 or future.ndim != 2:
        raise ValueError("PVFM cache item has unexpected target/covariate rank.")
    if len(target) != history.shape[0] or history.shape[1] != len(history_cols):
        raise ValueError("History target/covariate dimensions do not match cache metadata.")
    if future.shape != (int(item["pred_len"]), len(future_cols)):
        raise ValueError("Future NWP shape does not match this task horizon and cache metadata.")
    if len(timestamps) != len(target) + len(future):
        raise ValueError("PVFM metadata timestamps do not cover the exact C+H window.")
    if not (
        np.all(target_mask > 0)
        and np.all(history_mask > 0)
        and np.all(future_mask > 0)
        and np.isfinite(target).all()
        and np.isfinite(history).all()
        and np.isfinite(future).all()
    ):
        raise ValueError("Strict PVFM QC invariant failed: Moirai input contains an invalid value or mask.")
    return history_cols, target, history, future, timestamps


def build_moirai2_frame(
    item: dict[str, Any], item_id: str, *, covariate_mode: str = "era5_nwp"
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Make one public GluonTS long-dataframe item without future PV labels.

    ``target`` contains PV through the cutoff only.  Its H future rows are NaN
    placeholders required by the public GluonTS dataset representation.  The
    only values available in those rows are the frozen future-NWP covariates.
    ``covariate_mode=none`` is an explicit diagnostic ablation: it retains the
    same target/timestamps/window QC while omitting every covariate channel.
    """
    covariate_cols, target, history, future, timestamps = _validated_arrays(item)
    horizon = len(future)
    frame = pd.DataFrame(
        {
            ITEM_ID_COLUMN: item_id,
            TIMESTAMP_COLUMN: pd.to_datetime(timestamps),
            TARGET_COLUMN: np.concatenate([target, np.full(horizon, np.nan, dtype=np.float32)]),
        }
    )
    if covariate_mode == "none":
        return frame, ()
    if covariate_mode != "era5_nwp":
        raise ValueError(f"Unsupported Moirai-2 covariate mode: {covariate_mode}")
    full_covariates = np.concatenate([history, future], axis=0)
    for index, column in enumerate(covariate_cols):
        frame[column] = full_covariates[:, index]
    return frame, covariate_cols


def build_moirai2_dataset(items: list[dict[str, Any]], *, covariate_mode: str = "era5_nwp"):
    """Convert a same-task batch to the exact public Uni2TS/GluonTS input.

    Returns ``(dataset, item_ids, covariate_columns)``.  All inputs in a batch
    must share C, H and covariate semantics, matching the frozen task contract.
    """
    if not items:
        raise ValueError("Cannot build a Moirai-2 dataset from an empty batch.")

    frames: list[pd.DataFrame] = []
    item_ids: list[str] = []
    common_columns: tuple[str, ...] | None = None
    context_length: int | None = None
    prediction_length: int | None = None
    frequency: str | None = None
    for index, item in enumerate(items):
        item_id = f"pvfm_window_{index:06d}"
        frame, covariate_cols = build_moirai2_frame(item, item_id, covariate_mode=covariate_mode)
        this_context = int(item["seq_len"])
        this_horizon = int(item["pred_len"])
        this_frequency = pd.infer_freq(pd.DatetimeIndex(frame[TIMESTAMP_COLUMN]))
        if this_frequency is None:
            raise ValueError("Moirai-2 input timestamps must have one uniform frequency.")
        if common_columns is None:
            common_columns = covariate_cols
            context_length = this_context
            prediction_length = this_horizon
            frequency = this_frequency
        elif (
            covariate_cols != common_columns
            or this_context != context_length
            or this_horizon != prediction_length
            or this_frequency != frequency
        ):
            raise ValueError("One Moirai-2 batch must contain a single frozen PVFM task/frequency contract.")
        frames.append(frame)
        item_ids.append(item_id)

    # Import lazily so reading this adapter never requires the isolated Moirai environment.
    from gluonts.dataset.pandas import PandasDataset

    dataset = PandasDataset.from_long_dataframe(
        pd.concat(frames, ignore_index=True),
        item_id=ITEM_ID_COLUMN,
        timestamp=TIMESTAMP_COLUMN,
        target=TARGET_COLUMN,
        freq=str(frequency),
        future_length=int(prediction_length),
        feat_dynamic_real=list(common_columns),
    )
    return dataset, item_ids, common_columns


def forecast_quantiles(
    forecasts: list[Any],
    quantile_levels: list[float] | tuple[float, ...],
    prediction_length: int,
) -> np.ndarray:
    """Extract Moirai quantiles as ``[batch, horizon, quantile]``."""
    levels = validate_quantile_levels(quantile_levels)
    rows = []
    for forecast in forecasts:
        columns = [np.asarray(forecast.quantile(level), dtype=np.float64).reshape(-1) for level in levels]
        if any(column.shape != (prediction_length,) for column in columns):
            raise ValueError("Moirai-2 returned a quantile forecast with an invalid horizon.")
        row = np.stack(columns, axis=-1)
        if not np.isfinite(row).all():
            raise ValueError("Moirai-2 returned a non-finite quantile forecast.")
        rows.append(row)
    return np.stack(rows, axis=0)

