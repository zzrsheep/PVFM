from __future__ import annotations

from collections import Counter

import numpy as np


RESOLUTION_TO_SECONDS = {
    "5min": 5 * 60,
    "15min": 15 * 60,
    "30min": 30 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}

WINDOW_QC_VERSION = "window_qc_pointwise_v1"


def _as_2d_mask(mask, length):
    if mask is None:
        return np.ones((int(length), 0), dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(bool, copy=False)


def _as_2d_values(values):
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _window_valid_ratio(mask_window):
    if mask_window.size == 0:
        return 1.0
    return float(np.count_nonzero(mask_window)) / float(mask_window.size)


def _observed_values_are_finite(values, mask):
    """Return whether every mask-observed value is finite."""
    values_2d = _as_2d_values(values)
    mask_2d = _as_2d_mask(mask, len(values_2d))
    if values_2d.shape[0] != mask_2d.shape[0] or values_2d.shape[1] < mask_2d.shape[1]:
        return False
    if mask_2d.shape[1] <= 0:
        return True
    observed = values_2d[:, : mask_2d.shape[1]][mask_2d]
    return bool(np.isfinite(observed).all())


def _pointwise_array_contract_reason(
    target_mask,
    target_values,
    history_covariate_mask,
    future_covariate_mask,
    history_covariate_values,
    future_covariate_values,
):
    """Return a stable reason for malformed inputs to the pointwise API.

    The historical bulk scanner receives arrays produced by the cache builder
    and intentionally keeps its old assumptions.  A dynamic shape plan is a
    new boundary, so it must reject malformed station views before slicing
    them rather than allowing an ``IndexError`` to escape from a worker.
    """
    arrays = {
        "target_mask": target_mask,
        "target_values": target_values,
        "history_covariate_mask": history_covariate_mask,
        "future_covariate_mask": future_covariate_mask,
        "history_covariate_values": history_covariate_values,
        "future_covariate_values": future_covariate_values,
    }
    try:
        raw = {key: np.asarray(value) for key, value in arrays.items()}
    except (OverflowError, TypeError, ValueError):
        return "array_contract"
    for key, value in raw.items():
        if value.ndim not in (1, 2):
            return f"{key}_rank"
        if value.shape[0] <= 0:
            return f"{key}_length"
    target_length = int(raw["target_mask"].shape[0])
    if int(raw["target_values"].shape[0]) != target_length:
        return "target_length"
    for key in ("history_covariate_mask", "future_covariate_mask"):
        if int(raw[key].shape[0]) != target_length:
            return f"{key}_length"
    for values_key, mask_key in (
        ("history_covariate_values", "history_covariate_mask"),
        ("future_covariate_values", "future_covariate_mask"),
    ):
        if int(raw[values_key].shape[0]) != int(raw[mask_key].shape[0]):
            return f"{values_key}_length"

    def width(value):
        return 1 if value.ndim == 1 else int(value.shape[1])

    if width(raw["target_values"]) < width(raw["target_mask"]):
        return "target_width"
    for values_key, mask_key in (
        ("history_covariate_values", "history_covariate_mask"),
        ("future_covariate_values", "future_covariate_mask"),
    ):
        if width(raw[values_key]) < width(raw[mask_key]):
            return f"{values_key}_width"
    for key in ("target_mask", "history_covariate_mask", "future_covariate_mask"):
        try:
            if not np.isfinite(raw[key].astype(np.float64, copy=False)).all():
                return f"{key}_nonfinite"
        except (OverflowError, TypeError, ValueError):
            return f"{key}_contract"
    return None


def _future_sequence_passes_qc(
    future_values,
    future_mask,
    *,
    min_future_target_std=0.0,
    min_future_target_range=0.0,
    zero_run_limit=0,
    constant_run_limit=0,
    future_zero_tolerance=1e-6,
    future_constant_tolerance=1e-6,
):
    future_mask_bool = _as_2d_mask(future_mask, len(future_mask))
    future_values_2d = _as_2d_values(future_values)
    valid_rows = future_mask_bool.all(axis=1)
    if not np.any(valid_rows):
        return False, "future_target_empty"

    valid_future_values = future_values_2d[valid_rows]
    if float(min_future_target_std) > 0.0 and float(np.std(valid_future_values)) < float(min_future_target_std):
        return False, "future_target_std"
    if float(min_future_target_range) > 0.0:
        future_range = float(np.max(valid_future_values) - np.min(valid_future_values))
        if future_range < float(min_future_target_range):
            return False, "future_target_range"

    if int(zero_run_limit) > 0:
        zero_run = 0
        for row_valid, row_values in zip(valid_rows, future_values_2d):
            if row_valid and np.max(np.abs(row_values)) <= float(future_zero_tolerance):
                zero_run += 1
                if zero_run >= int(zero_run_limit):
                    return False, "future_zero_run"
            else:
                zero_run = 0

    if int(constant_run_limit) > 0:
        constant_run = 1
        prev_values = None
        for row_valid, row_values in zip(valid_rows, future_values_2d):
            if not row_valid:
                constant_run = 1
                prev_values = None
                continue
            if prev_values is not None and np.max(np.abs(row_values - prev_values)) <= float(future_constant_tolerance):
                constant_run += 1
                if constant_run >= int(constant_run_limit):
                    return False, "future_constant_run"
            else:
                constant_run = 1
            prev_values = row_values

    return True, ""


def _masked_std(values, mask):
    """Return the largest per-column observed standard deviation.

    This mirrors ``_soft_mask_covariate_block``: a covariate block is retained
    when at least one observed channel has enough variation.
    """
    values_2d = _as_2d_values(values)
    mask_2d = _as_2d_mask(mask, len(values_2d))
    if values_2d.shape[0] != mask_2d.shape[0] or values_2d.shape[1] < mask_2d.shape[1]:
        return 0.0
    if mask_2d.shape[1] == 0:
        return 0.0
    valid = mask_2d.astype(bool, copy=False)
    if values_2d.shape[1] > mask_2d.shape[1]:
        values_2d = values_2d[:, : mask_2d.shape[1]]
        valid = valid[:, : mask_2d.shape[1]]
    observed_stds = []
    for column in range(values_2d.shape[1]):
        column_valid = valid[:, column]
        if not np.any(column_valid):
            continue
        observed = values_2d[:, column][column_valid]
        observed_stds.append(float(np.std(observed)) if observed.size > 1 else 0.0)
    return max(observed_stds, default=0.0)


def _validate_window_at_normalized(
    *,
    target_mask,
    target_values,
    seq_len,
    pred_len,
    resolution,
    start,
    sample_nan_ratio_threshold,
    history_covariate_mask,
    future_covariate_mask,
    history_covariate_values,
    future_covariate_values,
    decoder_covariate_context_len,
    min_past_target_valid_ratio,
    min_future_target_valid_ratio,
    min_history_covariate_valid_ratio,
    min_future_covariate_valid_ratio,
    min_history_covariate_std,
    min_future_covariate_std,
    min_future_target_std,
    max_future_zero_run_hours,
    max_future_constant_run_hours,
    future_zero_tolerance,
    future_constant_tolerance,
    min_future_target_range,
    check_finite=True,
):
    """Validate one already-normalized window and return a stable reason code."""
    start = int(start)
    seq_len = int(seq_len)
    pred_len = int(pred_len)
    if start < 0 or start + seq_len + pred_len > len(target_mask):
        return "out_of_bounds"

    s_end = start + seq_len
    f_end = s_end + pred_len
    decoder_cov_begin = max(start, s_end - int(decoder_covariate_context_len or 0))
    past_target_mask = target_mask[start:s_end]
    future_target_mask = target_mask[s_end:f_end]
    if past_target_mask.size == 0 or future_target_mask.size == 0:
        return "empty_target_window"

    if check_finite:
        if not _observed_values_are_finite(target_values[start:s_end], past_target_mask):
            return "past_target_nonfinite"
        if not _observed_values_are_finite(target_values[s_end:f_end], future_target_mask):
            return "future_target_nonfinite"

    if _window_valid_ratio(past_target_mask) < float(min_past_target_valid_ratio):
        return "past_target_valid_ratio"
    if _window_valid_ratio(future_target_mask) < float(min_future_target_valid_ratio):
        return "future_target_valid_ratio"

    history_cov_window = history_covariate_mask[start:s_end]
    future_cov_window = future_covariate_mask[decoder_cov_begin:f_end]
    if check_finite:
        if not _observed_values_are_finite(
            history_covariate_values[start:s_end], history_cov_window
        ):
            return "history_covariate_nonfinite"
        if not _observed_values_are_finite(
            future_covariate_values[decoder_cov_begin:f_end], future_cov_window
        ):
            return "future_covariate_nonfinite"
    if _window_valid_ratio(history_cov_window) < float(min_history_covariate_valid_ratio):
        return "history_covariate_valid_ratio"
    if _window_valid_ratio(future_cov_window) < float(min_future_covariate_valid_ratio):
        return "future_covariate_valid_ratio"

    if float(min_history_covariate_std) > 0.0:
        if _masked_std(history_covariate_values[start:s_end], history_cov_window) < float(min_history_covariate_std):
            return "history_covariate_std"
    if float(min_future_covariate_std) > 0.0:
        if _masked_std(future_covariate_values[decoder_cov_begin:f_end], future_cov_window) < float(min_future_covariate_std):
            return "future_covariate_std"

    passes, reason = _future_sequence_passes_qc(
        target_values[s_end:f_end],
        future_target_mask,
        min_future_target_std=min_future_target_std,
        min_future_target_range=min_future_target_range,
        zero_run_limit=_run_limit(max_future_zero_run_hours, resolution),
        constant_run_limit=_run_limit(max_future_constant_run_hours, resolution),
        future_zero_tolerance=future_zero_tolerance,
        future_constant_tolerance=future_constant_tolerance,
    )
    if not passes:
        return reason or "future_sequence_qc"

    total_count = float(past_target_mask.size + future_target_mask.size)
    invalid_ratio = 1.0 - (
        (float(np.sum(past_target_mask)) + float(np.sum(future_target_mask))) / total_count
    )
    if invalid_ratio > float(sample_nan_ratio_threshold):
        return "sample_nan_ratio_threshold"
    return None


def _run_limit(hours, resolution):
    """Convert an hour-based run limit to a number of resolution steps."""
    step_seconds = int(RESOLUTION_TO_SECONDS.get(str(resolution or ""), 0) or 0)
    if step_seconds <= 0 or float(hours) <= 0.0:
        return 0
    return max(1, int(round((float(hours) * 3600.0) / step_seconds)))


def validate_window_at(
    *,
    target_mask,
    target_values,
    seq_len,
    pred_len,
    start,
    resolution="1h",
    sample_nan_ratio_threshold=0.0,
    history_covariate_mask=None,
    future_covariate_mask=None,
    history_covariate_values=None,
    future_covariate_values=None,
    decoder_covariate_context_len=0,
    min_past_target_valid_ratio=1.0,
    min_future_target_valid_ratio=1.0,
    min_history_covariate_valid_ratio=1.0,
    min_future_covariate_valid_ratio=1.0,
    min_history_covariate_std=0.0,
    min_future_covariate_std=0.0,
    min_future_target_std=0.0,
    max_future_zero_run_hours=0.0,
    max_future_constant_run_hours=0.0,
    future_zero_tolerance=1e-6,
    future_constant_tolerance=1e-6,
    min_future_target_range=0.0,
    check_finite=True,
):
    """Validate exactly one candidate window using the shared QC rules.

    This is the pointwise counterpart of :func:`compute_valid_window_starts`.
    It deliberately performs no split-boundary inference; callers that create
    dynamic windows must check the actual C/H against their split borders.
    """
    try:
        target_mask_raw = np.asarray(target_mask)
        target_values_raw = np.asarray(target_values)
        if target_mask_raw.ndim not in (1, 2) or target_values_raw.ndim not in (1, 2):
            return "array_contract"
        length = int(target_mask_raw.shape[0])
        history_covariate_mask_raw = (
            np.asarray(history_covariate_mask)
            if history_covariate_mask is not None
            else np.ones((length, 0), dtype=bool)
        )
        future_covariate_mask_raw = (
            np.asarray(future_covariate_mask)
            if future_covariate_mask is not None
            else np.ones((length, 0), dtype=bool)
        )
        history_mask_2d = _as_2d_mask(history_covariate_mask_raw, length)
        future_mask_2d = _as_2d_mask(future_covariate_mask_raw, length)
        if history_covariate_values is None:
            history_covariate_values_raw = np.zeros(
                (length, history_mask_2d.shape[1]), dtype=np.float32
            )
        else:
            history_covariate_values_raw = np.asarray(history_covariate_values)
        if future_covariate_values is None:
            future_covariate_values_raw = np.zeros(
                (length, future_mask_2d.shape[1]), dtype=np.float32
            )
        else:
            future_covariate_values_raw = np.asarray(future_covariate_values)
    except (OverflowError, TypeError, ValueError):
        return "array_contract"
    contract_reason = _pointwise_array_contract_reason(
        target_mask_raw,
        target_values_raw,
        history_covariate_mask_raw,
        future_covariate_mask_raw,
        history_covariate_values_raw,
        future_covariate_values_raw,
    )
    if contract_reason is not None:
        return contract_reason
    if int(seq_len) <= 0 or int(pred_len) <= 0:
        return "shape_contract"
    target_mask = _as_2d_mask(target_mask_raw, len(target_mask_raw))
    target_values = _as_2d_values(target_values_raw)
    history_covariate_mask = history_mask_2d
    future_covariate_mask = future_mask_2d
    return _validate_window_at_normalized(
        target_mask=target_mask,
        target_values=target_values,
        seq_len=seq_len,
        pred_len=pred_len,
        resolution=resolution,
        start=start,
        sample_nan_ratio_threshold=sample_nan_ratio_threshold,
        history_covariate_mask=history_covariate_mask,
        future_covariate_mask=future_covariate_mask,
        history_covariate_values=_as_2d_values(history_covariate_values_raw),
        future_covariate_values=_as_2d_values(future_covariate_values_raw),
        decoder_covariate_context_len=decoder_covariate_context_len,
        min_past_target_valid_ratio=min_past_target_valid_ratio,
        min_future_target_valid_ratio=min_future_target_valid_ratio,
        min_history_covariate_valid_ratio=min_history_covariate_valid_ratio,
        min_future_covariate_valid_ratio=min_future_covariate_valid_ratio,
        min_history_covariate_std=min_history_covariate_std,
        min_future_covariate_std=min_future_covariate_std,
        min_future_target_std=min_future_target_std,
        max_future_zero_run_hours=max_future_zero_run_hours,
        max_future_constant_run_hours=max_future_constant_run_hours,
        future_zero_tolerance=future_zero_tolerance,
        future_constant_tolerance=future_constant_tolerance,
        min_future_target_range=min_future_target_range,
        check_finite=bool(check_finite),
    )


def compute_valid_window_starts(
    *,
    target_mask,
    target_values,
    seq_len,
    pred_len,
    resolution="1h",
    sample_nan_ratio_threshold=0.0,
    history_covariate_mask=None,
    future_covariate_mask=None,
    decoder_covariate_context_len=0,
    min_past_target_valid_ratio=1.0,
    min_future_target_valid_ratio=1.0,
    min_history_covariate_valid_ratio=1.0,
    min_future_covariate_valid_ratio=1.0,
    min_future_target_std=0.0,
    max_future_zero_run_hours=0.0,
    max_future_constant_run_hours=0.0,
    future_zero_tolerance=1e-6,
    future_constant_tolerance=1e-6,
    min_future_target_range=0.0,
    return_reasons=False,
):
    """Return aligned valid sample starts using target and covariate masks.

    This function operates only at station/window level. It intentionally does
    not build batches, FoundationSample objects, scalers, or metadata-heavy
    payloads, so FM indexed collation remains independent.
    """
    target_mask = _as_2d_mask(target_mask, len(target_mask))
    target_values = _as_2d_values(target_values)
    history_covariate_mask = _as_2d_mask(history_covariate_mask, len(target_mask))
    future_covariate_mask = _as_2d_mask(future_covariate_mask, len(target_mask))

    seq_len = int(seq_len)
    pred_len = int(pred_len)
    decoder_covariate_context_len = max(0, int(decoder_covariate_context_len or 0))
    max_start = len(target_mask) - seq_len - pred_len + 1
    reasons = Counter()
    if max_start <= 0:
        if return_reasons:
            return [], 0, {"too_short": 1}
        return []

    step_seconds = int(RESOLUTION_TO_SECONDS.get(str(resolution or ""), 0) or 0)
    zero_run_limit = 0
    constant_run_limit = 0
    if step_seconds > 0:
        if float(max_future_zero_run_hours) > 0.0:
            zero_run_limit = max(1, int(round((float(max_future_zero_run_hours) * 3600.0) / step_seconds)))
        if float(max_future_constant_run_hours) > 0.0:
            constant_run_limit = max(1, int(round((float(max_future_constant_run_hours) * 3600.0) / step_seconds)))

    valid_starts = []
    for s_begin in range(max_start):
        s_end = s_begin + seq_len
        f_begin = s_end
        f_end = f_begin + pred_len
        decoder_cov_begin = max(s_begin, f_begin - decoder_covariate_context_len)

        past_target_mask = target_mask[s_begin:s_end]
        future_target_mask = target_mask[f_begin:f_end]
        if past_target_mask.size == 0 or future_target_mask.size == 0:
            reasons["empty_target_window"] += 1
            continue

        past_valid_ratio = _window_valid_ratio(past_target_mask)
        future_valid_ratio = _window_valid_ratio(future_target_mask)
        if past_valid_ratio < float(min_past_target_valid_ratio):
            reasons["past_target_valid_ratio"] += 1
            continue
        if future_valid_ratio < float(min_future_target_valid_ratio):
            reasons["future_target_valid_ratio"] += 1
            continue

        if _window_valid_ratio(history_covariate_mask[s_begin:s_end]) < float(min_history_covariate_valid_ratio):
            reasons["history_covariate_valid_ratio"] += 1
            continue
        if _window_valid_ratio(future_covariate_mask[decoder_cov_begin:f_end]) < float(min_future_covariate_valid_ratio):
            reasons["future_covariate_valid_ratio"] += 1
            continue

        future_values = target_values[f_begin:f_end]
        passes, reason = _future_sequence_passes_qc(
            future_values,
            future_target_mask,
            min_future_target_std=min_future_target_std,
            min_future_target_range=min_future_target_range,
            zero_run_limit=zero_run_limit,
            constant_run_limit=constant_run_limit,
            future_zero_tolerance=future_zero_tolerance,
            future_constant_tolerance=future_constant_tolerance,
        )
        if not passes:
            reasons[reason or "future_sequence_qc"] += 1
            continue

        total_count = float(past_target_mask.size + future_target_mask.size)
        invalid_ratio = 1.0 - ((float(np.sum(past_target_mask)) + float(np.sum(future_target_mask))) / total_count)
        if invalid_ratio > float(sample_nan_ratio_threshold):
            reasons["sample_nan_ratio_threshold"] += 1
            continue

        valid_starts.append(int(s_begin))

    if return_reasons:
        return valid_starts, max_start, dict(reasons)
    return valid_starts
