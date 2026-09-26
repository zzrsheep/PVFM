from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np


def _sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(source):
    """Return the cache identity used in every planned row.

    Binary tensors are opened read-only.  The cache metadata hash is enough to
    detect a changed cache contract here; the complete source signature below
    additionally covers the indexed views and control files.
    """
    cache_dir = str(getattr(source, "binary_dataset_dir", "") or "")
    metadata_path = Path(cache_dir) / "metadata.json" if cache_dir else None
    if metadata_path is not None and metadata_path.is_file():
        return _sha256_file(metadata_path)
    return "memory"


def _timestamps_as_datetime64_ns(values):
    """Normalize cache timestamps without changing their source representation.

    The binary caches use either ``datetime64`` arrays or integer epoch
    nanoseconds.  NumPy's direct cast preserves the latter convention, while
    making the unit explicit for cadence and provenance checks.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError("timestamps must be a one-dimensional array")
    try:
        result = array.astype("datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamps cannot be represented as datetime64[ns]") from exc
    if np.isnat(result).any():
        raise ValueError("timestamps contain NaT")
    return result


def _timestamp_text(value):
    value = _timestamps_as_datetime64_ns(np.asarray(value).reshape(1))[0]
    text = np.datetime_as_string(value, unit="s")
    return str(text).replace("T", " ")


def split_bounds(length, seq_len, pred_len, split):
    """Return the cache-compatible absolute bounds for one actual C/H."""
    length = int(length)
    seq_len = int(seq_len)
    pred_len = int(pred_len)
    split = str(split)
    if split == "all":
        return 0, length
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split!r}.")
    num_train = int(length * 0.7)
    num_test = int(length * 0.2)
    num_val = length - num_train - num_test
    border1s = [0, max(0, num_train - seq_len), max(0, length - num_test - seq_len)]
    border2s = [num_train, num_train + num_val, length]
    split_id = {"train": 0, "val": 1, "test": 2}[split]
    return int(border1s[split_id]), int(border2s[split_id])


_REQUIRED_ARRAY_KEYS = (
    "target",
    "target_mask",
    "history_covariates",
    "history_covariate_mask",
    "future_covariates",
    "future_covariate_mask",
    "time_features",
    "timestamps",
    "site_features",
)


def _array_contract_reason(arrays):
    """Return a stable reason when a source station view is malformed."""
    for key in _REQUIRED_ARRAY_KEYS:
        if key not in arrays:
            return f"missing_array_{key}"
    try:
        target = np.asarray(arrays["target"])
        target_mask = np.asarray(arrays["target_mask"])
        if target.ndim not in (1, 2) or target_mask.ndim not in (1, 2):
            return "array_rank"
        length = int(target.shape[0])
        if length <= 0 or int(target_mask.shape[0]) != length:
            return "target_length"
        for key in (
            "history_covariates",
            "history_covariate_mask",
            "future_covariates",
            "future_covariate_mask",
            "time_features",
        ):
            array = np.asarray(arrays[key])
            if array.ndim not in (1, 2) or int(array.shape[0]) != length:
                return f"{key}_length"
        timestamps = np.asarray(arrays["timestamps"])
        if timestamps.ndim != 1 or len(timestamps) != length:
            return "timestamps_length"
        site_features = np.asarray(arrays["site_features"])
        if site_features.ndim not in (1, 2) or site_features.size < 5:
            return "site_features_shape"
        try:
            if not np.isfinite(site_features.astype(np.float64, copy=False)).all():
                return "site_features_nonfinite"
        except (OverflowError, TypeError, ValueError):
            return "site_features_contract"
        target_width = 1 if target.ndim == 1 else int(target.shape[1])
        mask_width = 1 if target_mask.ndim == 1 else int(target_mask.shape[1])
        if target_width < mask_width:
            return "target_width"
        for values_key, mask_key in (
            ("history_covariates", "history_covariate_mask"),
            ("future_covariates", "future_covariate_mask"),
        ):
            values = np.asarray(arrays[values_key])
            mask = np.asarray(arrays[mask_key])
            values_width = 1 if values.ndim == 1 else int(values.shape[1])
            mask_width = 1 if mask.ndim == 1 else int(mask.shape[1])
            if values_width < mask_width:
                return f"{values_key}_width"
    except (KeyError, OverflowError, TypeError, ValueError, IndexError):
        return "array_contract"
    return None


class DynamicPoolUnavailable(RuntimeError):
    """A selected source-task/region pool has no batch for one dynamic C/H."""


def _soft_mask_covariate_block(values, mask, min_valid_ratio=0.0, min_std=0.0):
    if values.size == 0 or mask.size == 0:
        return values, mask

    valid_ratio = float(np.sum(mask)) / float(mask.size)
    if valid_ratio < float(min_valid_ratio):
        return np.zeros_like(values, dtype=np.float32), np.zeros_like(
            mask, dtype=np.float32
        )

    if float(min_std) <= 0.0:
        return values, mask

    observed_stds = []
    mask_bool = mask.astype(bool)
    for dim in range(values.shape[-1]):
        dim_vals = values[:, dim]
        dim_mask = mask_bool[:, dim]
        if not np.any(dim_mask):
            continue
        valid_vals = dim_vals[dim_mask]
        if valid_vals.size <= 1:
            observed_stds.append(0.0)
        else:
            observed_stds.append(float(np.std(valid_vals)))

    if observed_stds and max(observed_stds) < float(min_std):
        return np.zeros_like(values, dtype=np.float32), np.zeros_like(
            mask, dtype=np.float32
        )

    return values, mask
