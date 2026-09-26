import os
import time
import json
import hashlib
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from foundation.data.manifest import filter_manifest_rows, load_station_manifest
from foundation.data.alignment.window_qc import compute_valid_window_starts
from foundation.data.transforms import build_time_features, fit_or_identity_scaler
from foundation.task_specs import build_task_adapter, resolve_task_lengths, resolve_task_spec
from utils.sample_index import SampleIndexFilter, filter_valid_starts_by_sample_index


DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
]

RESOLUTION_TO_SECONDS = {
    "10sec": 10,
    "1min": 60,
    "5min": 300,
    "10min": 600,
    "15min": 900,
    "30min": 1800,
    "1h": 3600,
}

RESOLUTION_TO_PANDAS_FREQ = {
    "1min": "1min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
}

MIN_POINTS_PER_HOUR = {
    "10sec": 270,
    "1min": 45,
    "5min": 9,
    "10min": 5,
    "15min": 3,
    "30min": 2,
    "1h": 1,
}


@dataclass
class CapacityCandidate:
    value_kw: float
    field_path: str
    source: str


def parse_timestamp(value):
    value = str(value).strip()
    if not value:
        return None
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    if value.endswith("Z"):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_timestamp_series(values):
    """Return timezone-naive timestamps, converting explicit offsets to UTC first."""
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(timestamps, pd.Series):
        return timestamps.dt.tz_convert(None)
    if isinstance(timestamps, pd.DatetimeIndex):
        return pd.Series(timestamps.tz_convert(None))
    return pd.to_datetime(timestamps, errors="coerce")


def _parse_timestamp_series_fast(values):
    """Parse timestamps with a vectorized pandas fast path and a small-row fallback."""
    raw = pd.Series(values, copy=False)
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)

    if isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.Series(parsed, index=raw.index)

    failed_mask = parsed.isna()
    if failed_mask.any():
        raw_strings = raw.astype("string").str.strip()
        failed_mask &= raw_strings.notna() & raw_strings.ne("")
        if failed_mask.any():
            recovered = raw_strings.loc[failed_mask].map(parse_timestamp)
            recovered = pd.to_datetime(recovered, errors="coerce", utc=True)
            if isinstance(recovered, pd.DatetimeIndex):
                recovered = pd.Series(recovered, index=raw.loc[failed_mask].index)
            parsed.loc[failed_mask] = recovered

    return parsed.dt.tz_convert(None)


def _read_csv_usecols(csv_path, columns):
    selected_cols = list(dict.fromkeys(col for col in columns if col))
    try:
        return pd.read_csv(csv_path, usecols=selected_cols, low_memory=False)
    except ValueError:
        return None


def _maybe_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_station_metadata(payload):
    if not isinstance(payload, dict):
        return {}
    station_md = payload.get("station_metadata", {})
    return station_md if isinstance(station_md, dict) else {}


def _safe_float_from_dict(payload, key):
    if not isinstance(payload, dict):
        return None
    return _maybe_float(payload.get(key))


def _load_other_data_json(station_dir):
    path = os.path.join(station_dir, "other_data.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _extract_timezone_name(row, other_data):
    row = row or {}
    tz = str(row.get("timezone", "") or "").strip()
    if tz:
        return tz
    station_md = _extract_station_metadata(other_data or {})
    tz = str(station_md.get("timezone", "") or "").strip()
    return tz


def _infer_timezone_offset_hours(row, other_data, reference_timestamp=None):
    tz_name = _extract_timezone_name(row, other_data)
    if not tz_name:
        return np.nan, ""
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return np.nan, tz_name

    try:
        if reference_timestamp is None or pd.isna(reference_timestamp):
            reference_timestamp = pd.Timestamp("2020-01-01 12:00:00")
        ts = pd.Timestamp(reference_timestamp)
        if ts.tzinfo is not None:
            localized = ts.tz_convert(tz)
        else:
            localized = ts.tz_localize(tz)
        offset = localized.utcoffset()
        if offset is None:
            return np.nan, tz_name
        return float(offset.total_seconds() / 3600.0), tz_name
    except Exception:
        return np.nan, tz_name


def _solar_daylight_mask(timestamps, row, timezone_offset_hours):
    ts = _normalize_timestamp_series(pd.Series(timestamps))
    try:
        lat = float(row.get("lat", np.nan))
        lon = float(row.get("lon", np.nan))
    except Exception:
        lat = np.nan
        lon = np.nan

    if not np.isfinite(lat) or not np.isfinite(lon):
        hours = ts.dt.hour.to_numpy(dtype=np.float64) + ts.dt.minute.to_numpy(dtype=np.float64) / 60.0
        return ((hours >= 6.0) & (hours <= 18.0)).reshape(-1, 1)

    offset = float(timezone_offset_hours) if np.isfinite(timezone_offset_hours) else 0.0
    hour_of_day = (
        ts.dt.hour.to_numpy(dtype=np.float64)
        + ts.dt.minute.to_numpy(dtype=np.float64) / 60.0
        + ts.dt.second.to_numpy(dtype=np.float64) / 3600.0
    )
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=np.float64)
    solar_hour = hour_of_day - offset + lon / 15.0
    declination = 23.45 * np.sin(2.0 * np.pi * (284.0 + day_of_year) / 365.0)
    hour_angle = 15.0 * (solar_hour - 12.0)
    lat_rad = np.deg2rad(lat)
    dec_rad = np.deg2rad(declination)
    hour_rad = np.deg2rad(hour_angle)
    sin_elevation = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
    )
    return (sin_elevation > 0.0).reshape(-1, 1)


def _clean_pv_power_values(
    power_kw,
    timestamps,
    row,
    timezone_offset_hours,
    capacity_used_kw=np.nan,
    *,
    sentinel_negative=-1e5,
    night_small_negative_abs_kw=10.0,
    night_small_negative_capacity_frac=0.05,
    max_positive_capacity_frac=1.5,
):
    cleaned = np.asarray(power_kw, dtype=np.float32).copy()
    if cleaned.ndim == 1:
        cleaned = cleaned.reshape(-1, 1)
    mask = np.isfinite(cleaned)
    mask &= cleaned > float(sentinel_negative)

    daylight = _solar_daylight_mask(timestamps, row, timezone_offset_hours)
    if daylight.shape[0] != cleaned.shape[0]:
        daylight = np.ones((cleaned.shape[0], 1), dtype=bool)
    if daylight.shape[1] != cleaned.shape[1]:
        daylight = np.repeat(daylight[:, :1], cleaned.shape[1], axis=1)

    cap_kw = float(capacity_used_kw) if np.isfinite(capacity_used_kw) and float(capacity_used_kw) > 0 else np.nan
    small_neg_limit = float(night_small_negative_abs_kw)
    if np.isfinite(cap_kw):
        small_neg_limit = max(small_neg_limit, float(night_small_negative_capacity_frac) * cap_kw)

    negative = mask & (cleaned < 0.0)
    night_small_negative = negative & (~daylight) & (cleaned >= -small_neg_limit)
    cleaned[night_small_negative] = 0.0

    invalid_negative = negative & ~night_small_negative
    mask[invalid_negative] = False

    if np.isfinite(cap_kw) and float(max_positive_capacity_frac) > 0:
        extreme_positive = mask & (cleaned > float(max_positive_capacity_frac) * cap_kw)
        mask[extreme_positive] = False

    cleaned[~mask] = np.nan
    return cleaned.astype(np.float32), mask.astype(bool)


def _extract_capacity_candidates(payload):
    candidates = []
    candidate_specs = [
        ("station_metadata.capacity_kw", ("station_metadata", "capacity_kw"), 1.0),
        ("station_metadata.summary_capacity_kw", ("station_metadata", "summary_capacity_kw"), 1.0),
        ("station_metadata.official_dc_capacity_kW", ("station_metadata", "official_dc_capacity_kW"), 1.0),
        ("dataset_metadata.Capacity", ("dataset_metadata", "Capacity"), 1.0),
        ("top.capacity_kw", ("capacity_kw",), 1.0),
        ("top.summary_capacity_kw", ("summary_capacity_kw",), 1.0),
        ("top.official_dc_capacity_kW", ("official_dc_capacity_kW",), 1.0),
        ("top.dc_capacity_kW", ("dc_capacity_kW",), 1.0),
        ("top.capacity_mw", ("capacity_mw",), 1000.0),
        ("top.capacity_MW", ("capacity_MW",), 1000.0),
    ]
    for field_path, keys, multiplier in candidate_specs:
        node = payload
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        numeric = _maybe_float(node)
        if numeric is None or numeric <= 0:
            continue
        source = ""
        if field_path.startswith("station_metadata.") and isinstance(payload.get("station_metadata"), dict):
            source = str(payload["station_metadata"].get("capacity_source", "")).strip()
        candidates.append(CapacityCandidate(value_kw=float(numeric) * float(multiplier), field_path=field_path, source=source))
    return candidates


def _choose_capacity_candidate(candidates):
    if not candidates:
        return None
    preferred_prefixes = [
        "station_metadata.capacity_kw",
        "station_metadata.summary_capacity_kw",
        "station_metadata.official_dc_capacity_kW",
        "dataset_metadata.Capacity",
    ]
    for prefix in preferred_prefixes:
        for candidate in candidates:
            if candidate.field_path == prefix:
                return candidate
    return candidates[0]


def _classify_capacity_issue(ratio, ok_low, ok_high, suspect_low, suspect_high):
    if ratio is None or not np.isfinite(ratio):
        return "missing_or_unusable", "metadata_capacity_missing_or_station_unusable"
    if 5e-4 <= ratio <= 2e-3:
        return "unit_mismatch_x1000", "observed_power_is_about_1_over_1000_of_capacity_metadata"
    if 5e2 <= ratio <= 2e3:
        return "unit_mismatch_x1000", "observed_power_is_about_1000_times_capacity_metadata"
    if ok_low <= ratio <= ok_high:
        return "cap_ok", ""
    if suspect_low <= ratio <= suspect_high:
        return "cap_suspect", "metadata_capacity_is_plausible_but_ratio_is_outside_ok_band"
    return "cap_bad_or_missing", "metadata_capacity_failed_sanity_check"


def _infer_power_semantics(payload, power_p99_5_kw, power_max_kw, cap_meta_kw, ok_low, ok_high, suspect_low, suspect_high):
    payload = payload or {}
    station_md = _extract_station_metadata(payload)
    source_type = str(payload.get("source_type", "")).strip()
    resolution_note = str(payload.get("resolution_note", "")).strip().lower()
    notes = str(station_md.get("notes", "")).strip().lower()
    max_norm = _safe_float_from_dict(station_md, "max_norm")
    min_norm = _safe_float_from_dict(station_md, "min_norm")

    if source_type == "processed_station_series_from_dataset_only_pv_stations":
        return "already_normalized", "source_type_processed_station_series_from_dataset_only_pv_stations", 1.0
    if "normalized_output" in resolution_note:
        return "already_normalized", "resolution_note_mentions_normalized_output", 1.0
    if max_norm is not None and power_max_kw is not None and max_norm <= 1.5 and float(power_max_kw) <= 1.5:
        return "already_normalized", "max_norm_and_power_max_look_normalized", 1.0
    if "capacity_is_estimated_not_nameplate" in notes and max_norm is not None and max_norm <= 1.5 and min_norm is not None and min_norm >= -0.1:
        return "already_normalized", "notes_and_norm_range_look_normalized", 1.0

    if cap_meta_kw is not None and np.isfinite(cap_meta_kw) and cap_meta_kw > 0 and power_p99_5_kw is not None and np.isfinite(power_p99_5_kw):
        raw_ratio = float(power_p99_5_kw) / float(cap_meta_kw)
        mw_scaled_ratio = float(power_p99_5_kw) * 1000.0 / float(cap_meta_kw)
        if ok_low <= mw_scaled_ratio <= ok_high:
            return "power_unit_mw_suspect", "power_times_1000_matches_capacity_ratio_ok_band", 1000.0
        if suspect_low <= mw_scaled_ratio <= suspect_high:
            return "power_unit_mw_suspect", "power_times_1000_matches_capacity_ratio_suspect_band", 1000.0
        if 5e-4 <= raw_ratio <= 2e-3:
            return "power_unit_mw_suspect", "raw_ratio_near_1_over_1000_and_power_times_1000_is_more_plausible", 1000.0

    return "power_unit_kw_plausible", "", 1.0


def _resolve_capacity_normalization(power_series, other_data, proxy_quantile):
    candidates = _extract_capacity_candidates(other_data or {})
    chosen = _choose_capacity_candidate(candidates)
    cap_meta_kw = chosen.value_kw if chosen is not None else np.nan
    cap_meta_field = chosen.field_path if chosen is not None else ""
    cap_meta_source = chosen.source if chosen is not None else ""

    power_values = np.asarray(power_series, dtype=np.float64)
    power_values = power_values[np.isfinite(power_values)]
    if power_values.size == 0:
        power_p99_5 = np.nan
        power_max = np.nan
    else:
        power_p99_5 = float(np.percentile(power_values, float(proxy_quantile)))
        power_max = float(np.max(power_values))

    power_semantics, power_semantics_note, power_unit_scale_to_kw = _infer_power_semantics(
        other_data,
        None if not np.isfinite(power_p99_5) else power_p99_5,
        None if not np.isfinite(power_max) else power_max,
        None if not np.isfinite(cap_meta_kw) else cap_meta_kw,
        0.2,
        1.2,
        0.05,
        1.8,
    )
    adjusted_power_p99_5 = power_p99_5 * power_unit_scale_to_kw if np.isfinite(power_p99_5) else np.nan
    ratio_p99_5_to_cap = np.nan
    if chosen is not None and chosen.value_kw > 0 and np.isfinite(adjusted_power_p99_5):
        ratio_p99_5_to_cap = float(adjusted_power_p99_5 / chosen.value_kw)

    if power_semantics == "already_normalized":
        cap_status = "already_normalized"
        cap_note = power_semantics_note
        capacity_used_kw = chosen.value_kw if chosen is not None and chosen.value_kw > 0 else 1.0
        capacity_used_source = "already_normalized"
    else:
        cap_status, cap_note = _classify_capacity_issue(
            None if not np.isfinite(ratio_p99_5_to_cap) else float(ratio_p99_5_to_cap),
            0.2,
            1.2,
            0.05,
            1.8,
        )
        if chosen is not None and cap_status in {"cap_ok", "cap_suspect"}:
            capacity_used_kw = chosen.value_kw
            capacity_used_source = "metadata"
        else:
            capacity_used_kw = adjusted_power_p99_5 if np.isfinite(adjusted_power_p99_5) else np.nan
            capacity_used_source = "proxy_p99_5"

    return {
        "power_semantics": power_semantics,
        "power_semantics_note": power_semantics_note,
        "power_unit_scale_to_kw": float(power_unit_scale_to_kw),
        "cap_meta_kw": float(cap_meta_kw) if np.isfinite(cap_meta_kw) else np.nan,
        "cap_meta_field": cap_meta_field,
        "cap_meta_source": cap_meta_source,
        "ratio_p99_5_to_cap": float(ratio_p99_5_to_cap) if np.isfinite(ratio_p99_5_to_cap) else np.nan,
        "cap_status": cap_status,
        "cap_note": cap_note,
        "capacity_used_kw": float(capacity_used_kw) if np.isfinite(capacity_used_kw) else np.nan,
        "capacity_used_source": capacity_used_source,
    }


def _fit_scaler_on_observed(values, observed_mask, use_scale=True):
    if values.shape[-1] == 0:
        return None
    observed_rows = observed_mask.all(axis=1) if observed_mask.ndim == 2 else observed_mask
    train_values = values[observed_rows]
    if len(train_values) == 0:
        train_values = np.zeros((1, values.shape[-1]), dtype=np.float32)
    return fit_or_identity_scaler(train_values, use_scale=use_scale)


def _resolve_target_standardization_mode(target_normalization, target_standardization):
    mode = str(target_standardization or "auto").strip().lower()
    if mode not in {"auto", "standard", "none"}:
        raise ValueError(f"Unsupported target_standardization: {target_standardization}")
    if mode != "auto":
        return mode
    if str(target_normalization or "none").strip().lower() == "capacity_factor":
        return "none"
    return "standard"


def _resolve_target_transform(target_transform, target_normalization, target_standardization):
    transform = str(target_transform or "").strip().lower()
    if transform:
        aliases = {
            "cap": "capacity_factor",
            "cap_only": "capacity_factor",
            "capacity_factor_only": "capacity_factor",
            "cap_standard": "capacity_factor_standard",
            "capacity_factor_standardized": "capacity_factor_standard",
            "raw_power": "raw",
            "raw_standard": "standard",
            "raw_standardized": "standard",
        }
        transform = aliases.get(transform, transform)
        if transform not in {"raw", "standard", "capacity_factor", "capacity_factor_standard"}:
            raise ValueError(f"Unsupported target_transform: {target_transform}")
        if transform == "raw":
            return "raw", "none", "none"
        if transform == "standard":
            return "standard", "none", "standard"
        if transform == "capacity_factor":
            return "capacity_factor", "capacity_factor", "none"
        return "capacity_factor_standard", "capacity_factor", "standard"

    normalization = str(target_normalization or "none").strip().lower()
    standardization = _resolve_target_standardization_mode(normalization, target_standardization)
    if normalization == "capacity_factor":
        transform = "capacity_factor_standard" if standardization == "standard" else "capacity_factor"
    else:
        transform = "standard" if standardization == "standard" else "raw"
    return transform, normalization, standardization


def _transform_with_mask(values, observed_mask, scaler):
    if values.shape[-1] == 0:
        return values.astype(np.float32)
    filled = values.copy()
    filled[~observed_mask] = 0.0
    transformed = scaler.transform(filled).astype(np.float32)
    transformed[~observed_mask] = 0.0
    return transformed


def _compute_valid_window_starts(
    target_mask,
    target_values,
    seq_len,
    pred_len,
    resolution,
    sample_nan_ratio_threshold,
    min_past_target_valid_ratio=1.0,
    min_future_target_valid_ratio=1.0,
    min_future_target_std=0.0,
    max_future_zero_run_hours=0.0,
    max_future_constant_run_hours=0.0,
    future_zero_tolerance=1e-6,
    future_constant_tolerance=1e-6,
    min_future_target_range=0.0,
):
    max_start = len(target_mask) - seq_len - pred_len + 1
    if max_start <= 0:
        return []

    valid_starts = []
    threshold = float(sample_nan_ratio_threshold)
    min_past_valid = float(min_past_target_valid_ratio)
    min_future_valid = float(min_future_target_valid_ratio)
    min_future_std = float(min_future_target_std)
    min_future_range = float(min_future_target_range)
    step_seconds = int(RESOLUTION_TO_SECONDS.get(resolution, 0) or 0)
    zero_run_limit = 0
    constant_run_limit = 0
    if step_seconds > 0:
        if float(max_future_zero_run_hours) > 0.0:
            zero_run_limit = max(1, int(round((float(max_future_zero_run_hours) * 3600.0) / step_seconds)))
        if float(max_future_constant_run_hours) > 0.0:
            constant_run_limit = max(1, int(round((float(max_future_constant_run_hours) * 3600.0) / step_seconds)))

    def _future_sequence_passes_qc(future_values, future_mask):
        future_mask_bool = future_mask.astype(bool).reshape(future_mask.shape[0], -1)
        future_values_2d = future_values.reshape(future_values.shape[0], -1)
        valid_rows = future_mask_bool.all(axis=1)
        if not np.any(valid_rows):
            return False

        valid_future_values = future_values_2d[valid_rows]
        if min_future_range > 0.0:
            future_range = float(np.max(valid_future_values) - np.min(valid_future_values))
            if future_range < min_future_range:
                return False

        if zero_run_limit > 0:
            zero_run = 0
            for row_valid, row_values in zip(valid_rows, future_values_2d):
                if row_valid and np.max(np.abs(row_values)) <= float(future_zero_tolerance):
                    zero_run += 1
                    if zero_run >= zero_run_limit:
                        return False
                else:
                    zero_run = 0

        if constant_run_limit > 0:
            constant_run = 1
            prev_values = None
            for row_valid, row_values in zip(valid_rows, future_values_2d):
                if not row_valid:
                    constant_run = 1
                    prev_values = None
                    continue
                if prev_values is not None and np.max(np.abs(row_values - prev_values)) <= float(future_constant_tolerance):
                    constant_run += 1
                    if constant_run >= constant_run_limit:
                        return False
                else:
                    constant_run = 1
                prev_values = row_values

        return True

    for s_begin in range(max_start):
        s_end = s_begin + seq_len
        f_begin = s_end
        f_end = f_begin + pred_len

        past_mask = target_mask[s_begin:s_end]
        future_mask = target_mask[f_begin:f_end]
        if past_mask.size == 0 or future_mask.size == 0:
            continue

        past_valid_ratio = float(np.sum(past_mask)) / float(past_mask.size)
        future_valid_ratio = float(np.sum(future_mask)) / float(future_mask.size)
        if past_valid_ratio < min_past_valid or future_valid_ratio < min_future_valid:
            continue

        if min_future_std > 0.0:
            future_values = target_values[f_begin:f_end]
            valid_future_values = future_values[future_mask.astype(bool)]
            if valid_future_values.size == 0:
                continue
            if float(np.std(valid_future_values)) < min_future_std:
                continue

        future_values = target_values[f_begin:f_end]
        if not _future_sequence_passes_qc(future_values, future_mask):
            continue

        total_count = float(past_mask.size + future_mask.size)
        invalid_ratio = 1.0 - ((float(np.sum(past_mask)) + float(np.sum(future_mask))) / total_count)
        if invalid_ratio <= threshold:
            valid_starts.append(s_begin)
    return valid_starts


def _build_covariate_frame(csv_path, time_col, covariate_cols):
    if not csv_path or not covariate_cols or not os.path.exists(csv_path):
        return None

    frame = _read_csv_usecols(csv_path, [time_col, *covariate_cols])
    if frame is None:
        return None
    required_cols = [time_col, *covariate_cols]
    if any(col not in frame.columns for col in required_cols):
        return None

    cov_frame = pd.DataFrame({"timestamp": _parse_timestamp_series_fast(frame[time_col])})
    for col in covariate_cols:
        cov_frame[col] = pd.to_numeric(frame[col], errors="coerce")
    cov_frame = cov_frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    cov_frame = cov_frame.drop_duplicates(subset="timestamp", keep="last")
    return cov_frame


def _align_covariates(base_timestamps, covariate_frame, covariate_cols, *, base_resolution="", allow_coarse_hold=False):
    if not covariate_cols:
        empty = np.zeros((len(base_timestamps), 0), dtype=np.float32)
        return empty, empty.astype(bool)
    if covariate_frame is None:
        values = np.zeros((len(base_timestamps), len(covariate_cols)), dtype=np.float32)
        mask = np.zeros_like(values, dtype=bool)
        return values, mask

    aligned = pd.DataFrame({"timestamp": _normalize_timestamp_series(base_timestamps)})
    covariate_frame = covariate_frame.copy()
    covariate_frame["timestamp"] = _normalize_timestamp_series(covariate_frame["timestamp"])
    covariate_granularity = _infer_granularity_from_timestamps(covariate_frame["timestamp"])
    base_seconds = RESOLUTION_TO_SECONDS.get(base_resolution or "", 0)
    cov_seconds = RESOLUTION_TO_SECONDS.get(covariate_granularity or "", 0)

    if (
        allow_coarse_hold
        and base_seconds > 0
        and cov_seconds > 0
        and cov_seconds > base_seconds
    ):
        aligned = pd.merge_asof(
            aligned.sort_values("timestamp"),
            covariate_frame.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(seconds=cov_seconds),
        )
    else:
        aligned = aligned.merge(covariate_frame, on="timestamp", how="left")
    values = aligned[covariate_cols].to_numpy(dtype=np.float32)
    mask = ~np.isnan(values)
    values[~mask] = 0.0
    return values, mask


def _prepare_native_covariate_arrays(covariate_frame, covariate_cols):
    if not covariate_cols or covariate_frame is None or covariate_frame.empty:
        empty_values = np.zeros((0, len(covariate_cols)), dtype=np.float32)
        empty_mask = np.zeros((0, len(covariate_cols)), dtype=bool)
        return {
            "timestamps": pd.to_datetime([]),
            "values": empty_values,
            "mask": empty_mask,
            "resolution": "",
        }

    work = covariate_frame.copy()
    work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    work = work.drop_duplicates(subset="timestamp", keep="last")
    values = work[covariate_cols].to_numpy(dtype=np.float32)
    mask = ~np.isnan(values)
    values[~mask] = 0.0
    resolution = _infer_granularity_from_timestamps(work["timestamp"])
    return {
        "timestamps": pd.to_datetime(work["timestamp"]),
        "values": values,
        "mask": mask,
        "resolution": resolution,
    }


def _transform_native_covariates(native_payload, scaler):
    if native_payload is None:
        return _prepare_native_covariate_arrays(None, [])
    values = native_payload["values"]
    mask = native_payload["mask"]
    if values.size == 0 or scaler is None:
        return {
            "timestamps": native_payload["timestamps"],
            "values": values.astype(np.float32),
            "mask": mask.astype(np.float32),
            "resolution": native_payload.get("resolution", ""),
        }
    transformed = _transform_with_mask(values, mask, scaler)
    return {
        "timestamps": native_payload["timestamps"],
        "values": transformed.astype(np.float32),
        "mask": mask.astype(np.float32),
        "resolution": native_payload.get("resolution", ""),
    }


def _slice_native_covariate_window(native_payload, start_ts, end_ts):
    if native_payload is None:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            "",
        )
    timestamps = pd.to_datetime(native_payload.get("timestamps", []))
    values = np.asarray(native_payload.get("values", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    mask = np.asarray(native_payload.get("mask", np.zeros_like(values)), dtype=np.float32)
    resolution = str(native_payload.get("resolution", "") or "")
    if len(timestamps) == 0 or values.size == 0:
        time_dim = 5 if resolution and RESOLUTION_TO_SECONDS.get(resolution, 3600) < 3600 else 4
        return (
            values.reshape(0, values.shape[-1] if values.ndim == 2 else 0),
            mask.reshape(0, mask.shape[-1] if mask.ndim == 2 else 0),
            np.zeros((0, time_dim), dtype=np.float32),
            resolution,
        )
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    keep = (timestamps >= start_ts) & (timestamps <= end_ts)
    if not np.any(keep):
        time_dim = 5 if resolution and RESOLUTION_TO_SECONDS.get(resolution, 3600) < 3600 else 4
        return (
            np.zeros((0, values.shape[-1]), dtype=np.float32),
            np.zeros((0, mask.shape[-1]), dtype=np.float32),
            np.zeros((0, time_dim), dtype=np.float32),
            resolution,
        )
    sliced_ts = timestamps[keep]
    sliced_values = values[keep]
    sliced_mask = mask[keep]
    effective_resolution = resolution or _infer_granularity_from_timestamps(sliced_ts)
    sliced_time = build_time_features(sliced_ts.to_numpy(), effective_resolution or "1h").astype(np.float32)
    return sliced_values.astype(np.float32), sliced_mask.astype(np.float32), sliced_time, effective_resolution


def _soft_mask_covariate_block(values, mask, min_valid_ratio=0.0, min_std=0.0):
    if values.size == 0 or mask.size == 0:
        return values, mask

    valid_ratio = float(np.sum(mask)) / float(mask.size)
    if valid_ratio < float(min_valid_ratio):
        return np.zeros_like(values, dtype=np.float32), np.zeros_like(mask, dtype=np.float32)

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
        return np.zeros_like(values, dtype=np.float32), np.zeros_like(mask, dtype=np.float32)

    return values, mask


def _should_resample_to_task_resolution(source_granularity, target_resolution):
    if not source_granularity or source_granularity == target_resolution:
        return False
    source_seconds = RESOLUTION_TO_SECONDS.get(source_granularity)
    target_seconds = RESOLUTION_TO_SECONDS.get(target_resolution)
    if source_seconds is None or target_seconds is None:
        return False
    return source_seconds < target_seconds


def _should_exact_align_to_task_resolution(source_granularity, target_resolution, mode):
    if (mode or "").strip().lower() != "exact":
        return False
    source_seconds = RESOLUTION_TO_SECONDS.get(source_granularity or "")
    target_seconds = RESOLUTION_TO_SECONDS.get(target_resolution or "")
    if source_seconds is None or target_seconds is None:
        return False
    return source_seconds <= target_seconds


def _infer_granularity_from_timestamps(timestamps):
    if len(timestamps) < 2:
        return ""
    ts = _normalize_timestamp_series(pd.Series(timestamps)).dropna().sort_values()
    if len(ts) < 2:
        return ""
    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return ""
    median_seconds = float(deltas.median())
    candidates = {
        "10sec": 10,
        "1min": 60,
        "5min": 300,
        "10min": 600,
        "15min": 900,
        "30min": 1800,
        "1h": 3600,
    }
    best_name = ""
    best_gap = None
    for name, seconds in candidates.items():
        gap = abs(median_seconds - seconds)
        if best_gap is None or gap < best_gap:
            best_name = name
            best_gap = gap
    return best_name


def _filter_exact_grid_timestamps(timestamps, resolution):
    if resolution == "1h":
        return timestamps.dt.minute.eq(0) & timestamps.dt.second.eq(0)
    if resolution == "15min":
        return timestamps.dt.minute.mod(15).eq(0) & timestamps.dt.second.eq(0)
    raise ValueError(f"Exact sampling is only supported for 1h/15min resolution, got {resolution}")


def _sample_frame_at_resolution(frame, value_cols, resolution):
    if frame.empty:
        return frame
    if resolution not in RESOLUTION_TO_PANDAS_FREQ:
        raise ValueError(f"Exact sampling requires a supported resolution, got {resolution}")
    work = frame.copy()
    work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    work = work.drop_duplicates(subset="timestamp", keep="last")
    aligned = work[_filter_exact_grid_timestamps(work["timestamp"], resolution)].copy()
    if aligned.empty:
        return pd.DataFrame({"timestamp": pd.to_datetime([]), **{col: pd.Series(dtype=np.float32) for col in value_cols}})
    freq = RESOLUTION_TO_PANDAS_FREQ[resolution]
    start = aligned["timestamp"].min().floor(freq)
    end = aligned["timestamp"].max().floor(freq)
    full_index = pd.date_range(start, end, freq=freq)
    exact = (
        aligned.set_index("timestamp")[value_cols]
        .reindex(full_index)
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    return exact


def _resample_frame_to_resolution(frame, value_cols, resolution, min_count=None):
    if frame.empty:
        return frame
    freq = RESOLUTION_TO_PANDAS_FREQ[resolution]
    work = frame.copy()
    work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    work = work.drop_duplicates(subset="timestamp", keep="last")
    work = work.set_index("timestamp")
    resampled = work[value_cols].resample(freq).mean()
    if min_count is not None and value_cols:
        counts = work[value_cols[0]].resample(freq).count()
        invalid = counts < int(min_count)
        if invalid.any():
            resampled.loc[invalid, value_cols] = np.nan
    resampled = resampled.reset_index()
    return resampled


@dataclass
class ChronosCoreSample:
    past_target: np.ndarray
    past_observed_mask: np.ndarray
    historical_covariates: np.ndarray
    historical_covariates_mask: np.ndarray
    historical_covariates_native: np.ndarray
    historical_covariates_native_mask: np.ndarray
    future_covariates: np.ndarray
    future_covariates_mask: np.ndarray
    future_covariates_native: np.ndarray
    future_covariates_native_mask: np.ndarray
    future_target: np.ndarray
    future_observed_mask: np.ndarray
    past_time_features: np.ndarray
    future_time_features: np.ndarray
    historical_covariates_native_time_features: np.ndarray
    future_covariates_native_time_features: np.ndarray
    static_features: np.ndarray


@dataclass
class FoundationSample:
    task_name: str
    task_family: str
    station_id: str
    resolution: str
    seq_len: int
    label_len: int
    pred_len: int
    chronos_core: ChronosCoreSample
    task_adapter: object
    metadata: dict

    @property
    def x(self):
        pieces = [self.chronos_core.past_target]
        if self.chronos_core.historical_covariates.shape[-1] > 0:
            pieces.append(self.chronos_core.historical_covariates)
        return np.concatenate(pieces, axis=-1).astype(np.float32)

    @property
    def y(self):
        decoder_target = self.chronos_core.past_target[-self.label_len:]
        return np.concatenate([decoder_target, self.chronos_core.future_target], axis=0).astype(np.float32)

    @property
    def x_time(self):
        return self.chronos_core.past_time_features.astype(np.float32)

    @property
    def y_time(self):
        decoder_time = self.chronos_core.past_time_features[-self.label_len:]
        return np.concatenate([decoder_time, self.chronos_core.future_time_features], axis=0).astype(np.float32)

    @property
    def y_observed_mask(self):
        decoder_mask = self.chronos_core.past_observed_mask[-self.label_len:]
        return np.concatenate([decoder_mask, self.chronos_core.future_observed_mask], axis=0).astype(np.float32)


class PVTaskDataset(Dataset):
    """Chronos-like forecasting samples with an extra PV task-spec adapter layer."""

    def __init__(
        self,
        manifest_path,
        station_data_root,
        task_name,
        split="train",
        regions=None,
        station_dirs=None,
        max_stations=0,
        scale=True,
        data_file_name=None,
        feature_cols=None,
        target_col_override=None,
        time_col_override=None,
        history_covariate_file_name=None,
        history_covariate_cols=None,
        history_covariate_time_col="datetime",
        future_covariate_file_name=None,
        future_covariate_cols=None,
        future_covariate_time_col="datetime",
        strict_hourly_resample=False,
        hourly_resample_mode="exact",
        sample_nan_ratio_threshold=0.0,
        min_past_target_valid_ratio=1.0,
        min_future_target_valid_ratio=1.0,
        min_future_target_std=0.0,
        min_history_covariate_valid_ratio=1.0,
        min_future_covariate_valid_ratio=1.0,
        min_history_covariate_std=0.0,
        min_future_covariate_std=0.0,
        max_future_zero_run_hours=0.0,
        max_future_constant_run_hours=0.0,
        future_zero_tolerance=1e-6,
        future_constant_tolerance=1e-6,
        min_future_target_range=0.0,
        min_station_valid_samples=0,
        target_transform="",
        target_normalization="none",
        target_standardization="auto",
        capacity_proxy_quantile=99.5,
        target_negative_sentinel=-1e5,
        night_small_negative_abs_kw=10.0,
        night_small_negative_capacity_frac=0.05,
        target_extreme_positive_capacity_frac=1.5,
        sample_index_csv="",
        dataset_cache_dir="",
        use_dataset_cache=False,
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.station_data_root = os.path.abspath(station_data_root)
        self.task_spec = resolve_task_spec(task_name)
        if self.task_spec is None:
            raise ValueError(f"Unknown task_name: {task_name}")
        self.task_name = task_name
        self.split = split
        self.scale = scale
        self.seq_len, self.label_len, self.pred_len = resolve_task_lengths(self.task_spec)
        self.regions = regions or []
        self.station_dirs = station_dirs or []
        self.max_stations = max_stations
        self.data_file_name = data_file_name
        self.feature_cols = feature_cols or []
        self.target_col_override = target_col_override
        self.time_col_override = time_col_override
        self.history_covariate_file_name = history_covariate_file_name or ""
        self.history_covariate_cols = history_covariate_cols or []
        self.history_covariate_time_col = history_covariate_time_col or "datetime"
        self.future_covariate_file_name = future_covariate_file_name or ""
        self.future_covariate_cols = future_covariate_cols or []
        self.future_covariate_time_col = future_covariate_time_col or "datetime"
        self.strict_hourly_resample = bool(strict_hourly_resample)
        self.hourly_resample_mode = (hourly_resample_mode or "exact").strip().lower()
        self.sample_nan_ratio_threshold = float(sample_nan_ratio_threshold)
        self.min_past_target_valid_ratio = float(min_past_target_valid_ratio)
        self.min_future_target_valid_ratio = float(min_future_target_valid_ratio)
        self.min_future_target_std = float(min_future_target_std)
        self.min_history_covariate_valid_ratio = float(min_history_covariate_valid_ratio)
        self.min_future_covariate_valid_ratio = float(min_future_covariate_valid_ratio)
        self.min_history_covariate_std = float(min_history_covariate_std)
        self.min_future_covariate_std = float(min_future_covariate_std)
        self.max_future_zero_run_hours = float(max_future_zero_run_hours)
        self.max_future_constant_run_hours = float(max_future_constant_run_hours)
        self.future_zero_tolerance = float(future_zero_tolerance)
        self.future_constant_tolerance = float(future_constant_tolerance)
        self.min_future_target_range = float(min_future_target_range)
        self.min_station_valid_samples = int(min_station_valid_samples or 0)
        (
            self.target_transform,
            self.target_normalization,
            self.target_standardization,
        ) = _resolve_target_transform(
            target_transform,
            target_normalization,
            target_standardization,
        )
        self.capacity_proxy_quantile = float(capacity_proxy_quantile)
        self.target_negative_sentinel = float(target_negative_sentinel)
        self.night_small_negative_abs_kw = float(night_small_negative_abs_kw)
        self.night_small_negative_capacity_frac = float(night_small_negative_capacity_frac)
        self.target_extreme_positive_capacity_frac = float(target_extreme_positive_capacity_frac)
        self.sample_index_csv = str(sample_index_csv or "")
        self.sample_index_filter = SampleIndexFilter(self.sample_index_csv, split=self.split)
        self.dataset_cache_dir = str(dataset_cache_dir or "")
        self.use_dataset_cache = bool(use_dataset_cache and self.dataset_cache_dir)
        if self.sample_index_filter.strict:
            # Strict manifests must prove every row against freshly enumerated
            # source windows; a cached payload cannot provide that proof.
            self.use_dataset_cache = False
        self.task_adapter = build_task_adapter(
            self.task_spec,
            historical_covariate_cols=self.history_covariate_cols,
            future_covariate_cols=self.future_covariate_cols,
        )
        self.station_records = []
        self.station_payloads = []
        self.index = []
        self._load()

    def _source_file_signature(self, path):
        if not path or not os.path.exists(path):
            return {"path": os.path.abspath(path or ""), "exists": False}
        stat = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _dataset_cache_key(self, rows):
        source_files = [self._source_file_signature(self.manifest_path)]
        for row in rows:
            station_dir = os.path.join(self.station_data_root, row.get("station_dir", ""))
            pv_file = self.data_file_name or row.get("pv_file", "")
            source_files.append(self._source_file_signature(os.path.join(station_dir, pv_file)))
            source_files.append(self._source_file_signature(os.path.join(station_dir, "other_data.json")))
            if self.history_covariate_file_name:
                source_files.append(
                    self._source_file_signature(os.path.join(station_dir, self.history_covariate_file_name))
                )
            if self.future_covariate_file_name:
                source_files.append(
                    self._source_file_signature(os.path.join(station_dir, self.future_covariate_file_name))
                )
        payload = {
            "cache_version": 4,
            "manifest_path": self.manifest_path,
            "station_data_root": self.station_data_root,
            "task_name": self.task_name,
            "split": self.split,
            "scale": self.scale,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "regions": tuple(self.regions),
            "station_dirs": tuple(self.station_dirs),
            "max_stations": int(self.max_stations or 0),
            "data_file_name": self.data_file_name or "",
            "feature_cols": tuple(self.feature_cols),
            "target_col_override": self.target_col_override or "",
            "time_col_override": self.time_col_override or "",
            "history_covariate_file_name": self.history_covariate_file_name,
            "history_covariate_cols": tuple(self.history_covariate_cols),
            "history_covariate_time_col": self.history_covariate_time_col,
            "future_covariate_file_name": self.future_covariate_file_name,
            "future_covariate_cols": tuple(self.future_covariate_cols),
            "future_covariate_time_col": self.future_covariate_time_col,
            "strict_hourly_resample": self.strict_hourly_resample,
            "hourly_resample_mode": self.hourly_resample_mode,
            "sample_nan_ratio_threshold": self.sample_nan_ratio_threshold,
            "min_past_target_valid_ratio": self.min_past_target_valid_ratio,
            "min_future_target_valid_ratio": self.min_future_target_valid_ratio,
            "min_future_target_std": self.min_future_target_std,
            "min_history_covariate_valid_ratio": self.min_history_covariate_valid_ratio,
            "min_future_covariate_valid_ratio": self.min_future_covariate_valid_ratio,
            "min_history_covariate_std": self.min_history_covariate_std,
            "min_future_covariate_std": self.min_future_covariate_std,
            "max_future_zero_run_hours": self.max_future_zero_run_hours,
            "max_future_constant_run_hours": self.max_future_constant_run_hours,
            "future_zero_tolerance": self.future_zero_tolerance,
            "future_constant_tolerance": self.future_constant_tolerance,
            "min_future_target_range": self.min_future_target_range,
            "min_station_valid_samples": self.min_station_valid_samples,
            "target_transform": self.target_transform,
            "target_normalization": self.target_normalization,
            "target_standardization": self.target_standardization,
            "capacity_proxy_quantile": self.capacity_proxy_quantile,
            "target_negative_sentinel": self.target_negative_sentinel,
            "night_small_negative_abs_kw": self.night_small_negative_abs_kw,
            "night_small_negative_capacity_frac": self.night_small_negative_capacity_frac,
            "target_extreme_positive_capacity_frac": self.target_extreme_positive_capacity_frac,
            "sample_index_csv": self.sample_index_csv,
            "sample_index_file": self._source_file_signature(self.sample_index_csv),
            "station_rows": tuple(row.get("station_dir", "") for row in rows),
            "source_files": source_files,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _dataset_cache_path(self, cache_key):
        prefix = f"{self.task_name}_{self.split}_{self.task_spec.resolution}_{cache_key[:16]}"
        return os.path.join(self.dataset_cache_dir, f"{prefix}.pkl")

    def _try_load_dataset_cache(self, rows):
        if not self.use_dataset_cache:
            return False, ""
        cache_key = self._dataset_cache_key(rows)
        cache_path = self._dataset_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return False, cache_path
        started_at = time.time()
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
            self.station_records = payload["station_records"]
            self.station_payloads = payload["station_payloads"]
            self.index = payload["index"]
        except Exception as exc:
            print(
                f"[PVTaskDataset] cache_load_failed task={self.task_name} split={self.split} "
                f"path={cache_path} error={exc}",
                flush=True,
            )
            self.station_records = []
            self.station_payloads = []
            self.index = []
            return False, cache_path
        elapsed = time.time() - started_at
        print(
            f"[PVTaskDataset] cache_hit task={self.task_name} split={self.split} "
            f"stations={len(self.station_records)} samples={len(self.index)} elapsed={elapsed:.1f}s "
            f"path={cache_path}",
            flush=True,
        )
        return True, cache_path

    def _save_dataset_cache(self, cache_path):
        if not self.use_dataset_cache or not cache_path:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "station_records": self.station_records,
            "station_payloads": self.station_payloads,
            "index": self.index,
        }
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_pv_task_dataset_", suffix=".pkl", dir=os.path.dirname(cache_path))
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            print(
                f"[PVTaskDataset] cache_saved task={self.task_name} split={self.split} path={cache_path}",
                flush=True,
            )
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            print(
                f"[PVTaskDataset] cache_save_failed task={self.task_name} split={self.split} "
                f"path={cache_path} error={exc}",
                flush=True,
            )

    def _resample_station_frame(self, frame, value_cols, source_granularity):
        if self.hourly_resample_mode == "exact":
            return _sample_frame_at_resolution(frame, value_cols, self.task_spec.resolution)
        resample_min_count = self._resolve_resample_min_count(source_granularity)
        return _resample_frame_to_resolution(
            frame,
            value_cols,
            self.task_spec.resolution,
            min_count=resample_min_count,
        )

    def _load(self):
        started_at = time.time()
        rows = load_station_manifest(self.manifest_path)
        rows = filter_manifest_rows(
            rows,
            allowed_granularities=self.task_spec.compatible_granularities or (self.task_spec.resolution,),
            regions=self.regions,
            station_dirs=self.station_dirs,
            max_stations=self.max_stations,
        )
        cache_hit, cache_path = self._try_load_dataset_cache(rows)
        if cache_hit:
            return
        total_rows = len(rows)
        kept_rows = 0
        skipped_rows = 0
        progress_interval = 1 if total_rows <= 10 else 10 if total_rows <= 100 else 50
        print(
            f"[PVTaskDataset] start task={self.task_name} split={self.split} "
            f"stations_to_scan={total_rows} resolution={self.task_spec.resolution}",
            flush=True,
        )
        for row_index, row in enumerate(rows, start=1):
            payload = self._build_station_payload(row)
            if payload is None:
                skipped_rows += 1
                if row_index % progress_interval == 0 or row_index == total_rows:
                    print(
                        f"[PVTaskDataset] progress task={self.task_name} split={self.split} "
                        f"processed={row_index}/{total_rows} kept={kept_rows} skipped={skipped_rows}",
                        flush=True,
                    )
                continue
            station_index = len(self.station_records)
            self.station_records.append(row)
            self.station_payloads.append(payload)
            kept_rows += 1
            for sample_start in payload["valid_start_indices"]:
                self.index.append((station_index, sample_start))
            if row_index % progress_interval == 0 or row_index == total_rows:
                print(
                    f"[PVTaskDataset] progress task={self.task_name} split={self.split} "
                    f"processed={row_index}/{total_rows} kept={kept_rows} skipped={skipped_rows} "
                    f"samples={len(self.index)}",
                    flush=True,
                )
        self.sample_index_filter.assert_complete()
        elapsed = time.time() - started_at
        print(
            f"[PVTaskDataset] done task={self.task_name} split={self.split} "
            f"stations={kept_rows} skipped={skipped_rows} samples={len(self.index)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        self._save_dataset_cache(cache_path)

    def _resolve_resample_min_count(self, source_granularity):
        if not self.strict_hourly_resample:
            return None
        if self.task_spec.resolution != "1h":
            return None
        return MIN_POINTS_PER_HOUR.get(source_granularity or "", None)

    def _build_station_payload(self, row):
        csv_file_name = self.data_file_name or row["pv_file"]
        csv_path = os.path.join(self.station_data_root, row["station_dir"], csv_file_name)
        if not os.path.exists(csv_path):
            return None

        time_col = self.time_col_override or row.get("time_col", "")
        target_col = self.target_col_override or row.get("target_col", "")
        extra_feature_cols = [col for col in self.feature_cols if col and col != target_col]
        required_cols = [time_col, target_col, *extra_feature_cols]
        frame = _read_csv_usecols(csv_path, required_cols)
        if frame is None:
            return None
        if any(col not in frame.columns for col in required_cols):
            return None

        timestamps = _parse_timestamp_series_fast(frame[time_col])
        target_values = pd.to_numeric(frame[target_col], errors="coerce").astype(np.float32)
        target_values = target_values.mask(target_values <= self.target_negative_sentinel, np.nan)

        target_frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "target": target_values,
            }
        ).dropna(subset=["timestamp"])
        target_frame = target_frame.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
        source_granularity = row.get("granularity", "")
        if not source_granularity or source_granularity not in RESOLUTION_TO_SECONDS:
            source_granularity = _infer_granularity_from_timestamps(target_frame["timestamp"])
        if _should_exact_align_to_task_resolution(source_granularity, self.task_spec.resolution, self.hourly_resample_mode):
            target_frame = self._resample_station_frame(target_frame, ["target"], source_granularity)
        elif _should_resample_to_task_resolution(source_granularity, self.task_spec.resolution):
            target_frame = self._resample_station_frame(target_frame, ["target"], source_granularity)
        if len(target_frame) <= self.seq_len + self.pred_len:
            return None

        base_timestamps = target_frame["timestamp"].to_numpy()
        station_dir_path = os.path.join(self.station_data_root, row["station_dir"])
        other_data = _load_other_data_json(station_dir_path)
        timezone_offset_hours, timezone_name = _infer_timezone_offset_hours(
            row,
            other_data,
            reference_timestamp=target_frame["timestamp"].iloc[0] if len(target_frame) > 0 else None,
        )
        raw_power_array = target_frame[["target"]].to_numpy(dtype=np.float32)
        capacity_info = {
            "target_normalization_mode": "none",
            "power_semantics": "",
            "power_semantics_note": "",
            "power_unit_scale_to_kw": 1.0,
            "cap_meta_kw": np.nan,
            "cap_meta_field": "",
            "cap_meta_source": "",
            "ratio_p99_5_to_cap": np.nan,
            "cap_status": "",
            "cap_note": "",
            "capacity_used_kw": np.nan,
            "capacity_used_source": "",
            "target_to_power_scale": 1.0,
        }
        target_array = raw_power_array.copy()
        clean_power_kw = raw_power_array.astype(np.float32)

        if self.target_normalization == "capacity_factor":
            train_count = int(len(raw_power_array) * 0.7)
            train_power_series = raw_power_array[:train_count, 0]
            capacity_info = _resolve_capacity_normalization(train_power_series, other_data, self.capacity_proxy_quantile)
            scale_to_kw = float(capacity_info.get("power_unit_scale_to_kw", 1.0) or 1.0)
            target_power_kw = raw_power_array.astype(np.float32) * scale_to_kw
            capacity_used_kw = capacity_info.get("capacity_used_kw", np.nan)
            clean_power_kw, _ = _clean_pv_power_values(
                target_power_kw,
                base_timestamps,
                row,
                timezone_offset_hours,
                capacity_used_kw=capacity_used_kw,
                sentinel_negative=self.target_negative_sentinel * scale_to_kw,
                night_small_negative_abs_kw=self.night_small_negative_abs_kw,
                night_small_negative_capacity_frac=self.night_small_negative_capacity_frac,
                max_positive_capacity_frac=self.target_extreme_positive_capacity_frac,
            )
            if capacity_info.get("power_semantics") == "already_normalized":
                target_array = clean_power_kw.copy()
                capacity_info["target_normalization_mode"] = "already_normalized"
                capacity_info["target_to_power_scale"] = float(capacity_used_kw) if np.isfinite(capacity_used_kw) and capacity_used_kw > 0 else 1.0
            elif np.isfinite(capacity_used_kw) and float(capacity_used_kw) > 0:
                target_array = (clean_power_kw / float(capacity_used_kw)).astype(np.float32)
                capacity_info["target_normalization_mode"] = "capacity_factor"
                capacity_info["target_to_power_scale"] = float(capacity_used_kw)
            else:
                target_array = clean_power_kw.astype(np.float32)
                capacity_info["target_normalization_mode"] = "power_kw_fallback"
                capacity_info["target_to_power_scale"] = 1.0
        else:
            clean_power_kw, _ = _clean_pv_power_values(
                raw_power_array.astype(np.float32),
                base_timestamps,
                row,
                timezone_offset_hours,
                capacity_used_kw=np.nan,
                sentinel_negative=self.target_negative_sentinel,
                night_small_negative_abs_kw=self.night_small_negative_abs_kw,
                night_small_negative_capacity_frac=self.night_small_negative_capacity_frac,
                max_positive_capacity_frac=0.0,
            )
            target_array = clean_power_kw.copy()

        target_mask = ~np.isnan(target_array)
        target_array[~target_mask] = 0.0

        base_extra_cov = None
        if extra_feature_cols:
            extra_frame = pd.DataFrame({"timestamp": timestamps})
            for col in extra_feature_cols:
                extra_frame[col] = pd.to_numeric(frame[col], errors="coerce")
            extra_frame = extra_frame.dropna(subset=["timestamp"]).sort_values("timestamp")
            extra_frame = extra_frame.drop_duplicates(subset="timestamp", keep="last")
            if _should_exact_align_to_task_resolution(source_granularity, self.task_spec.resolution, self.hourly_resample_mode):
                extra_frame = self._resample_station_frame(extra_frame, extra_feature_cols, source_granularity)
            elif _should_resample_to_task_resolution(source_granularity, self.task_spec.resolution):
                extra_frame = self._resample_station_frame(extra_frame, extra_feature_cols, source_granularity)
            base_extra_cov = extra_frame

        history_cov_frame = _build_covariate_frame(
            os.path.join(self.station_data_root, row["station_dir"], self.history_covariate_file_name),
            self.history_covariate_time_col,
            self.history_covariate_cols,
        )
        future_cov_frame = _build_covariate_frame(
            os.path.join(self.station_data_root, row["station_dir"], self.future_covariate_file_name),
            self.future_covariate_time_col,
            self.future_covariate_cols,
        )
        history_native = _prepare_native_covariate_arrays(history_cov_frame, self.history_covariate_cols)
        future_native = _prepare_native_covariate_arrays(future_cov_frame, self.future_covariate_cols)

        history_values, history_mask = _align_covariates(
            base_timestamps=base_timestamps,
            covariate_frame=history_cov_frame,
            covariate_cols=self.history_covariate_cols,
            base_resolution=self.task_spec.resolution,
            allow_coarse_hold=True,
        )
        future_values, future_mask = _align_covariates(
            base_timestamps=base_timestamps,
            covariate_frame=future_cov_frame,
            covariate_cols=self.future_covariate_cols,
            base_resolution=self.task_spec.resolution,
            allow_coarse_hold=True,
        )

        if extra_feature_cols:
            extra_values, extra_mask = _align_covariates(
                base_timestamps=base_timestamps,
                covariate_frame=base_extra_cov,
                covariate_cols=extra_feature_cols,
                base_resolution=self.task_spec.resolution,
                allow_coarse_hold=False,
            )
            if history_values.shape[-1] == 0:
                history_values = extra_values
                history_mask = extra_mask
            else:
                history_values = np.concatenate([history_values, extra_values], axis=-1)
                history_mask = np.concatenate([history_mask, extra_mask], axis=-1)

        if len(target_array) <= self.seq_len + self.pred_len:
            return None

        num_train = int(len(target_array) * 0.7)
        num_test = int(len(target_array) * 0.2)
        num_val = len(target_array) - num_train - num_test
        border1s = [0, max(0, num_train - self.seq_len), max(0, len(target_array) - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_val, len(target_array)]
        if self.split == "all":
            border1 = 0
            border2 = len(target_array)
        else:
            split_to_id = {"train": 0, "val": 1, "test": 2}
            split_id = split_to_id[self.split]
            border1 = border1s[split_id]
            border2 = border2s[split_id]
        if self.sample_index_filter.enabled:
            border1 = 0
            border2 = len(target_array)
        if border2 - border1 <= self.seq_len + self.pred_len:
            return None

        train_slice = slice(border1s[0], border2s[0])
        target_scaler = _fit_scaler_on_observed(
            target_array[train_slice],
            target_mask[train_slice],
            use_scale=(self.scale and self.target_standardization == "standard"),
        )
        history_scaler = _fit_scaler_on_observed(history_values[train_slice], history_mask[train_slice], use_scale=self.scale)
        future_scaler = _fit_scaler_on_observed(future_values[train_slice], future_mask[train_slice], use_scale=self.scale)

        scaled_target = _transform_with_mask(target_array, target_mask, target_scaler)
        scaled_history = _transform_with_mask(history_values, history_mask, history_scaler) if history_scaler is not None else history_values
        scaled_future = _transform_with_mask(future_values, future_mask, future_scaler) if future_scaler is not None else future_values
        scaled_history_native = _transform_native_covariates(history_native, history_scaler)
        scaled_future_native = _transform_native_covariates(future_native, future_scaler)
        time_values = build_time_features(base_timestamps, self.task_spec.resolution)

        payload = {
            "timestamps": base_timestamps[border1:border2],
            "target_raw_values": target_array[border1:border2].astype(np.float32),
            "target_power_values": clean_power_kw[border1:border2].astype(np.float32),
            "target_values": scaled_target[border1:border2],
            "target_mask": target_mask[border1:border2].astype(np.float32),
            "history_values": scaled_history[border1:border2].astype(np.float32),
            "history_mask": history_mask[border1:border2].astype(np.float32),
            "future_values": scaled_future[border1:border2].astype(np.float32),
            "future_mask": future_mask[border1:border2].astype(np.float32),
            "history_native": scaled_history_native,
            "future_native": scaled_future_native,
            "time_values": time_values[border1:border2].astype(np.float32),
            "num_samples": border2 - border1 - self.seq_len - self.pred_len + 1,
            "target_scaler": target_scaler,
            "history_scaler": history_scaler,
            "future_scaler": future_scaler,
            "target_col": target_col,
            "history_covariate_cols": tuple(self.history_covariate_cols + extra_feature_cols),
            "future_covariate_cols": tuple(self.future_covariate_cols),
            "data_file_name": csv_file_name,
            "history_covariate_file_name": self.history_covariate_file_name,
            "future_covariate_file_name": self.future_covariate_file_name,
            "target_transform": self.target_transform,
            "target_normalization_mode": capacity_info.get("target_normalization_mode", "none"),
            "target_standardization": self.target_standardization,
            "power_semantics": capacity_info.get("power_semantics", ""),
            "power_semantics_note": capacity_info.get("power_semantics_note", ""),
            "power_unit_scale_to_kw": float(capacity_info.get("power_unit_scale_to_kw", 1.0) or 1.0),
            "cap_meta_kw": capacity_info.get("cap_meta_kw", np.nan),
            "cap_meta_field": capacity_info.get("cap_meta_field", ""),
            "cap_meta_source": capacity_info.get("cap_meta_source", ""),
            "ratio_p99_5_to_cap": capacity_info.get("ratio_p99_5_to_cap", np.nan),
            "cap_status": capacity_info.get("cap_status", ""),
            "cap_note": capacity_info.get("cap_note", ""),
            "capacity_used_kw": capacity_info.get("capacity_used_kw", np.nan),
            "capacity_used_source": capacity_info.get("capacity_used_source", ""),
            "target_to_power_scale": float(capacity_info.get("target_to_power_scale", 1.0) or 1.0),
            "timezone_offset_hours": float(timezone_offset_hours) if np.isfinite(timezone_offset_hours) else np.nan,
            "timezone_name": timezone_name,
        }
        payload["num_possible_samples"] = payload["num_samples"]
        payload["valid_start_indices"] = compute_valid_window_starts(
            target_mask=payload["target_mask"],
            target_values=payload["target_raw_values"],
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            resolution=self.task_spec.resolution,
            sample_nan_ratio_threshold=self.sample_nan_ratio_threshold,
            history_covariate_mask=payload["history_mask"],
            future_covariate_mask=payload["future_mask"],
            min_past_target_valid_ratio=self.min_past_target_valid_ratio,
            min_future_target_valid_ratio=self.min_future_target_valid_ratio,
            min_history_covariate_valid_ratio=self.min_history_covariate_valid_ratio,
            min_future_covariate_valid_ratio=self.min_future_covariate_valid_ratio,
            min_future_target_std=self.min_future_target_std,
            max_future_zero_run_hours=self.max_future_zero_run_hours,
            max_future_constant_run_hours=self.max_future_constant_run_hours,
            future_zero_tolerance=self.future_zero_tolerance,
            future_constant_tolerance=self.future_constant_tolerance,
            min_future_target_range=self.min_future_target_range,
        )
        payload["valid_start_indices"] = filter_valid_starts_by_sample_index(
            payload["valid_start_indices"],
            payload["timestamps"],
            self.seq_len,
            self.pred_len,
            row["station_dir"],
            row.get("station_id", ""),
            self.sample_index_filter,
        )
        payload["num_samples"] = len(payload["valid_start_indices"])
        if self.min_station_valid_samples > 0 and payload["num_samples"] < self.min_station_valid_samples:
            return None
        if payload["num_samples"] <= 0:
            return None
        return payload

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        station_index, sample_index = self.index[item]
        row = self.station_records[station_index]
        payload = self.station_payloads[station_index]

        s_begin = sample_index
        s_end = s_begin + self.seq_len
        f_begin = s_end
        f_end = f_begin + self.pred_len

        past_target = payload["target_values"][s_begin:s_end]
        past_mask = payload["target_mask"][s_begin:s_end]
        future_target = payload["target_values"][f_begin:f_end]
        future_mask = payload["target_mask"][f_begin:f_end]
        history_cov = payload["history_values"][s_begin:s_end]
        history_cov_mask = payload["history_mask"][s_begin:s_end]
        future_cov = payload["future_values"][f_begin:f_end]
        future_cov_mask = payload["future_mask"][f_begin:f_end]
        past_time = payload["time_values"][s_begin:s_end]
        future_time = payload["time_values"][f_begin:f_end]
        hist_native_values, hist_native_mask, hist_native_time, hist_native_resolution = _slice_native_covariate_window(
            payload.get("history_native"),
            payload["timestamps"][s_begin],
            payload["timestamps"][s_end - 1],
        )
        fut_native_values, fut_native_mask, fut_native_time, fut_native_resolution = _slice_native_covariate_window(
            payload.get("future_native"),
            payload["timestamps"][f_begin],
            payload["timestamps"][f_end - 1],
        )

        history_cov, history_cov_mask = _soft_mask_covariate_block(
            history_cov,
            history_cov_mask,
            min_valid_ratio=self.min_history_covariate_valid_ratio,
            min_std=self.min_history_covariate_std,
        )
        future_cov, future_cov_mask = _soft_mask_covariate_block(
            future_cov,
            future_cov_mask,
            min_valid_ratio=self.min_future_covariate_valid_ratio,
            min_std=self.min_future_covariate_std,
        )

        static_features = np.array(
            [
                float(self.seq_len),
                float(self.pred_len),
                float(self.label_len),
                float(self.task_spec.horizon_hours),
            ],
            dtype=np.float32,
        )

        return FoundationSample(
            task_name=self.task_name,
            task_family=self.task_spec.task_family,
            station_id=row["station_dir"],
            resolution=self.task_spec.resolution,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            chronos_core=ChronosCoreSample(
                past_target=past_target,
                past_observed_mask=past_mask,
                historical_covariates=history_cov,
                historical_covariates_mask=history_cov_mask,
                historical_covariates_native=hist_native_values,
                historical_covariates_native_mask=hist_native_mask,
                future_covariates=future_cov,
                future_covariates_mask=future_cov_mask,
                future_covariates_native=fut_native_values,
                future_covariates_native_mask=fut_native_mask,
                future_target=future_target,
                future_observed_mask=future_mask,
                past_time_features=past_time,
                future_time_features=future_time,
                historical_covariates_native_time_features=hist_native_time,
                future_covariates_native_time_features=fut_native_time,
                static_features=static_features,
            ),
            task_adapter=self.task_adapter,
            metadata={
                "station_record": row,
                "timestamps": payload["timestamps"][s_begin:f_end],
                "target_scaler": payload["target_scaler"],
                "history_scaler": payload["history_scaler"],
                "future_scaler": payload["future_scaler"],
                "y_scaler": payload["target_scaler"],
                "feature_cols": payload["history_covariate_cols"],
                "history_covariate_cols": payload["history_covariate_cols"],
                "future_covariate_cols": payload["future_covariate_cols"],
                "data_file_name": payload["data_file_name"],
                "history_covariate_file_name": payload["history_covariate_file_name"],
                "future_covariate_file_name": payload["future_covariate_file_name"],
                "target_transform": payload.get("target_transform", ""),
                "historical_covariates_native_resolution": hist_native_resolution,
                "future_covariates_native_resolution": fut_native_resolution,
                "target_normalization_mode": payload.get("target_normalization_mode", "none"),
                "target_standardization": payload.get("target_standardization", "standard"),
                "power_semantics": payload.get("power_semantics", ""),
                "power_semantics_note": payload.get("power_semantics_note", ""),
                "power_unit_scale_to_kw": payload.get("power_unit_scale_to_kw", 1.0),
                "cap_meta_kw": payload.get("cap_meta_kw", np.nan),
                "cap_meta_field": payload.get("cap_meta_field", ""),
                "cap_meta_source": payload.get("cap_meta_source", ""),
                "ratio_p99_5_to_cap": payload.get("ratio_p99_5_to_cap", np.nan),
                "cap_status": payload.get("cap_status", ""),
                "cap_note": payload.get("cap_note", ""),
                "capacity_used_kw": payload.get("capacity_used_kw", np.nan),
                "capacity_used_source": payload.get("capacity_used_source", ""),
                "target_to_power_scale": payload.get("target_to_power_scale", 1.0),
            },
        )


def build_multitask_dataset(
    manifest_path,
    station_data_root,
    task_names,
    split="train",
    regions=None,
    station_dirs=None,
    max_stations=0,
    scale=True,
    data_file_name=None,
    feature_cols=None,
    target_col_override=None,
    time_col_override=None,
    history_covariate_file_name=None,
    history_covariate_cols=None,
    history_covariate_time_col="datetime",
    future_covariate_file_name=None,
    future_covariate_cols=None,
    future_covariate_time_col="datetime",
    sample_nan_ratio_threshold=0.0,
    min_past_target_valid_ratio=1.0,
    min_future_target_valid_ratio=1.0,
    min_future_target_std=0.0,
    min_history_covariate_valid_ratio=1.0,
    min_future_covariate_valid_ratio=1.0,
    min_history_covariate_std=0.0,
    min_future_covariate_std=0.0,
    max_future_zero_run_hours=0.0,
    max_future_constant_run_hours=0.0,
    future_zero_tolerance=1e-6,
    future_constant_tolerance=1e-6,
    min_future_target_range=0.0,
    min_station_valid_samples=0,
    target_transform="",
    target_normalization="none",
    target_standardization="auto",
    capacity_proxy_quantile=99.5,
    target_negative_sentinel=-1e5,
    night_small_negative_abs_kw=10.0,
    night_small_negative_capacity_frac=0.05,
    target_extreme_positive_capacity_frac=1.5,
    sample_index_csv="",
):
    datasets = []
    for task_name in task_names:
        dataset = PVTaskDataset(
            manifest_path=manifest_path,
            station_data_root=station_data_root,
            task_name=task_name,
            split=split,
            regions=regions,
            station_dirs=station_dirs,
            max_stations=max_stations,
            scale=scale,
            data_file_name=data_file_name,
            feature_cols=feature_cols,
            target_col_override=target_col_override,
            time_col_override=time_col_override,
            history_covariate_file_name=history_covariate_file_name,
            history_covariate_cols=history_covariate_cols,
            history_covariate_time_col=history_covariate_time_col,
            future_covariate_file_name=future_covariate_file_name,
            future_covariate_cols=future_covariate_cols,
            future_covariate_time_col=future_covariate_time_col,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            min_past_target_valid_ratio=min_past_target_valid_ratio,
            min_future_target_valid_ratio=min_future_target_valid_ratio,
            min_future_target_std=min_future_target_std,
            min_history_covariate_valid_ratio=min_history_covariate_valid_ratio,
            min_future_covariate_valid_ratio=min_future_covariate_valid_ratio,
            min_history_covariate_std=min_history_covariate_std,
            min_future_covariate_std=min_future_covariate_std,
            max_future_zero_run_hours=max_future_zero_run_hours,
            max_future_constant_run_hours=max_future_constant_run_hours,
            future_zero_tolerance=future_zero_tolerance,
            future_constant_tolerance=future_constant_tolerance,
            min_future_target_range=min_future_target_range,
            min_station_valid_samples=min_station_valid_samples,
            target_transform=target_transform,
            target_normalization=target_normalization,
            target_standardization=target_standardization,
            capacity_proxy_quantile=capacity_proxy_quantile,
            target_negative_sentinel=target_negative_sentinel,
            night_small_negative_abs_kw=night_small_negative_abs_kw,
            night_small_negative_capacity_frac=night_small_negative_capacity_frac,
            target_extreme_positive_capacity_frac=target_extreme_positive_capacity_frac,
            sample_index_csv=sample_index_csv,
        )
        if len(dataset) > 0:
            datasets.append(dataset)

    if not datasets:
        raise ValueError("No non-empty datasets were built for the requested task list.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def summarize_task_datasets(
    task_names,
    manifest_path,
    station_data_root,
    split="train",
    regions=None,
    station_dirs=None,
    max_stations=0,
    scale=True,
    data_file_name=None,
    feature_cols=None,
    target_col_override=None,
    time_col_override=None,
    history_covariate_file_name=None,
    history_covariate_cols=None,
    history_covariate_time_col="datetime",
    future_covariate_file_name=None,
    future_covariate_cols=None,
    future_covariate_time_col="datetime",
    sample_nan_ratio_threshold=0.0,
    min_past_target_valid_ratio=1.0,
    min_future_target_valid_ratio=1.0,
    min_future_target_std=0.0,
    min_history_covariate_valid_ratio=1.0,
    min_future_covariate_valid_ratio=1.0,
    min_history_covariate_std=0.0,
    min_future_covariate_std=0.0,
    max_future_zero_run_hours=0.0,
    max_future_constant_run_hours=0.0,
    future_zero_tolerance=1e-6,
    future_constant_tolerance=1e-6,
    min_future_target_range=0.0,
    min_station_valid_samples=0,
    sample_index_csv="",
):
    summary = []
    for task_name in task_names:
        dataset = PVTaskDataset(
            manifest_path=manifest_path,
            station_data_root=station_data_root,
            task_name=task_name,
            split=split,
            regions=regions,
            station_dirs=station_dirs,
            max_stations=max_stations,
            scale=scale,
            data_file_name=data_file_name,
            feature_cols=feature_cols,
            target_col_override=target_col_override,
            time_col_override=time_col_override,
            history_covariate_file_name=history_covariate_file_name,
            history_covariate_cols=history_covariate_cols,
            history_covariate_time_col=history_covariate_time_col,
            future_covariate_file_name=future_covariate_file_name,
            future_covariate_cols=future_covariate_cols,
            future_covariate_time_col=future_covariate_time_col,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            min_past_target_valid_ratio=min_past_target_valid_ratio,
            min_future_target_valid_ratio=min_future_target_valid_ratio,
            min_future_target_std=min_future_target_std,
            min_history_covariate_valid_ratio=min_history_covariate_valid_ratio,
            min_future_covariate_valid_ratio=min_future_covariate_valid_ratio,
            min_history_covariate_std=min_history_covariate_std,
            min_future_covariate_std=min_future_covariate_std,
            max_future_zero_run_hours=max_future_zero_run_hours,
            max_future_constant_run_hours=max_future_constant_run_hours,
            future_zero_tolerance=future_zero_tolerance,
            future_constant_tolerance=future_constant_tolerance,
            min_future_target_range=min_future_target_range,
            min_station_valid_samples=min_station_valid_samples,
            sample_index_csv=sample_index_csv,
        )
        summary.append(
            {
                "task_name": task_name,
                "resolution": dataset.task_spec.resolution,
                "seq_len": dataset.seq_len,
                "pred_len": dataset.pred_len,
                "label_len": dataset.label_len,
                "num_samples": len(dataset),
                "num_stations": len(dataset.station_records),
                "input_dim": dataset[0].x.shape[-1] if len(dataset) else 0,
                "output_dim": dataset[0].y.shape[-1] if len(dataset) else 0,
                "historical_covariate_dim": dataset[0].chronos_core.historical_covariates.shape[-1] if len(dataset) else 0,
                "future_covariate_dim": dataset[0].chronos_core.future_covariates.shape[-1] if len(dataset) else 0,
            }
        )
    return summary
