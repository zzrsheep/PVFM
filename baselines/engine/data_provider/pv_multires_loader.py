import csv
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from foundation.data.alignment.window_qc import compute_valid_window_starts
from foundation.data.alignment.virtual_qc import VirtualQCRelease
from data_provider.weather_mapping import get_feature_spec
from utils.sample_index import SampleIndexFilter, filter_valid_starts_by_sample_index
from utils.timefeatures import time_features


DATETIME_FORMATS = [
    '%Y-%m-%d %H:%M:%S%z',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y/%m/%d %H:%M',
    '%m/%d/%Y %H:%M',
    '%m/%d/%Y %H:%M:%S',
]

GRANULARITY_TO_FREQ = {
    '10sec': 's',
    '15sec': 's',
    '30sec': 's',
    '1min': 't',
    '5min': 't',
    '10min': 't',
    '15min': 't',
    '30min': 't',
    '35min': 't',
    '40min': 't',
    '1h': 'h',
}


def _is_primary_process():
    """Keep dataset audit logs readable when Accelerate starts multiple ranks."""
    return os.environ.get('RANK', '0') == '0'

STEPS_PER_HOUR = {
    '10sec': 360,
    '15sec': 240,
    '30sec': 120,
    '1min': 60,
    '5min': 12,
    '10min': 6,
    '15min': 4,
    '30min': 2,
    '1h': 1,
}

TASK_SPECS = {
    '1min_intraday_1h': {
        'granularity': '1min',
        'horizon_hours': 1,
        'context_ratio': 3.0,
    },
    '5min_dayahead_24h': {
        'granularity': '5min',
        'horizon_hours': 24,
        'context_ratio': 3.0,
    },
    '15min_dayahead_24h': {
        'granularity': '15min',
        'horizon_hours': 24,
        'context_ratio': 3.0,
    },
    '1h_dayahead_24h': {
        'granularity': '1h',
        'horizon_hours': 24,
        'context_ratio': 3.0,
    },
    '1h_short_6h': {
        'granularity': '1h',
        'horizon_hours': 6,
        'context_ratio': 4.0,
    },
    '1h_week_168h': {
        'granularity': '1h',
        'horizon_hours': 168,
        'context_ratio': 2.0,
    },
}

NATIVE_EQUAL_CONTEXT_MODELS = {'Cross_Unet', 'FusionSFNoSpatial', 'FusionSFMasked'}


def _tide_time_covariates(timestamps):
    """Match TiDE ``TimeCovariates(..., holiday=False)`` exactly."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors='coerce'))
    if index.isna().any():
        raise ValueError("TiDEOfficial received invalid timestamps")
    return np.stack(
        [
            index.minute.to_numpy(dtype=np.float32) / 59.0 - 0.5,
            index.hour.to_numpy(dtype=np.float32) / 23.0 - 0.5,
            index.day.to_numpy(dtype=np.float32) / 30.0 - 0.5,
            index.dayofweek.to_numpy(dtype=np.float32) / 6.0 - 0.5,
            index.dayofyear.to_numpy(dtype=np.float32) / 364.0 - 0.5,
            index.month.to_numpy(dtype=np.float32) / 11.0 - 0.5,
            index.strftime('%U').astype(np.float32) / 51.0 - 0.5,
        ],
        axis=0,
    ).astype(np.float32)

MIN_POINTS_PER_HOUR = {
    '1min': 45,
    '5min': 9,
    '10min': 5,
    '15min': 3,
    '30min': 2,
    '1h': 1,
}


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
    station_md = payload.get('station_metadata', {})
    return station_md if isinstance(station_md, dict) else {}


def _safe_float_from_dict(payload, key):
    if not isinstance(payload, dict):
        return None
    return _maybe_float(payload.get(key))


def _load_other_data_json_for_station(csv_path):
    station_dir = os.path.dirname(os.path.abspath(csv_path))
    path = os.path.join(station_dir, 'other_data.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return None


def _extract_capacity_candidates(payload):
    candidates = []
    candidate_specs = [
        ('station_metadata.capacity_kw', ('station_metadata', 'capacity_kw'), 1.0),
        ('station_metadata.summary_capacity_kw', ('station_metadata', 'summary_capacity_kw'), 1.0),
        ('station_metadata.official_dc_capacity_kW', ('station_metadata', 'official_dc_capacity_kW'), 1.0),
        ('dataset_metadata.Capacity', ('dataset_metadata', 'Capacity'), 1.0),
        ('top.capacity_kw', ('capacity_kw',), 1.0),
        ('top.summary_capacity_kw', ('summary_capacity_kw',), 1.0),
        ('top.official_dc_capacity_kW', ('official_dc_capacity_kW',), 1.0),
        ('top.dc_capacity_kW', ('dc_capacity_kW',), 1.0),
        ('top.capacity_mw', ('capacity_mw',), 1000.0),
        ('top.capacity_MW', ('capacity_MW',), 1000.0),
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
        source = ''
        if field_path.startswith('station_metadata.') and isinstance(payload.get('station_metadata'), dict):
            source = str(payload['station_metadata'].get('capacity_source', '')).strip()
        candidates.append({
            'value_kw': float(numeric) * float(multiplier),
            'field_path': field_path,
            'source': source,
        })
    return candidates


def _choose_capacity_candidate(candidates):
    if not candidates:
        return None
    preferred = [
        'station_metadata.capacity_kw',
        'station_metadata.summary_capacity_kw',
        'station_metadata.official_dc_capacity_kW',
        'dataset_metadata.Capacity',
    ]
    for field_path in preferred:
        for candidate in candidates:
            if candidate['field_path'] == field_path:
                return candidate
    return candidates[0]


def _classify_capacity_issue(ratio, ok_low, ok_high, suspect_low, suspect_high):
    if ratio is None or not np.isfinite(ratio):
        return 'missing_or_unusable', 'metadata_capacity_missing_or_station_unusable'
    if 5e-4 <= ratio <= 2e-3:
        return 'unit_mismatch_x1000', 'observed_power_is_about_1_over_1000_of_capacity_metadata'
    if 5e2 <= ratio <= 2e3:
        return 'unit_mismatch_x1000', 'observed_power_is_about_1000_times_capacity_metadata'
    if ok_low <= ratio <= ok_high:
        return 'cap_ok', ''
    if suspect_low <= ratio <= suspect_high:
        return 'cap_suspect', 'metadata_capacity_is_plausible_but_ratio_is_outside_ok_band'
    return 'cap_bad_or_missing', 'metadata_capacity_failed_sanity_check'


def _infer_power_semantics(payload, power_p99_5_kw, power_max_kw, cap_meta_kw, ok_low, ok_high, suspect_low, suspect_high):
    payload = payload or {}
    station_md = _extract_station_metadata(payload)
    source_type = str(payload.get('source_type', '')).strip()
    resolution_note = str(payload.get('resolution_note', '')).strip().lower()
    notes = str(station_md.get('notes', '')).strip().lower()
    max_norm = _safe_float_from_dict(station_md, 'max_norm')
    min_norm = _safe_float_from_dict(station_md, 'min_norm')

    if source_type == 'processed_station_series_from_dataset_only_pv_stations':
        return 'already_normalized', 'source_type_processed_station_series_from_dataset_only_pv_stations', 1.0
    if 'normalized_output' in resolution_note:
        return 'already_normalized', 'resolution_note_mentions_normalized_output', 1.0
    if max_norm is not None and power_max_kw is not None and max_norm <= 1.5 and float(power_max_kw) <= 1.5:
        return 'already_normalized', 'max_norm_and_power_max_look_normalized', 1.0
    if (
        'capacity_is_estimated_not_nameplate' in notes
        and max_norm is not None and max_norm <= 1.5
        and min_norm is not None and min_norm >= -0.1
    ):
        return 'already_normalized', 'notes_and_norm_range_look_normalized', 1.0

    if cap_meta_kw is not None and np.isfinite(cap_meta_kw) and cap_meta_kw > 0 and power_p99_5_kw is not None and np.isfinite(power_p99_5_kw):
        raw_ratio = float(power_p99_5_kw) / float(cap_meta_kw)
        mw_scaled_ratio = float(power_p99_5_kw) * 1000.0 / float(cap_meta_kw)
        if ok_low <= mw_scaled_ratio <= ok_high:
            return 'power_unit_mw_suspect', 'power_times_1000_matches_capacity_ratio_ok_band', 1000.0
        if suspect_low <= mw_scaled_ratio <= suspect_high:
            return 'power_unit_mw_suspect', 'power_times_1000_matches_capacity_ratio_suspect_band', 1000.0
        if 5e-4 <= raw_ratio <= 2e-3:
            return 'power_unit_mw_suspect', 'raw_ratio_near_1_over_1000_and_power_times_1000_is_more_plausible', 1000.0

    return 'power_unit_kw_plausible', '', 1.0


def _resolve_capacity_normalization(power_series, other_data, proxy_quantile):
    candidates = _extract_capacity_candidates(other_data or {})
    chosen = _choose_capacity_candidate(candidates)
    cap_meta_kw = chosen['value_kw'] if chosen is not None else np.nan
    cap_meta_field = chosen['field_path'] if chosen is not None else ''
    cap_meta_source = chosen['source'] if chosen is not None else ''

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
    if chosen is not None and chosen['value_kw'] > 0 and np.isfinite(adjusted_power_p99_5):
        ratio_p99_5_to_cap = float(adjusted_power_p99_5 / chosen['value_kw'])

    if power_semantics == 'already_normalized':
        cap_status = 'already_normalized'
        cap_note = power_semantics_note
        capacity_used_kw = chosen['value_kw'] if chosen is not None and chosen['value_kw'] > 0 else 1.0
        capacity_used_source = 'already_normalized'
    else:
        cap_status, cap_note = _classify_capacity_issue(
            None if not np.isfinite(ratio_p99_5_to_cap) else float(ratio_p99_5_to_cap),
            0.2,
            1.2,
            0.05,
            1.8,
        )
        if chosen is not None and cap_status in {'cap_ok', 'cap_suspect'}:
            capacity_used_kw = chosen['value_kw']
            capacity_used_source = 'metadata'
        else:
            capacity_used_kw = adjusted_power_p99_5 if np.isfinite(adjusted_power_p99_5) else np.nan
            capacity_used_source = 'proxy_p99_5'

    return {
        'power_semantics': power_semantics,
        'power_semantics_note': power_semantics_note,
        'power_unit_scale_to_kw': float(power_unit_scale_to_kw),
        'cap_meta_kw': float(cap_meta_kw) if np.isfinite(cap_meta_kw) else np.nan,
        'cap_meta_field': cap_meta_field,
        'cap_meta_source': cap_meta_source,
        'ratio_p99_5_to_cap': float(ratio_p99_5_to_cap) if np.isfinite(ratio_p99_5_to_cap) else np.nan,
        'cap_status': cap_status,
        'cap_note': cap_note,
        'capacity_used_kw': float(capacity_used_kw) if np.isfinite(capacity_used_kw) else np.nan,
        'capacity_used_source': capacity_used_source,
    }


def _is_distributed_station_dir(station_dir):
    normalized = station_dir.replace('\\', '/')
    return normalized.startswith('2_distributed_non_rooftop/') or normalized.startswith('3_distributed_rooftop/')


def _parse_timestamp(value):
    value = str(value).strip()
    if not value:
        return None

    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    if value.endswith('Z'):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_timestamp_series(series):
    def _normalize_one(value):
        dt = _parse_timestamp(value)
        if dt is None:
            return None
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    return pd.to_datetime(series.map(_normalize_one), errors='coerce')


class Dataset_PV_MultiRes(Dataset):
    """Manifest-driven dataset for station-organized PV data."""

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None, target='OT', scale=True,
                 timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.default_freq = freq
        self.task_spec_name = getattr(args, 'task_spec_name', '') or ''
        self.task_spec = TASK_SPECS.get(self.task_spec_name)

        if self.task_spec is not None:
            self.seq_len, self.label_len, self.pred_len = self._resolve_task_lengths(self.task_spec)
        elif size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.manifest_path = os.path.abspath(args.manifest_path)
        self.station_data_root = os.path.abspath(args.station_data_root)
        self.allowed_granularities = {
            item.strip() for item in args.granularity_filter.split(',') if item.strip()
        }
        self.allowed_regions = {
            item.strip() for item in args.region_filter.split(',') if item.strip()
        }
        self.allowed_source_regions = {
            item.strip() for item in getattr(args, 'source_region_filter', '').split(',') if item.strip()
        }
        self.allowed_station_dirs = {
            item.strip() for item in args.station_dir_filter.split(',') if item.strip()
        }
        self.max_stations = max(0, int(args.max_stations))
        self.data_file_name = getattr(args, 'data_file_name', '') or ''
        self.time_col_override = getattr(args, 'time_col_override', '') or ''
        self.target_col_override = getattr(args, 'target_col_override', '') or ''
        self.feature_cols = [item.strip() for item in getattr(args, 'feature_cols', '').split(',') if item.strip()]
        self.nwp_mode = getattr(args, 'nwp_mode', 'none') or 'none'
        self.cross_unet_nwp_direct = self.nwp_mode == 'cross_unet_nwp_direct'
        self.pvtc_v2_direct = self.nwp_mode == 'pvtc_v2_direct'
        self.tide_direct = self.nwp_mode == 'tide_direct'
        self.tft_direct = self.nwp_mode == 'tft_direct'
        self.native_equal_context = getattr(args, 'model', '') in NATIVE_EQUAL_CONTEXT_MODELS
        if self.native_equal_context:
            native_equal_context_tasks = {'1h_short_6h', '1h_dayahead_24h', '1h_week_168h'}
            if self.task_spec_name and self.task_spec_name not in native_equal_context_tasks:
                raise ValueError(
                    "native equal-context baselines support only 1h tasks: "
                    "1h_short_6h, 1h_dayahead_24h, or 1h_week_168h"
                )
            self.seq_len = self.pred_len
            self.label_len = self.pred_len
        self.resample_to_1h = bool(getattr(args, 'resample_to_1h', False))
        self.hourly_resample_mode = (getattr(args, 'hourly_resample_mode', 'mean') or 'mean').strip().lower()
        self.hourly_resample_tolerance_minutes = float(
            getattr(args, 'hourly_resample_tolerance_minutes', 0.0) or 0.0
        )
        self.weather_file_name = getattr(args, 'weather_file_name', '') or ''
        self.weather_time_col = getattr(args, 'weather_time_col', '') or 'datetime'
        self.history_covariate_file_name = getattr(args, 'history_covariate_file_name', '') or ''
        self.history_covariate_time_col = getattr(args, 'history_covariate_time_col', '') or 'datetime'
        self.history_covariate_cols = [
            item.strip() for item in getattr(args, 'history_covariate_cols', '').split(',') if item.strip()
        ]
        self.future_covariate_file_name = getattr(args, 'future_covariate_file_name', '') or self.weather_file_name
        self.future_covariate_time_col = getattr(args, 'future_covariate_time_col', '') or self.weather_time_col
        self.future_covariate_cols = [
            item.strip() for item in getattr(args, 'future_covariate_cols', '').split(',') if item.strip()
        ]
        self.future_covariate_min_datetime_raw = (
            getattr(args, 'future_covariate_min_datetime', '') or ''
        ).strip()
        self.target_normalization = (getattr(args, 'target_normalization', 'none') or 'none').strip().lower()
        self.capacity_proxy_quantile = float(getattr(args, 'capacity_proxy_quantile', 99.5) or 99.5)
        self.future_covariate_min_datetime = None
        if self.future_covariate_min_datetime_raw:
            parsed_future_min_dt = _parse_timestamp(self.future_covariate_min_datetime_raw)
            if parsed_future_min_dt is None:
                raise ValueError(
                    f"Unsupported future_covariate_min_datetime: {self.future_covariate_min_datetime_raw}"
                )
            if parsed_future_min_dt.tzinfo is not None:
                parsed_future_min_dt = parsed_future_min_dt.astimezone(timezone.utc).replace(tzinfo=None)
            self.future_covariate_min_datetime = pd.Timestamp(parsed_future_min_dt)
        self.data_min_datetime_raw = (
            getattr(args, 'data_min_datetime', '') or ''
        ).strip()
        self.data_min_datetime = None
        if self.data_min_datetime_raw:
            parsed_data_min_dt = _parse_timestamp(self.data_min_datetime_raw)
            if parsed_data_min_dt is None:
                raise ValueError(f"Unsupported data_min_datetime: {self.data_min_datetime_raw}")
            if parsed_data_min_dt.tzinfo is not None:
                parsed_data_min_dt = parsed_data_min_dt.astimezone(timezone.utc).replace(tzinfo=None)
            self.data_min_datetime = pd.Timestamp(parsed_data_min_dt)
        self.future_covariate_align_to_hour = (
            getattr(args, 'future_covariate_align_to_hour', 'none') or 'none'
        ).strip().lower()
        self.sample_nan_ratio_threshold = float(getattr(args, 'sample_nan_ratio_threshold', 0.0) or 0.0)
        self.min_past_target_valid_ratio = float(getattr(args, 'min_past_target_valid_ratio', 1.0))
        self.min_future_target_valid_ratio = float(getattr(args, 'min_future_target_valid_ratio', 1.0))
        self.min_history_covariate_valid_ratio = float(getattr(args, 'min_history_covariate_valid_ratio', 1.0))
        self.min_future_covariate_valid_ratio = float(getattr(args, 'min_future_covariate_valid_ratio', 1.0))
        raw_eval_sample_stride = getattr(args, 'eval_sample_stride', 6)
        self.eval_sample_stride = int(6 if raw_eval_sample_stride is None else raw_eval_sample_stride)
        if self.eval_sample_stride < 1:
            raise ValueError('eval_sample_stride must be a positive integer')
        self.sample_index_csv = (getattr(args, 'sample_index_csv', '') or '').strip()
        # V6 full-shot trains on the baseline train/val splits. The frozen
        # evaluation whitelist must NEVER restrict (or empty) those splits.
        if self.nwp_mode == 'pvfm_v6_direct' and getattr(args, 'eval_origin_manifest', '') and self.flag != 'test':
            self.sample_index_csv = ''
        self.sample_index_filter = SampleIndexFilter(self.sample_index_csv, split=self.flag)
        self.audit_mode = (getattr(args, 'audit_mode', 'original') or 'original').strip().lower()
        self.data_qc_version = (
            getattr(args, 'data_qc_version', 'original-aligned-qc')
            or 'original-aligned-qc'
        ).strip()
        self.data_qc_root = (getattr(args, 'data_qc_root', '') or '').strip()
        self.eval_origin_manifest = (
            getattr(args, 'eval_origin_manifest', '') or ''
        ).strip()
        self.virtual_qc_release = None
        if self.audit_mode not in {'original', 'clean', 'paired'}:
            raise ValueError(
                f"Unsupported audit_mode={self.audit_mode!r}; expected original, clean, or paired"
            )
        if self.audit_mode in {'clean', 'paired'}:
            if not self.data_qc_root:
                raise ValueError(
                    f"audit_mode={self.audit_mode} requires --data_qc_root/--data-qc-root"
                )
            if not self.resample_to_1h:
                raise ValueError(
                    "virtual QC overlays require --resample_to_1h so boundary timestamps "
                    "remain on the canonical hourly grid"
                )
            self.virtual_qc_release = VirtualQCRelease(self.data_qc_root)
        self.strict_future_target_nan = bool(getattr(args, 'strict_future_target_nan', False))
        self.use_masked_future_loss = bool(getattr(args, 'use_masked_future_loss', False))
        raw_min_future_valid_ratio = getattr(args, 'min_future_valid_ratio', 1.0)
        self.min_future_valid_ratio = float(raw_min_future_valid_ratio) if raw_min_future_valid_ratio is not None else 1.0
        raw_seq_nan_interp_threshold = getattr(args, 'seq_nan_interp_threshold', -1.0)
        self.seq_nan_interp_threshold = float(raw_seq_nan_interp_threshold) if raw_seq_nan_interp_threshold is not None else -1.0
        self.seq_nan_fill_method = (getattr(args, 'seq_nan_fill_method', 'linear') or 'linear').strip().lower()
        self.enable_station_cache = bool(getattr(args, 'enable_station_cache', False))
        self.refresh_station_cache = bool(getattr(args, 'refresh_station_cache', False))
        if self.sample_index_filter.strict:
            # A cached payload was filtered before this process could record
            # exact manifest matches, so strict origin audits always rebuild.
            self.enable_station_cache = False
        self.station_cache_dir = os.path.abspath(
            getattr(args, 'station_cache_dir', './cache/pv_multires_station_cache')
        )
        self.enable_physical_sample_filter = bool(getattr(args, 'enable_physical_sample_filter', False))
        self.physical_filter_history_solar_col = (
            getattr(args, 'physical_filter_history_solar_col', 'shortwave_radiation') or 'shortwave_radiation'
        ).strip()
        self.physical_filter_future_solar_col = (
            getattr(args, 'physical_filter_future_solar_col', 'shortwave_radiation') or 'shortwave_radiation'
        ).strip()
        self.physical_filter_daylight_threshold = float(
            getattr(args, 'physical_filter_daylight_threshold', 100.0) or 100.0
        )
        self.physical_filter_target_zero_threshold = float(
            getattr(args, 'physical_filter_target_zero_threshold', 0.1) or 0.1
        )
        self.physical_filter_max_future_day_zero_ratio = float(
            getattr(args, 'physical_filter_max_future_day_zero_ratio', 1.0) or 1.0
        )
        self.physical_filter_max_history_day_zero_ratio = float(
            getattr(args, 'physical_filter_max_history_day_zero_ratio', 1.0) or 1.0
        )
        self.physical_filter_min_history_day_corr = float(
            getattr(args, 'physical_filter_min_history_day_corr', -2.0) or -2.0
        )
        self.physical_filter_min_future_daylight_points = int(
            getattr(args, 'physical_filter_min_future_daylight_points', 0) or 0
        )
        # PVTC stores source-manifest latitude/longitude in cached station data.
        self.station_cache_version = 9 if self.nwp_mode == 'pvfm_v6_direct' else 8
        self._validate_feature_configuration()
        if self.enable_station_cache:
            os.makedirs(self.station_cache_dir, exist_ok=True)

        self.samples = []
        self.station_offsets = []
        self.station_scalers = []
        self.station_records = []
        self._load_manifest()
        if self.nwp_mode == 'pvfm_v6_direct' and len(self.station_records) != 1:
            raise ValueError('PVFMV6FullShot requires exactly one nonempty station per run')

    def _validate_feature_configuration(self):
        if self.nwp_mode not in {
            'pvfm_v6_direct',
            'none',
            'feature_concat',
            'time_concat',
            'cross_unet_nwp_direct',
            'fusionsf_masked_nwp_direct',
            'pvtc_v2_direct',
            'tide_direct',
            'tft_direct',
            'dag_external_direct',
        }:
            raise ValueError(f"Unsupported nwp_mode: {self.nwp_mode}")
        if self.target_normalization not in {'none', 'capacity_factor'}:
            raise ValueError(f"Unsupported target_normalization: {self.target_normalization}")
        if self.future_covariate_align_to_hour not in {'none', 'floor', 'ceil', 'round'}:
            raise ValueError(f"Unsupported future_covariate_align_to_hour: {self.future_covariate_align_to_hour}")
        if self.seq_nan_fill_method not in {'linear'}:
            raise ValueError(f"Unsupported seq_nan_fill_method: {self.seq_nan_fill_method}")
        if self.hourly_resample_mode not in {'mean', 'exact', 'nearest'}:
            raise ValueError(f"Unsupported hourly_resample_mode: {self.hourly_resample_mode}")
        if self.hourly_resample_mode == 'nearest' and self.hourly_resample_tolerance_minutes <= 0.0:
            raise ValueError("hourly_resample_tolerance_minutes must be > 0 when hourly_resample_mode=nearest")
        if self.seq_nan_interp_threshold > 1.0:
            raise ValueError("seq_nan_interp_threshold must be <= 1.0")
        if not 0.0 <= self.min_future_valid_ratio <= 1.0:
            raise ValueError("min_future_valid_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.physical_filter_max_future_day_zero_ratio <= 1.0:
            raise ValueError("physical_filter_max_future_day_zero_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.physical_filter_max_history_day_zero_ratio <= 1.0:
            raise ValueError("physical_filter_max_history_day_zero_ratio must be between 0.0 and 1.0")
        if self.use_masked_future_loss and self.strict_future_target_nan:
            raise ValueError("use_masked_future_loss and strict_future_target_nan cannot both be enabled")

        if self.feature_cols and len(set(self.feature_cols)) != len(self.feature_cols):
            raise ValueError("feature_cols contains duplicates")
        if self.history_covariate_cols and len(set(self.history_covariate_cols)) != len(self.history_covariate_cols):
            raise ValueError("history_covariate_cols contains duplicates")
        if self.future_covariate_cols and len(set(self.future_covariate_cols)) != len(self.future_covariate_cols):
            raise ValueError("future_covariate_cols contains duplicates")

        target_col = self.target_col_override.strip()
        if target_col and self.feature_cols and target_col not in self.feature_cols:
            raise ValueError("feature_cols must include the target column when feature_cols is provided")

        if target_col and target_col in self.history_covariate_cols:
            raise ValueError("history_covariate_cols should not contain the target column")
        if target_col and target_col in self.future_covariate_cols:
            raise ValueError("future_covariate_cols should not contain the target column")

        if self.history_covariate_cols and self.features not in {'M', 'MS'}:
            raise ValueError("history_covariates require --features M or MS")
        active_model = getattr(self.args, 'model', '')
        if self.nwp_mode == 'pvfm_v6_direct':
            if active_model != 'PVFMV6FullShot' or self.features != 'MS':
                raise ValueError('pvfm_v6_direct requires PVFMV6FullShot and features=MS')
            if not self.history_covariate_cols or not self.future_covariate_cols:
                raise ValueError('v6 requires both historical and future weather')
        if self.nwp_mode == 'cross_unet_nwp_direct' and active_model != 'Cross_Unet':
            raise ValueError("nwp_mode=cross_unet_nwp_direct requires --model Cross_Unet")
        if self.nwp_mode == 'fusionsf_masked_nwp_direct' and active_model != 'FusionSFMasked':
            raise ValueError("nwp_mode=fusionsf_masked_nwp_direct requires --model FusionSFMasked")
        if self.nwp_mode == 'pvtc_v2_direct':
            if active_model not in {'PVTC_V2', 'PVTC_Ablation'}:
                raise ValueError("nwp_mode=pvtc_v2_direct requires --model PVTC_V2 or PVTC_Ablation")
            if not self.history_covariate_cols:
                raise ValueError("PVTC_V2 requires --history_covariate_cols")
            if not self.future_covariate_cols:
                raise ValueError("PVTC_V2 requires --future_covariate_cols")
        if self.nwp_mode == 'tide_direct':
            if active_model != 'TiDEOfficial':
                raise ValueError("nwp_mode=tide_direct requires --model TiDEOfficial")
            if not self.history_covariate_cols:
                raise ValueError("TiDEOfficial requires --history_covariate_cols")
            if not self.future_covariate_cols:
                raise ValueError("TiDEOfficial requires --future_covariate_cols")
        if self.nwp_mode == 'tft_direct':
            if active_model != 'TemporalFusionTransformerPV':
                raise ValueError("nwp_mode=tft_direct requires --model TemporalFusionTransformerPV")
            if not self.history_covariate_cols:
                raise ValueError("TemporalFusionTransformerPV requires --history_covariate_cols")
            if not self.future_covariate_cols:
                raise ValueError("TemporalFusionTransformerPV requires --future_covariate_cols")
        if self.nwp_mode == 'dag_external_direct':
            external_models = {'DAGWrapper', 'GCGNetWrapper', 'TimeXerWrapper', 'TiDEBenchmarkWrapper'}
            if active_model not in external_models:
                raise ValueError("nwp_mode=dag_external_direct requires one of the four DAG benchmark adapters")
            if not self.history_covariate_cols:
                raise ValueError(f"{active_model} requires --history_covariate_cols")
            if not self.future_covariate_cols:
                raise ValueError(f"{active_model} requires --future_covariate_cols")
        if self.future_covariate_cols and self.nwp_mode not in {
            'pvfm_v6_direct',
            'feature_concat',
            'time_concat',
            'cross_unet_nwp_direct',
            'fusionsf_masked_nwp_direct',
            'pvtc_v2_direct',
            'tide_direct',
            'tft_direct',
            'dag_external_direct',
        }:
            direct_future_cov_models = {
                'FusionSFBaseline', 'FusionSFNoSpatial', 'FusionSFMasked', 'UNetCFDirectNWP'
            }
            if active_model not in direct_future_cov_models:
                raise ValueError(
                    "future_covariates require a supported concat/direct NWP mode; "
                    "use --nwp_mode feature_concat, time_concat, or the active model's direct mode"
                )

    def _resolve_task_lengths(self, task_spec):
        granularity = task_spec['granularity']
        steps_per_hour = STEPS_PER_HOUR[granularity]
        horizon_hours = float(task_spec['horizon_hours'])
        context_hours = horizon_hours * float(task_spec['context_ratio'])

        pred_len = int(horizon_hours * steps_per_hour)
        seq_len = int(context_hours * steps_per_hour)
        label_len = max(1, pred_len)
        return seq_len, label_len, pred_len

    def _load_manifest(self):
        with open(self.manifest_path, 'r', encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))

        if self.task_spec is not None:
            should_skip_granularity_filter = self.resample_to_1h and self.task_spec.get('granularity') == '1h'
            if not should_skip_granularity_filter:
                rows = [row for row in rows if row.get('granularity', '') == self.task_spec['granularity']]

        if self.allowed_regions:
            rows = [row for row in rows if row.get('region', '') in self.allowed_regions]
        if self.allowed_source_regions:
            rows = [row for row in rows if row.get('source_region', '') in self.allowed_source_regions]
        if self.allowed_granularities:
            rows = [row for row in rows if row.get('granularity', '') in self.allowed_granularities]
        if self.allowed_station_dirs:
            rows = [row for row in rows if row.get('station_dir', '') in self.allowed_station_dirs]
        if self.max_stations:
            rows = rows[:self.max_stations]

        total_possible_samples = 0
        for row in rows:
            station_data = self._build_station_data(row)
            if station_data is None:
                continue

            self.station_offsets.append(len(self.samples))
            self.station_scalers.append(station_data['scaler'])
            self.station_records.append(row)
            for index in range(station_data['num_samples']):
                self.samples.append((len(self.station_records) - 1, index, station_data))
            total_possible_samples += station_data['num_possible_samples']
            if _is_primary_process():
                print(
                    f"[pv_multires] station={station_data['station_dir']} "
                    f"cache={station_data.get('cache_status', 'miss')} "
                    f"source={station_data['pv_source_file']} "
                    f"rows={station_data['num_rows']} "
                    f"samples={station_data['num_samples']}/{station_data['num_possible_samples']}"
                )

        self.sample_index_filter.assert_complete()

        if _is_primary_process():
            print(
                f"[pv_multires] dataset flag={self.flag} loaded_stations={len(self.station_records)} "
                f"valid_samples={len(self.samples)} possible_samples={total_possible_samples}"
            )

    def _infer_min_points_per_hour(self, timestamps, fallback_granularity):
        if len(timestamps) >= 2:
            ts = pd.Series(pd.to_datetime(timestamps)).sort_values().drop_duplicates()
            deltas = ts.diff().dropna().dt.total_seconds()
            if not deltas.empty:
                median_seconds = float(deltas.median())
                if median_seconds <= 90:
                    return 45, '1min'
                if median_seconds <= 600:
                    return 9, '5min'
                if median_seconds <= 1800:
                    return 3, '15min'
                return 1, '1h'
        return MIN_POINTS_PER_HOUR.get(fallback_granularity, 1), fallback_granularity

    def _prepare_hourly_power_frame(self, df, time_col, target_col, granularity):
        timestamps = _normalize_timestamp_series(df[time_col])
        target_values = pd.to_numeric(df[target_col], errors='coerce').clip(lower=0)
        valid_timestamp = timestamps.notna()
        if not (valid_timestamp & target_values.notna()).any():
            return None, granularity

        power_df = pd.DataFrame({
            'timestamp': timestamps[valid_timestamp],
            'target': target_values[valid_timestamp].astype(float),
        }).sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')
        if self.hourly_resample_mode == 'exact':
            aligned = power_df[
                power_df['timestamp'].dt.minute.eq(0)
                & power_df['timestamp'].dt.second.eq(0)
            ].copy()
            if aligned.empty:
                return None, granularity
            start = aligned['timestamp'].min().floor('1h')
            end = aligned['timestamp'].max().floor('1h')
            full_index = pd.date_range(start, end, freq='1h')
            hourly = (
                aligned.set_index('timestamp')[['target']]
                .reindex(full_index)
                .reset_index()
                .rename(columns={'index': 'timestamp'})
            )
            return hourly[['timestamp', 'target']], granularity
        if self.hourly_resample_mode == 'nearest':
            tolerance = pd.Timedelta(minutes=self.hourly_resample_tolerance_minutes)
            start = power_df['timestamp'].min().floor('1h')
            end = power_df['timestamp'].max().floor('1h')
            full_index = pd.date_range(start, end, freq='1h')
            base = pd.DataFrame({'timestamp': full_index})
            obs = power_df.rename(columns={'timestamp': 'obs_timestamp'}).sort_values('obs_timestamp')
            backward = pd.merge_asof(
                base,
                obs,
                left_on='timestamp',
                right_on='obs_timestamp',
                direction='backward',
                tolerance=tolerance,
            )
            forward = pd.merge_asof(
                base,
                obs,
                left_on='timestamp',
                right_on='obs_timestamp',
                direction='forward',
                tolerance=tolerance,
            )
            back_delta = (backward['timestamp'] - backward['obs_timestamp']).abs()
            forward_delta = (forward['obs_timestamp'] - forward['timestamp']).abs()
            choose_backward = forward['obs_timestamp'].isna() | (
                backward['obs_timestamp'].notna() & (back_delta <= forward_delta)
            )
            hourly = base.copy()
            hourly['target'] = np.where(choose_backward, backward['target'], forward['target'])
            return hourly[['timestamp', 'target']], granularity
        min_points, inferred_granularity = self._infer_min_points_per_hour(power_df['timestamp'], granularity)
        power_df = power_df.set_index('timestamp')
        hourly = power_df['target'].resample('1h').agg(['mean', 'count'])
        full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq='1h')
        hourly = hourly.reindex(full_index)
        hourly.index.name = 'timestamp'
        hourly = hourly.reset_index()
        hourly['target'] = hourly['mean'].where(hourly['count'] >= min_points)
        if hourly.empty:
            return None, inferred_granularity
        return hourly[['timestamp', 'target']], inferred_granularity

    def _load_covariate_frame(self, csv_path, file_name, time_col, requested_cols):
        if not file_name or not requested_cols:
            return None
        cov_path = os.path.join(os.path.dirname(csv_path), file_name)
        if not os.path.exists(cov_path):
            return None

        cov = pd.read_csv(cov_path)
        if time_col not in cov.columns:
            return None

        missing = [col for col in requested_cols if col not in cov.columns]
        if missing:
            return None

        cov = cov[[time_col] + requested_cols].copy()
        cov['timestamp'] = _normalize_timestamp_series(cov[time_col])
        cov = cov.drop(columns=[time_col])
        if file_name == self.future_covariate_file_name and self.future_covariate_min_datetime is not None:
            cov = cov[cov['timestamp'] >= self.future_covariate_min_datetime].copy()
        if file_name == self.future_covariate_file_name and self.future_covariate_align_to_hour != 'none':
            timestamp_series = pd.Series(cov['timestamp'])
            if self.future_covariate_align_to_hour == 'floor':
                cov['timestamp'] = timestamp_series.dt.floor('1h')
            elif self.future_covariate_align_to_hour == 'ceil':
                cov['timestamp'] = timestamp_series.dt.ceil('1h')
            elif self.future_covariate_align_to_hour == 'round':
                cov['timestamp'] = timestamp_series.dt.round('1h')
        for col in requested_cols:
            cov[col] = pd.to_numeric(cov[col], errors='coerce')
        cov = cov.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')
        return cov

    def _resolve_pv_csv_path(self, station_dir, pv_file):
        station_root = os.path.join(self.station_data_root, station_dir)
        if self.data_file_name:
            candidate = os.path.join(station_root, self.data_file_name)
            if os.path.exists(candidate):
                return candidate, self.data_file_name
        candidate = os.path.join(station_root, pv_file)
        if os.path.exists(candidate):
            return candidate, pv_file
        if _is_distributed_station_dir(station_dir):
            candidate = os.path.join(station_root, 'pv_regularized_nointerp.csv')
            if os.path.exists(candidate):
                return candidate, 'pv_regularized_nointerp.csv'
        return None, ''

    def _resolve_target_idx(self, data_array, target_idx):
        if target_idx is not None and target_idx >= 0:
            return int(target_idx)
        if data_array.ndim == 2 and data_array.shape[1] == 1:
            return 0
        return None

    def _interpolate_sample_window(self, window):
        if window.ndim != 2 or not np.any(~np.isfinite(window)):
            return window

        filled = (
            pd.DataFrame(window)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(method=self.seq_nan_fill_method, axis=0, limit_direction='both')
        )
        return filled.to_numpy(dtype=np.float32)

    def _apply_sample_nan_policy(self, seq_x, seq_y, target_idx):
        seq_x_work = seq_x.copy()
        seq_y_work = seq_y.copy()

        resolved_target_idx = self._resolve_target_idx(seq_y_work, target_idx)
        if resolved_target_idx is not None:
            future_target = seq_y_work[self.label_len:, resolved_target_idx]
            future_valid_ratio = float(np.count_nonzero(np.isfinite(future_target))) / float(len(future_target)) if len(future_target) else 0.0
            if self.use_masked_future_loss:
                if future_valid_ratio < self.min_future_valid_ratio:
                    return None
            elif self.strict_future_target_nan and not np.isfinite(future_target).all():
                return None

        if self.seq_nan_interp_threshold >= 0.0:
            total_count = seq_x_work.size
            if total_count == 0:
                return None

            seq_invalid_ratio = np.count_nonzero(~np.isfinite(seq_x_work)) / float(total_count)
            if seq_invalid_ratio > self.seq_nan_interp_threshold:
                return None

            if seq_invalid_ratio > 0.0:
                seq_x_work = self._interpolate_sample_window(seq_x_work)
                if self.label_len > 0:
                    # Keep decoder history aligned with the interpolated encoder-side history.
                    seq_y_work[:self.label_len] = self._interpolate_sample_window(seq_y_work[:self.label_len])

        return seq_x_work, seq_y_work

    def _build_invalid_mask(self, seq_x, seq_y, target_idx):
        seq_x_invalid = ~np.isfinite(seq_x)
        seq_y_invalid = ~np.isfinite(seq_y)

        if self.use_masked_future_loss:
            resolved_target_idx = self._resolve_target_idx(seq_y, target_idx)
            if resolved_target_idx is not None and self.pred_len > 0:
                future_invalid_target = seq_y_invalid[self.label_len:, resolved_target_idx].copy()
                seq_y_invalid[self.label_len:, resolved_target_idx] = False
                ignored_count = int(np.count_nonzero(future_invalid_target))
            else:
                ignored_count = 0
        else:
            ignored_count = 0

        return seq_x_invalid, seq_y_invalid, ignored_count

    def _passes_physical_sample_filter(self, s_begin, seq_x_raw_target, future_raw_target, history_raw_solar, future_raw_solar):
        if not self.enable_physical_sample_filter:
            return True

        daylight_threshold = self.physical_filter_daylight_threshold
        near_zero_threshold = self.physical_filter_target_zero_threshold

        if future_raw_solar is not None:
            future_solar_window = future_raw_solar[s_begin + self.seq_len:s_begin + self.seq_len + self.pred_len]
            future_target_window = future_raw_target[s_begin + self.seq_len:s_begin + self.seq_len + self.pred_len]
            future_day_mask = np.isfinite(future_solar_window) & np.isfinite(future_target_window) & (future_solar_window >= daylight_threshold)
            future_day_count = int(np.count_nonzero(future_day_mask))
            if future_day_count >= self.physical_filter_min_future_daylight_points:
                future_zero_ratio = float(np.mean(future_target_window[future_day_mask] <= near_zero_threshold))
                if future_zero_ratio > self.physical_filter_max_future_day_zero_ratio:
                    return False

        if history_raw_solar is not None:
            history_solar_window = history_raw_solar[s_begin:s_begin + self.seq_len]
            history_target_window = seq_x_raw_target[s_begin:s_begin + self.seq_len]
            history_day_mask = np.isfinite(history_solar_window) & np.isfinite(history_target_window) & (history_solar_window >= daylight_threshold)
            history_day_count = int(np.count_nonzero(history_day_mask))
            if history_day_count > 0:
                history_zero_ratio = float(np.mean(history_target_window[history_day_mask] <= near_zero_threshold))
                if history_zero_ratio > self.physical_filter_max_history_day_zero_ratio:
                    return False
            if self.physical_filter_min_history_day_corr > -1.0 and history_day_count >= 3:
                corr = np.corrcoef(history_target_window[history_day_mask], history_solar_window[history_day_mask])[0, 1]
                if np.isfinite(corr) and corr < self.physical_filter_min_history_day_corr:
                    return False

        return True

    def _compute_valid_sample_starts(
        self,
        data_x,
        data_y,
        target_idx,
        seq_x_raw_target=None,
        future_raw_target=None,
        history_raw_solar=None,
        future_raw_solar=None,
    ):
        future_context_len = self.seq_len if self.cross_unet_nwp_direct else self.pred_len
        max_start = len(data_x) - self.seq_len - future_context_len + 1
        if max_start <= 0:
            return [], 0

        valid_start_indices = []
        for s_begin in range(max_start):
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = data_x[s_begin:s_end]
            seq_y = data_y[r_begin:r_end]
            policy_applied = self._apply_sample_nan_policy(seq_x, seq_y, target_idx)
            if policy_applied is None:
                continue
            seq_x, seq_y = policy_applied

            seq_x_invalid, seq_y_invalid, ignored_count = self._build_invalid_mask(seq_x, seq_y, target_idx)
            total_count = seq_x.size + seq_y.size - ignored_count
            if total_count == 0:
                continue

            if self.cross_unet_nwp_direct:
                w_end = s_end + self.seq_len
                if w_end > len(data_y):
                    continue
                if data_y.ndim < 2 or data_y.shape[1] <= 1:
                    continue
                safe_target_idx = self._resolve_target_idx(data_y, target_idx)
                future_w = np.delete(data_y[s_end:w_end], safe_target_idx, axis=1)
                hist_w = np.delete(data_y[s_begin:s_end], safe_target_idx, axis=1)
                if future_w.size and not np.all(np.isfinite(future_w)):
                    continue
                if hist_w.size and not np.all(np.isfinite(hist_w)):
                    continue
                target_x_idx = self._resolve_target_idx(data_x, target_idx)
                if s_begin < self.seq_len:
                    hist_x = data_x[s_begin:s_end, target_x_idx:target_x_idx + 1]
                else:
                    hist_x = data_x[s_begin - self.seq_len:s_begin, target_x_idx:target_x_idx + 1]
                if hist_x.shape[0] != self.seq_len or not np.all(np.isfinite(hist_x)):
                    continue

            if not self._passes_physical_sample_filter(
                s_begin,
                seq_x_raw_target,
                future_raw_target,
                history_raw_solar,
                future_raw_solar,
            ):
                continue

            invalid_ratio = (np.count_nonzero(seq_x_invalid) + np.count_nonzero(seq_y_invalid)) / float(total_count)
            if invalid_ratio <= self.sample_nan_ratio_threshold:
                valid_start_indices.append(s_begin)

        return valid_start_indices, max_start

    def _resolve_encoder_feature_cols(self, target_col):
        requested_feature_cols = self.feature_cols or [target_col]
        non_target_cols = [col for col in requested_feature_cols if col != target_col]
        if self.history_covariate_cols:
            missing = [col for col in non_target_cols if col not in self.history_covariate_cols]
            if missing:
                raise ValueError(
                    "feature_cols contains columns that are not provided by history_covariate_cols: "
                    + ', '.join(missing)
                )
        return requested_feature_cols

    def _get_covariate_path(self, csv_path, file_name):
        if not file_name:
            return None
        cov_path = os.path.join(os.path.dirname(csv_path), file_name)
        if not os.path.exists(cov_path):
            return None
        return cov_path

    def _build_file_signature(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return None
        stat = os.stat(file_path)
        return {
            'path': os.path.abspath(file_path),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    def _build_station_cache_key(self, row, csv_path, target_col, requested_feature_cols, history_cov_path, future_cov_path):
        payload = {
            'cache_version': self.station_cache_version,
            'flag': self.flag,
            'set_type': self.set_type,
            'station_dir': row.get('station_dir', ''),
            'station_id': row.get('station_id', ''),
            'granularity': row.get('granularity', ''),
            'time_col_override': self.time_col_override,
            'target_col': target_col,
            'requested_feature_cols': requested_feature_cols,
            'task_spec_name': self.task_spec_name,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'scale': bool(self.scale),
            'timeenc': int(self.timeenc),
            'default_freq': self.default_freq,
            'features': self.features,
            'nwp_mode': self.nwp_mode,
            'resample_to_1h': bool(self.resample_to_1h),
            'hourly_resample_mode': self.hourly_resample_mode,
            'hourly_resample_tolerance_minutes': self.hourly_resample_tolerance_minutes,
            'history_covariate_file_name': self.history_covariate_file_name,
            'history_covariate_time_col': self.history_covariate_time_col,
            'history_covariate_cols': self.history_covariate_cols,
            'future_covariate_file_name': self.future_covariate_file_name,
            'future_covariate_time_col': self.future_covariate_time_col,
            'future_covariate_cols': self.future_covariate_cols,
            'future_covariate_min_datetime': self.future_covariate_min_datetime_raw,
            'target_normalization': self.target_normalization,
            'capacity_proxy_quantile': self.capacity_proxy_quantile,
            'target_standard_scaler_enabled': bool(self.scale and self.target_normalization == 'none'),
            'data_min_datetime': self.data_min_datetime_raw,
            'future_covariate_align_to_hour': self.future_covariate_align_to_hour,
            'sample_nan_ratio_threshold': self.sample_nan_ratio_threshold,
            'min_past_target_valid_ratio': self.min_past_target_valid_ratio,
            'min_future_target_valid_ratio': self.min_future_target_valid_ratio,
            'min_history_covariate_valid_ratio': self.min_history_covariate_valid_ratio,
            'min_future_covariate_valid_ratio': self.min_future_covariate_valid_ratio,
            'eval_sample_stride': self.eval_sample_stride,
            'sample_index_csv': self.sample_index_csv,
            'sample_index_file': self._build_file_signature(self.sample_index_csv),
            'audit_mode': self.audit_mode,
            'data_qc_version': self.data_qc_version,
            'data_qc_root': self.data_qc_root,
            'virtual_qc_release_signature': (
                self.virtual_qc_release.release_signature
                if self.virtual_qc_release is not None else None
            ),
            'eval_origin_manifest': self.eval_origin_manifest,
            'strict_future_target_nan': bool(self.strict_future_target_nan),
            'use_masked_future_loss': bool(self.use_masked_future_loss),
            'min_future_valid_ratio': self.min_future_valid_ratio,
            'seq_nan_interp_threshold': self.seq_nan_interp_threshold,
            'seq_nan_fill_method': self.seq_nan_fill_method,
            'enable_physical_sample_filter': bool(self.enable_physical_sample_filter),
            'physical_filter_history_solar_col': self.physical_filter_history_solar_col,
            'physical_filter_future_solar_col': self.physical_filter_future_solar_col,
            'physical_filter_daylight_threshold': self.physical_filter_daylight_threshold,
            'physical_filter_target_zero_threshold': self.physical_filter_target_zero_threshold,
            'physical_filter_max_future_day_zero_ratio': self.physical_filter_max_future_day_zero_ratio,
            'physical_filter_max_history_day_zero_ratio': self.physical_filter_max_history_day_zero_ratio,
            'physical_filter_min_history_day_corr': self.physical_filter_min_history_day_corr,
            'physical_filter_min_future_daylight_points': self.physical_filter_min_future_daylight_points,
            'pv_file': self._build_file_signature(csv_path),
            'history_cov_file': self._build_file_signature(history_cov_path),
            'future_cov_file': self._build_file_signature(future_cov_path),
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _get_station_cache_path(self, station_dir, cache_key):
        safe_station_dir = station_dir.replace('\\', '__').replace('/', '__')
        return os.path.join(self.station_cache_dir, f"{safe_station_dir}__{self.flag}__{cache_key}.pkl")

    def _serialize_scaler(self, scaler):
        return {
            'mean_': np.asarray(scaler.mean_, dtype=np.float64),
            'scale_': np.asarray(scaler.scale_, dtype=np.float64),
            'var_': np.asarray(scaler.var_, dtype=np.float64),
            'n_features_in_': int(scaler.n_features_in_),
        }

    def _deserialize_scaler(self, payload):
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload['mean_'], dtype=np.float64)
        scaler.scale_ = np.asarray(payload['scale_'], dtype=np.float64)
        scaler.var_ = np.asarray(payload['var_'], dtype=np.float64)
        scaler.n_features_in_ = int(payload['n_features_in_'])
        return scaler

    def _save_station_cache(self, cache_path, station_data):
        payload = {
            'cache_version': self.station_cache_version,
            'station_data': {
                'data_x': np.asarray(station_data['data_x'], dtype=np.float32),
                'data_y': np.asarray(station_data['data_y'], dtype=np.float32),
                'data_stamp': np.asarray(station_data['data_stamp'], dtype=np.float32),
                'timestamps': np.asarray(station_data.get('timestamps', []), dtype=object),
                'target_step_minutes': float(station_data.get('target_step_minutes', 60.0) or 60.0),
                'scaler': self._serialize_scaler(station_data['scaler']),
                'num_samples': int(station_data['num_samples']),
                'num_possible_samples': int(station_data['num_possible_samples']),
                'eval_sample_stride': int(station_data.get('eval_sample_stride', self.eval_sample_stride)),
                'valid_start_indices': np.asarray(station_data['valid_start_indices'], dtype=np.int32),
                'num_rows': int(station_data['num_rows']),
                'target_idx': int(station_data['target_idx']),
                'target_to_power_scale': float(station_data.get('target_to_power_scale', 1.0) or 1.0),
                'target_normalization_mode': station_data.get('target_normalization_mode', 'none'),
                'capacity_used_kw': float(station_data.get('capacity_used_kw', np.nan)),
                'capacity_used_source': station_data.get('capacity_used_source', ''),
                'target_standard_scaler_enabled': bool(station_data.get('target_standard_scaler_enabled', True)),
                'lat': float(station_data.get('lat', np.nan)),
                'lon': float(station_data.get('lon', np.nan)),
                'station_dir': station_data['station_dir'],
                'station_id': station_data['station_id'],
                'station_name': station_data['station_name'],
                'region': station_data['region'],
                'source_region': station_data['source_region'],
                'effective_granularity': station_data['effective_granularity'],
                'pv_source_file': station_data['pv_source_file'],
                'audit_mode': station_data.get('audit_mode', self.audit_mode),
                'data_qc_version': station_data.get('data_qc_version', self.data_qc_version),
                'virtual_qc_cropped_rows': int(
                    station_data.get('virtual_qc_cropped_rows', 0) or 0
                ),
                'virtual_qc_masked_rows': int(
                    station_data.get('virtual_qc_masked_rows', 0) or 0
                ),
                'virtual_qc_identity_transform': bool(
                    station_data.get('virtual_qc_identity_transform', True)
                ),
            },
        }
        if self.nwp_mode == 'pvfm_v6_direct':
            for key in ('v6_site_features', 'v6_time_features', 'v6_timezone_name'):
                payload['station_data'][key] = station_data[key]
        # Multiple DDP ranks can build the same immutable station cache at
        # startup. Write-then-replace prevents another rank from reading a
        # partially written pickle; the final payload is deterministic.
        temp_path = f"{cache_path}.tmp.{os.getpid()}"
        try:
            with open(temp_path, 'wb') as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, cache_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load_station_cache(self, cache_path):
        try:
            with open(cache_path, 'rb') as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError, ValueError, TypeError):
            return None

        if not isinstance(payload, dict) or payload.get('cache_version') != self.station_cache_version:
            return None

        station_payload = payload.get('station_data')
        if not isinstance(station_payload, dict):
            return None

        station_data = {
            'data_x': np.asarray(station_payload['data_x'], dtype=np.float32),
            'data_y': np.asarray(station_payload['data_y'], dtype=np.float32),
            'data_stamp': np.asarray(station_payload['data_stamp'], dtype=np.float32),
            'timestamps': np.asarray(station_payload.get('timestamps', []), dtype=object),
            'target_step_minutes': float(station_payload.get('target_step_minutes', 60.0) or 60.0),
            'scaler': self._deserialize_scaler(station_payload['scaler']),
            'num_samples': int(station_payload['num_samples']),
            'num_possible_samples': int(station_payload['num_possible_samples']),
            'eval_sample_stride': int(station_payload.get('eval_sample_stride', self.eval_sample_stride)),
            'valid_start_indices': np.asarray(station_payload['valid_start_indices'], dtype=np.int32).tolist(),
            'num_rows': int(station_payload['num_rows']),
            'target_idx': int(station_payload['target_idx']),
            'target_to_power_scale': float(station_payload.get('target_to_power_scale', 1.0) or 1.0),
            'target_normalization_mode': station_payload.get('target_normalization_mode', 'none'),
            'capacity_used_kw': float(station_payload.get('capacity_used_kw', np.nan)),
            'capacity_used_source': station_payload.get('capacity_used_source', ''),
            'target_standard_scaler_enabled': bool(station_payload.get('target_standard_scaler_enabled', True)),
            'lat': float(station_payload.get('lat', np.nan)),
            'lon': float(station_payload.get('lon', np.nan)),
            'station_dir': station_payload['station_dir'],
            'station_id': station_payload['station_id'],
            'station_name': station_payload['station_name'],
            'region': station_payload['region'],
            'source_region': station_payload['source_region'],
            'effective_granularity': station_payload['effective_granularity'],
            'pv_source_file': station_payload['pv_source_file'],
            'audit_mode': station_payload.get('audit_mode', self.audit_mode),
            'data_qc_version': station_payload.get('data_qc_version', self.data_qc_version),
            'virtual_qc_cropped_rows': int(
                station_payload.get('virtual_qc_cropped_rows', 0) or 0
            ),
            'virtual_qc_masked_rows': int(
                station_payload.get('virtual_qc_masked_rows', 0) or 0
            ),
            'virtual_qc_identity_transform': bool(
                station_payload.get('virtual_qc_identity_transform', True)
            ),
            'cache_status': 'hit',
        }
        if self.nwp_mode == 'pvfm_v6_direct':
            for key in ('v6_site_features', 'v6_time_features', 'v6_timezone_name'):
                if key not in station_payload:
                    return None
                station_data[key] = station_payload[key]
        return station_data

    def _build_station_data(self, row):
        station_dir = row['station_dir'].replace('\\', os.sep).replace('/', os.sep)
        pv_file = row.get('pv_file', 'pv.csv')
        granularity = row.get('granularity', '')
        time_col = self.time_col_override or row.get('time_col', '')
        target_col = self.target_col_override or row.get('target_col', '')

        if not time_col or not target_col:
            return None

        csv_path, resolved_pv_file = self._resolve_pv_csv_path(station_dir, pv_file)
        if not csv_path:
            return None

        requested_feature_cols = self._resolve_encoder_feature_cols(target_col)
        history_cov_path = self._get_covariate_path(csv_path, self.history_covariate_file_name)
        future_cov_path = self._get_covariate_path(csv_path, self.future_covariate_file_name)
        cache_path = None
        if self.enable_station_cache:
            cache_key = self._build_station_cache_key(
                row=row,
                csv_path=csv_path,
                target_col=target_col,
                requested_feature_cols=requested_feature_cols,
                history_cov_path=history_cov_path,
                future_cov_path=future_cov_path,
            )
            cache_path = self._get_station_cache_path(station_dir, cache_key)
            if not self.refresh_station_cache and os.path.exists(cache_path):
                cached_station_data = self._load_station_cache(cache_path)
                if cached_station_data is not None:
                    return cached_station_data

        df = pd.read_csv(csv_path)
        if time_col not in df.columns or target_col not in df.columns:
            return None
        other_data = _load_other_data_json_for_station(csv_path)

        model_df = None
        effective_granularity = granularity
        resolved_future_covariate_cols = list(self.future_covariate_cols)

        if self.resample_to_1h:
            hourly_power, inferred_granularity = self._prepare_hourly_power_frame(df, time_col, target_col, granularity)
            if hourly_power is None:
                return None
            model_df = hourly_power.copy()
            model_df[target_col] = model_df['target']
            effective_granularity = '1h'

            if self.history_covariate_cols:
                history_df = self._load_covariate_frame(
                    csv_path,
                    self.history_covariate_file_name,
                    self.history_covariate_time_col,
                    self.history_covariate_cols,
                )
                if history_df is None:
                    return None
                model_df = model_df.merge(history_df, on='timestamp', how='left')
            if self.future_covariate_cols:
                future_df = self._load_covariate_frame(
                    csv_path,
                    self.future_covariate_file_name,
                    self.future_covariate_time_col,
                    self.future_covariate_cols,
                )
                if future_df is None:
                    return None
                overlap_cols = [col for col in self.future_covariate_cols if col in model_df.columns]
                if overlap_cols:
                    renamed_future_cols = {col: f"{col}__future" for col in overlap_cols}
                    future_df = future_df.rename(columns=renamed_future_cols)
                    resolved_future_covariate_cols = [
                        renamed_future_cols.get(col, col) for col in self.future_covariate_cols
                    ]
                model_df = model_df.merge(future_df, on='timestamp', how='left')
            granularity = inferred_granularity
        else:
            timestamps = _normalize_timestamp_series(df[time_col])
            values = pd.to_numeric(df[target_col], errors='coerce')

            model_df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps),
                'target': values.astype(float),
            })

            if self.history_covariate_cols:
                history_df = self._load_covariate_frame(
                    csv_path,
                    self.history_covariate_file_name,
                    self.history_covariate_time_col,
                    self.history_covariate_cols,
                )
                if history_df is None:
                    return None
                model_df = model_df.merge(history_df, on='timestamp', how='inner')
                if model_df.empty:
                    return None

            model_df[target_col] = model_df['target']

            if any(col not in model_df.columns for col in requested_feature_cols):
                return None
            if self.future_covariate_cols:
                future_df = self._load_covariate_frame(
                    csv_path,
                    self.future_covariate_file_name,
                    self.future_covariate_time_col,
                    self.future_covariate_cols,
                )
                if future_df is None:
                    return None
                overlap_cols = [col for col in self.future_covariate_cols if col in requested_feature_cols]
                if overlap_cols:
                    renamed_future_cols = {col: f"{col}__future" for col in overlap_cols}
                    future_df = future_df.rename(columns=renamed_future_cols)
                    resolved_future_covariate_cols = [
                        renamed_future_cols.get(col, col) for col in self.future_covariate_cols
                    ]
                model_df = model_df.merge(future_df, on='timestamp', how='inner')
                if model_df.empty:
                    return None
            model_df = model_df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')

            feature_df = model_df[requested_feature_cols].apply(pd.to_numeric, errors='coerce')
            valid = model_df['timestamp'].notna() & model_df['target'].notna() & feature_df.notna().all(axis=1)
            if not valid.any():
                return None

            model_df = model_df.loc[valid].copy()

        virtual_qc_stats = {
            'cropped_rows': 0,
            'masked_rows': 0,
            'identity_transform': True,
        }
        if self.virtual_qc_release is not None:
            target_columns = list(dict.fromkeys(('target', target_col)))
            model_df, virtual_qc_stats = self.virtual_qc_release.apply(
                model_df,
                station_dir,
                target_columns=target_columns,
            )
            if model_df.empty:
                return None

        if self.data_min_datetime is not None:
            model_df = model_df[model_df['timestamp'] >= self.data_min_datetime].copy()

        if len(model_df) <= self.seq_len + self.pred_len:
            return None

        num_train = int(len(model_df) * 0.7)
        num_test = int(len(model_df) * 0.2)
        num_val = len(model_df) - num_train - num_test

        border1s = [0, max(0, num_train - self.seq_len), max(0, len(model_df) - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_val, len(model_df)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.sample_index_filter.enabled:
            border1 = 0
            border2 = len(model_df)

        capacity_info = {
            'target_normalization_mode': 'none',
            'power_semantics': '',
            'power_semantics_note': '',
            'power_unit_scale_to_kw': 1.0,
            'cap_meta_kw': np.nan,
            'cap_meta_field': '',
            'cap_meta_source': '',
            'ratio_p99_5_to_cap': np.nan,
            'cap_status': '',
            'cap_note': '',
            'capacity_used_kw': np.nan,
            'capacity_used_source': '',
            'target_to_power_scale': 1.0,
        }
        scaler = StandardScaler()
        train_target_full = model_df.iloc[border1s[0]:border2s[0]][['target']].dropna()
        if self.target_normalization == 'capacity_factor':
            if train_target_full.empty:
                return None
            capacity_info = _resolve_capacity_normalization(
                train_target_full['target'].to_numpy(dtype=np.float32),
                other_data,
                self.capacity_proxy_quantile,
            )
            scale_to_kw = float(capacity_info.get('power_unit_scale_to_kw', 1.0) or 1.0)
            model_df['target'] = pd.to_numeric(model_df['target'], errors='coerce') * scale_to_kw
            capacity_used_kw = capacity_info.get('capacity_used_kw', np.nan)
            if capacity_info.get('power_semantics') == 'already_normalized':
                capacity_info['target_normalization_mode'] = 'already_normalized'
                capacity_info['target_to_power_scale'] = (
                    float(capacity_used_kw) if np.isfinite(capacity_used_kw) and float(capacity_used_kw) > 0 else 1.0
                )
            elif np.isfinite(capacity_used_kw) and float(capacity_used_kw) > 0:
                model_df['target'] = model_df['target'] / float(capacity_used_kw)
                capacity_info['target_normalization_mode'] = 'capacity_factor'
                capacity_info['target_to_power_scale'] = float(capacity_used_kw)
            else:
                capacity_info['target_normalization_mode'] = 'power_kw_fallback'
                capacity_info['target_to_power_scale'] = 1.0

        series_df = model_df.iloc[border1:border2].copy()
        if len(series_df) <= self.seq_len + self.pred_len:
            return None

        target_standard_scaler_enabled = bool(self.scale and self.target_normalization == 'none')
        if target_standard_scaler_enabled:
            train_target = model_df.iloc[border1s[0]:border2s[0]][['target']].dropna()
            if train_target.empty:
                return None
            scaler.fit(train_target.values)
        else:
            scaler.mean_ = np.zeros(1, dtype=np.float64)
            scaler.scale_ = np.ones(1, dtype=np.float64)
            scaler.var_ = np.ones(1, dtype=np.float64)
            scaler.n_features_in_ = 1

        raw_target_series = series_df['target'].to_numpy(dtype=np.float32)
        history_raw_solar = None
        future_raw_solar = None
        if self.history_covariate_cols and self.physical_filter_history_solar_col in series_df.columns:
            history_raw_solar = pd.to_numeric(
                series_df[self.physical_filter_history_solar_col], errors='coerce'
            ).to_numpy(dtype=np.float32)
        target_scaled = scaler.transform(series_df[['target']].values).astype(np.float32)
        target_idx = None
        ordered_encoder_cols = [col for col in requested_feature_cols if col != target_col] + [target_col]
        encoder_extra_cols = [col for col in ordered_encoder_cols if col != target_col]

        if encoder_extra_cols:
            extra_scaler = StandardScaler()
            train_extra = model_df.iloc[border1s[0]:border2s[0]][encoder_extra_cols].dropna()
            if self.scale:
                if train_extra.empty:
                    return None
                extra_scaler.fit(train_extra)
                encoder_extra_values = extra_scaler.transform(series_df[encoder_extra_cols].values).astype(np.float32)
            else:
                encoder_extra_values = series_df[encoder_extra_cols].values.astype(np.float32)
            data_x = np.concatenate([encoder_extra_values, target_scaled], axis=1).astype(np.float32)
            target_idx = len(ordered_encoder_cols) - 1
        else:
            data_x = target_scaled

        if self.future_covariate_cols:
            ordered_decoder_cols = [col for col in resolved_future_covariate_cols if col != target_col] + [target_col]
            decoder_extra_cols = [col for col in ordered_decoder_cols if col != target_col]
            future_solar_col = self.physical_filter_future_solar_col
            if future_solar_col not in series_df.columns and f"{future_solar_col}__future" in series_df.columns:
                future_solar_col = f"{future_solar_col}__future"
            if future_solar_col in series_df.columns:
                future_raw_solar = pd.to_numeric(
                    series_df[future_solar_col], errors='coerce'
                ).to_numpy(dtype=np.float32)
            decoder_parts = []
            if decoder_extra_cols:
                future_scaler = StandardScaler()
                train_future = model_df.iloc[border1s[0]:border2s[0]][decoder_extra_cols].dropna()
                if self.scale:
                    if train_future.empty:
                        return None
                    future_scaler.fit(train_future)
                    future_values = future_scaler.transform(series_df[decoder_extra_cols].values).astype(np.float32)
                else:
                    future_values = series_df[decoder_extra_cols].values.astype(np.float32)
                decoder_parts.append(future_values)
            decoder_parts.append(target_scaled)
            data_y = np.concatenate(decoder_parts, axis=1).astype(np.float32)
            target_idx = len(ordered_decoder_cols) - 1
        else:
            data_y = data_x

        freq = GRANULARITY_TO_FREQ.get(effective_granularity, self.default_freq)
        stamp_df = pd.DataFrame({'date': series_df['timestamp'].values})
        if self.timeenc == 0:
            stamp_df['date'] = pd.to_datetime(stamp_df['date'])
            stamp_df['month'] = stamp_df.date.apply(lambda x: x.month)
            stamp_df['day'] = stamp_df.date.apply(lambda x: x.day)
            stamp_df['weekday'] = stamp_df.date.apply(lambda x: x.weekday())
            stamp_df['hour'] = stamp_df.date.apply(lambda x: x.hour)
            if freq == 't':
                stamp_df['minute'] = stamp_df.date.apply(lambda x: x.minute // 15)
            elif freq == 's':
                stamp_df['minute'] = stamp_df.date.apply(lambda x: x.minute)
                stamp_df['second'] = stamp_df.date.apply(lambda x: x.second)
            data_stamp = stamp_df.drop(columns=['date']).values
        else:
            data_stamp = time_features(pd.to_datetime(stamp_df['date'].values), freq=freq).transpose(1, 0)

        if self.nwp_mode == 'time_concat' and not self.future_covariate_cols:
            extra_cols = [col for col in requested_feature_cols if col != target_col]
            if extra_cols:
                extra_values = series_df[extra_cols].values.astype(np.float32)
                data_stamp = np.concatenate([data_stamp.astype(np.float32), extra_values], axis=1)

        history_target_idx = data_x.shape[1] - 1 if data_x.ndim == 2 and data_x.shape[1] > 0 else 0
        future_target_idx = data_y.shape[1] - 1 if data_y.ndim == 2 and data_y.shape[1] > 0 else 0
        history_covariate_mask = (
            np.isfinite(np.delete(data_x, history_target_idx, axis=1))
            if data_x.ndim == 2 and data_x.shape[1] > 1
            else np.ones((len(data_x), 0), dtype=bool)
        )
        future_covariate_mask = (
            np.isfinite(np.delete(data_y, future_target_idx, axis=1))
            if data_y.ndim == 2 and data_y.shape[1] > 1
            else np.ones((len(data_y), 0), dtype=bool)
        )
        valid_start_indices, num_possible_samples, _dropped_reasons = compute_valid_window_starts(
            target_mask=np.isfinite(raw_target_series),
            target_values=raw_target_series,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            resolution=effective_granularity,
            sample_nan_ratio_threshold=self.sample_nan_ratio_threshold,
            history_covariate_mask=history_covariate_mask,
            future_covariate_mask=future_covariate_mask,
            decoder_covariate_context_len=self.label_len,
            min_past_target_valid_ratio=self.min_past_target_valid_ratio,
            min_future_target_valid_ratio=self.min_future_target_valid_ratio,
            min_history_covariate_valid_ratio=self.min_history_covariate_valid_ratio,
            min_future_covariate_valid_ratio=self.min_future_covariate_valid_ratio,
            return_reasons=True,
        )
        valid_start_indices = filter_valid_starts_by_sample_index(
            valid_start_indices,
            pd.to_datetime(series_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(dtype=object),
            self.seq_len,
            self.pred_len,
            station_dir,
            row.get('station_id', ''),
            self.sample_index_filter,
        )
        if self.cross_unet_nwp_direct:
            cross_unet_valid_starts = []
            safe_future_target_idx = self._resolve_target_idx(data_y, target_idx)
            safe_history_target_idx = self._resolve_target_idx(data_x, target_idx)
            for s_begin in valid_start_indices:
                s_end = s_begin + self.seq_len
                w_end = s_end + self.seq_len
                if w_end > len(data_y):
                    continue
                if safe_future_target_idx is None or safe_history_target_idx is None:
                    continue
                future_w = np.delete(data_y[s_end:w_end], safe_future_target_idx, axis=1)
                hist_w = np.delete(data_y[s_begin:s_end], safe_future_target_idx, axis=1)
                if future_w.size and not np.all(np.isfinite(future_w)):
                    continue
                if hist_w.size and not np.all(np.isfinite(hist_w)):
                    continue
                if s_begin < self.seq_len:
                    hist_x = data_x[s_begin:s_end, safe_history_target_idx:safe_history_target_idx + 1]
                else:
                    hist_x = data_x[s_begin - self.seq_len:s_begin, safe_history_target_idx:safe_history_target_idx + 1]
                if hist_x.shape[0] != self.seq_len or not np.all(np.isfinite(hist_x)):
                    continue
                cross_unet_valid_starts.append(s_begin)
            valid_start_indices = cross_unet_valid_starts
        if (
            self.flag == 'test'
            and self.eval_sample_stride > 1
            and not self.sample_index_filter.strict
        ):
            # A strict origin manifest is already an exact whitelist. It is
            # authoritative and must not be thinned a second time.
            valid_start_indices = valid_start_indices[::self.eval_sample_stride]
        if not valid_start_indices:
            return None

        station_data = {
            'data_x': data_x.astype(np.float32),
            'data_y': data_y.astype(np.float32),
            'data_stamp': data_stamp.astype(np.float32),
            'timestamps': pd.to_datetime(series_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(dtype=object),
            'scaler': scaler,
            'num_samples': len(valid_start_indices),
            'num_possible_samples': num_possible_samples,
            'eval_sample_stride': self.eval_sample_stride,
            'valid_start_indices': valid_start_indices,
            'num_rows': len(series_df),
            'target_idx': -1 if target_idx is None else int(target_idx),
            'target_to_power_scale': float(capacity_info.get('target_to_power_scale', 1.0) or 1.0),
            'target_normalization_mode': capacity_info.get('target_normalization_mode', 'none'),
            'capacity_used_kw': float(capacity_info.get('capacity_used_kw', np.nan)),
            'capacity_used_source': capacity_info.get('capacity_used_source', ''),
            'target_standard_scaler_enabled': bool(target_standard_scaler_enabled),
            'lat': float(_maybe_float(row.get('lat')) if _maybe_float(row.get('lat')) is not None else np.nan),
            'lon': float(_maybe_float(row.get('lon')) if _maybe_float(row.get('lon')) is not None else np.nan),
            'station_dir': station_dir,
            'station_id': row.get('station_id', ''),
            'station_name': row.get('station_name', ''),
            'region': row.get('region', ''),
            'source_region': row.get('source_region', ''),
            'effective_granularity': effective_granularity,
            'pv_source_file': resolved_pv_file,
            'audit_mode': self.audit_mode,
            'data_qc_version': self.data_qc_version,
            'virtual_qc_cropped_rows': int(virtual_qc_stats.get('cropped_rows', 0) or 0),
            'virtual_qc_masked_rows': int(virtual_qc_stats.get('masked_rows', 0) or 0),
            'virtual_qc_identity_transform': bool(
                virtual_qc_stats.get('identity_transform', True)
            ),
            'cache_status': 'miss',
        }
        if self.nwp_mode == 'pvfm_v6_direct':
            from foundation.data.datasets import _infer_timezone_offset_hours
            from foundation.data.transforms import build_time_features
            from foundation.data.multires_binary_indexed_chronos import align_multires_time_features
            offset, name = _infer_timezone_offset_hours(row, other_data, pd.to_datetime(df[time_col]).iloc[0])
            if not np.isfinite(offset) or not np.isfinite(station_data['lat']) or not np.isfinite(station_data['lon']):
                raise ValueError(f'v6 requires explicit timezone and lat/lon metadata: {station_dir}')
            station_data['v6_site_features'] = np.asarray([
                station_data['lat'], station_data['lon'], 1.0,
                station_data['capacity_used_kw'], offset], dtype=np.float32)
            if not np.isfinite(station_data['v6_site_features']).all():
                raise ValueError(f'v6 requires finite station capacity: {station_dir}')
            station_data['v6_timezone_name'] = name
            station_data['v6_time_features'] = align_multires_time_features(
                build_time_features(station_data['timestamps'], '1h'), '1h', 5,
                schema='minute_hour_weekday_day_dayofyear_v1')
        if len(station_data['timestamps']) >= 2:
            delta = pd.to_datetime(station_data['timestamps'][1]) - pd.to_datetime(station_data['timestamps'][0])
            station_data['target_step_minutes'] = float(delta.total_seconds() / 60.0)
        else:
            station_data['target_step_minutes'] = 60.0
        if self.enable_station_cache and cache_path:
            self._save_station_cache(cache_path, station_data)
        return station_data

    def __getitem__(self, index):
        station_idx, sample_idx, station_data = self.samples[index]

        s_begin = station_data['valid_start_indices'][sample_idx]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = station_data['data_x'][s_begin:s_end]
        seq_y = station_data['data_y'][r_begin:r_end]
        policy_applied = self._apply_sample_nan_policy(seq_x, seq_y, station_data['target_idx'])
        if policy_applied is not None:
            seq_x, seq_y = policy_applied
        seq_x_mark = station_data['data_stamp'][s_begin:s_end]
        seq_y_mark = station_data['data_stamp'][r_begin:r_end]
        timestamps = station_data.get('timestamps')
        input_end_time = ''
        future_start_time = ''
        if timestamps is not None and len(timestamps) > 0:
            if 0 <= s_end - 1 < len(timestamps):
                input_end_time = str(timestamps[s_end - 1])
            if 0 <= s_end < len(timestamps):
                future_start_time = str(timestamps[s_end])

        metadata = {
            'station_idx': station_idx,
            'sample_index': int(sample_idx),
            'station_dir': station_data['station_dir'],
            'station_id': station_data['station_id'],
            'station_name': station_data['station_name'],
            'region': station_data['region'],
            'source_region': station_data['source_region'],
            'input_end_time': input_end_time,
            'future_start_time': future_start_time,
            'target_step_minutes': float(station_data.get('target_step_minutes', 60.0) or 60.0),
            'target_idx': station_data['target_idx'],
            'target_to_power_scale': float(station_data.get('target_to_power_scale', 1.0) or 1.0),
            'target_normalization_mode': station_data.get('target_normalization_mode', 'none'),
            'capacity_used_kw': float(station_data.get('capacity_used_kw', np.nan)),
            'capacity_used_source': station_data.get('capacity_used_source', ''),
            'target_standard_scaler_enabled': bool(station_data.get('target_standard_scaler_enabled', True)),
        }

        if self.cross_unet_nwp_direct:
            w_end = s_end + self.seq_len
            target_idx = station_data['target_idx']
            safe_target_idx = self._resolve_target_idx(station_data['data_y'], target_idx)
            batch_w = np.delete(station_data['data_y'][s_end:w_end], safe_target_idx, axis=1).astype(np.float32)
            seq_w_nwp_hist = np.delete(station_data['data_y'][s_begin:s_end], safe_target_idx, axis=1).astype(np.float32)
            target_x_idx = self._resolve_target_idx(station_data['data_x'], target_idx)
            if s_begin < self.seq_len:
                seq_x_hist = station_data['data_x'][s_begin:s_end, target_x_idx:target_x_idx + 1]
            else:
                seq_x_hist = station_data['data_x'][s_begin - self.seq_len:s_begin, target_x_idx:target_x_idx + 1]
            seq_x_hist = seq_x_hist.astype(np.float32)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, metadata, batch_w, seq_w_nwp_hist, seq_x_hist

        if self.pvtc_v2_direct:
            if timestamps is None or s_end + self.pred_len > len(timestamps):
                raise ValueError("PVTC_V2 requires absolute timestamps through the forecast horizon")
            timestamp_slice = pd.to_datetime(timestamps[s_begin:s_end + self.pred_len], errors='coerce')
            if timestamp_slice.isna().any():
                raise ValueError("PVTC_V2 received invalid timestamps")
            epoch_hours = (timestamp_slice.astype('int64') // 3_600_000_000_000).to_numpy(dtype=np.int64)
            lat = float(station_data.get('lat', np.nan))
            lon = float(station_data.get('lon', np.nan))
            if not np.isfinite(lat) or not np.isfinite(lon):
                raise ValueError(f"PVTC_V2 requires finite manifest lat/lon for {station_data['station_dir']}")
            capacity_kw = float(station_data.get('capacity_used_kw', np.nan))
            capacity_feature = 0.0 if not np.isfinite(capacity_kw) else np.clip(np.log1p(max(0.0, capacity_kw)) / np.log1p(1_000_000.0), 0.0, 1.0)
            history_specs = [get_feature_spec(name) for name in self.history_covariate_cols]
            future_specs = [get_feature_spec(name.replace('__future', '')) for name in self.future_covariate_cols]
            history_feature_mask = np.isfinite(seq_x).all(axis=0).astype(np.float32)
            future_nwp = seq_y[-self.pred_len:, :-1]
            future_nwp_mask = np.isfinite(future_nwp).all(axis=0).astype(np.float32)
            pvtc_extras = {
                'pvtc_static_feat': np.asarray([(lat + 90.0) / 180.0, (lon + 180.0) / 360.0, 1.0, capacity_feature], dtype=np.float32),
                'pvtc_timestamps': epoch_hours,
                'pvtc_feature_mask': history_feature_mask,
                'pvtc_nwp_mask': future_nwp_mask,
                'pvtc_hist_feature_ids': np.asarray([spec['feature_id'] for spec in history_specs], dtype=np.int64),
                'pvtc_hist_group_ids': np.asarray([spec['group_id'] for spec in history_specs], dtype=np.int64),
                'pvtc_nwp_feature_ids': np.asarray([spec['feature_id'] for spec in future_specs], dtype=np.int64),
                'pvtc_nwp_group_ids': np.asarray([spec['group_id'] for spec in future_specs], dtype=np.int64),
            }
            return seq_x, seq_y, seq_x_mark, seq_y_mark, metadata, pvtc_extras

        if self.nwp_mode == 'pvfm_v6_direct':
            if station_data.get('target_step_minutes') != 60.0:
                raise ValueError('v6 full-shot adapter requires hourly data')
            extra = {
                'past_time_features': station_data['v6_time_features'][s_begin:s_end],
                'future_time_features': station_data['v6_time_features'][s_end:s_end + self.pred_len],
                'site_features': station_data['v6_site_features'],
                # Match the current dynamic V6 collator's [C,H,H,horizon_hours],
                # not the unrelated baseline decoder label_len.
                'static_features': np.asarray([self.seq_len, self.pred_len, self.pred_len, self.pred_len], dtype=np.float32),
            }
            return seq_x, seq_y, seq_x_mark, seq_y_mark, metadata, extra

        if self.tide_direct:
            if timestamps is None or s_end + self.pred_len > len(timestamps):
                raise ValueError("TiDEOfficial requires timestamps through the forecast horizon")
            tide_times = _tide_time_covariates(timestamps[s_begin:s_end + self.pred_len])
            tide_extras = {
                'tide_past_time_features': tide_times[:, :self.seq_len],
                'tide_future_time_features': tide_times[:, self.seq_len:],
            }
            return seq_x, seq_y, seq_x_mark, seq_y_mark, metadata, tide_extras

        return seq_x, seq_y, seq_x_mark, seq_y_mark, metadata

    def __len__(self):
        return len(self.samples)

    def inverse_transform(self, data):
        if not self.station_scalers:
            raise RuntimeError('No station scalers were loaded for pv_multires dataset.')
        data = np.asarray(data)
        if data.ndim == 3:
            flat = data.reshape(-1, data.shape[-1])
            restored = self.station_scalers[0].inverse_transform(flat[:, -1:].reshape(-1, 1))
            return restored.reshape(data.shape[0], data.shape[1], 1)
        return self.station_scalers[0].inverse_transform(data)
