import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from foundation.data.datasets import (
    PVTaskDataset,
    _slice_native_covariate_window,
    _soft_mask_covariate_block,
)


LEGACY_TIME_FEATURE_SCHEMA = "legacy_right_pad_v1"
CANONICAL_TIME_FEATURE_SCHEMA = "minute_hour_weekday_day_dayofyear_v1"


def _normalize_time_feature_schema(value):
    """Normalize the versioned calendar-channel aliases used by all loaders."""
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return None
    if text in {"legacy", LEGACY_TIME_FEATURE_SCHEMA}:
        return LEGACY_TIME_FEATURE_SCHEMA
    if text in {"canonical", CANONICAL_TIME_FEATURE_SCHEMA}:
        return CANONICAL_TIME_FEATURE_SCHEMA
    raise ValueError(f"unsupported legacy-pipeline time_feature_schema={value!r}.")


def _align_chronos_time_features(values, resolution, target_dim, schema):
    """Align one legacy-pipeline row to the requested calendar contract."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"time features must be rank-2, got shape={array.shape}.")
    target_dim = int(target_dim)
    if target_dim <= 0 or array.shape[-1] > target_dim:
        raise ValueError(
            f"cannot align time features shape={array.shape} to target_dim={target_dim}."
        )
    schema = _normalize_time_feature_schema(schema) or LEGACY_TIME_FEATURE_SCHEMA
    resolution = str(resolution or "1h").strip().lower()
    if schema == CANONICAL_TIME_FEATURE_SCHEMA:
        if target_dim != 5:
            raise ValueError(
                f"{CANONICAL_TIME_FEATURE_SCHEMA} requires target_dim=5, got {target_dim}."
            )
        if resolution == "1h" and array.shape[-1] == 4:
            aligned = np.empty((array.shape[0], 5), dtype=np.float32)
            aligned[:, 0] = -0.5
            aligned[:, 1:] = array
            return aligned
        if array.shape[-1] != 5:
            raise ValueError(
                f"{CANONICAL_TIME_FEATURE_SCHEMA} expects 1h/4D or sub-hourly/5D rows, "
                f"got resolution={resolution!r} shape={array.shape}."
            )
        return array
    if array.shape[-1] == target_dim:
        return array
    padded = np.zeros((array.shape[0], target_dim), dtype=np.float32)
    padded[:, : array.shape[-1]] = array
    return padded


def _pad_last_dim(values, target_dim):
    values = values.astype(np.float32)
    if values.shape[-1] == target_dim:
        return values
    padded = np.zeros((*values.shape[:-1], target_dim), dtype=np.float32)
    padded[..., : values.shape[-1]] = values
    return padded


def build_task_datasets(
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
            strict_hourly_resample=strict_hourly_resample,
            hourly_resample_mode=hourly_resample_mode,
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
            dataset_cache_dir=dataset_cache_dir,
            use_dataset_cache=use_dataset_cache,
        )
        if len(dataset) > 0:
            datasets.append(dataset)
    if not datasets:
        raise ValueError("No non-empty datasets were built for the requested task list.")
    return datasets


class ChronosLikePVBatchDataset(IterableDataset):
    """A minimal Chronos-style iterable dataset that directly yields full batches."""

    def __init__(
        self,
        task_datasets,
        batch_size,
        split="train",
        shuffle=True,
        drop_last=False,
        sample_nan_ratio_threshold=0.0,
        fixed_seq_len=0,
        distributed_rank=0,
        distributed_world_size=1,
        train_sampler_mode="default",
        region_balance_alpha=0.5,
        region_balance_max_prob=0.0,
        region_balance_max_repeat_per_epoch=0.0,
        eval_sample_stride=6,
        time_feature_schema=None,
    ):
        super().__init__()
        self.task_datasets = list(task_datasets)
        self.batch_size = int(batch_size)
        self.split = split
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sample_nan_ratio_threshold = float(sample_nan_ratio_threshold)
        self.fixed_seq_len = int(fixed_seq_len or 0)
        self.distributed_rank = int(distributed_rank or 0)
        self.distributed_world_size = max(1, int(distributed_world_size or 1))
        self.train_sampler_mode = str(train_sampler_mode or "default").strip().lower()
        self.region_balance_alpha = max(0.0, float(region_balance_alpha or 0.0))
        self.region_balance_max_prob = max(0.0, float(region_balance_max_prob or 0.0))
        self.region_balance_max_repeat_per_epoch = max(
            0.0, float(region_balance_max_repeat_per_epoch or 0.0)
        )
        self.eval_sample_stride = int(6 if eval_sample_stride is None else eval_sample_stride)
        if self.eval_sample_stride < 1:
            raise ValueError("eval_sample_stride must be a positive integer")
        # A strict sample-index manifest is an exact origin whitelist.  The
        # evaluator must not thin it a second time with the numeric stride.
        self.has_strict_sample_index = any(
            bool(
                getattr(dataset, "sample_index_filter", None)
                and dataset.sample_index_filter.strict
            )
            for dataset in self.task_datasets
        )
        # ``None`` is intentionally the historical right-pad contract.  The
        # parser's explicit ``auto`` sentinel may opt into resolution-aware
        # canonical alignment, while old direct callers that omit this
        # argument retain byte-compatible 1h/mixed behavior.
        schema_text = str(time_feature_schema or "").strip().lower()
        requested_schema = _normalize_time_feature_schema(time_feature_schema)
        if time_feature_schema is None or not schema_text:
            requested_schema = LEGACY_TIME_FEATURE_SCHEMA
        elif requested_schema is None:
            resolutions = {
                str(getattr(getattr(dataset, "task_spec", None), "resolution", "1h") or "1h")
                .strip()
                .lower()
                for dataset in self.task_datasets
            }
            requested_schema = (
                CANONICAL_TIME_FEATURE_SCHEMA
                if any(resolution != "1h" for resolution in resolutions)
                else LEGACY_TIME_FEATURE_SCHEMA
            )
        self.time_feature_schema = requested_schema
        self.task_summaries = []
        self.model_dims = self._infer_global_dims()
        if self.time_feature_schema == CANONICAL_TIME_FEATURE_SCHEMA:
            self.model_dims["past_time_dim"] = max(5, int(self.model_dims["past_time_dim"]))
            self.model_dims["future_time_dim"] = max(5, int(self.model_dims["future_time_dim"]))
            self.model_dims["context_input_dim"] = (
                int(self.model_dims["target_dim"])
                + int(self.model_dims["historical_covariate_dim"])
                + int(self.model_dims["past_time_dim"])
            )
            self.model_dims["future_input_dim"] = (
                int(self.model_dims["future_covariate_dim"])
                + int(self.model_dims["future_time_dim"])
            )
        self._region_sampling_state = self._build_region_sampling_state()

    def _owned_batch(self, batch_id):
        return self.distributed_world_size <= 1 or (batch_id % self.distributed_world_size) == self.distributed_rank

    def _sample_history_length(self, dataset):
        max_seq_len = int(dataset.seq_len)
        if self.fixed_seq_len > 0:
            return min(max_seq_len, self.fixed_seq_len)
        candidate_hours = tuple(getattr(dataset.task_spec, "history_choices_hours", ()) or ())
        if not candidate_hours or self.split != "train":
            return max_seq_len
        max_hours = int(getattr(dataset.task_spec, "max_context_hours", 0) or dataset.task_spec.context_hours)
        allowed = []
        for hours in candidate_hours:
            hours = int(hours)
            if hours <= 0 or hours > max_hours:
                continue
            steps = hours * int(dataset.task_adapter.prediction_length / dataset.task_spec.horizon_hours)
            allowed.append(min(max_seq_len, int(steps)))
        if not allowed:
            return max_seq_len
        return random.choice(sorted(set(allowed)))

    def _infer_global_dims(self):
        target_dim = 0
        hist_cov_dim = 0
        future_cov_dim = 0
        past_time_dim = 0
        future_time_dim = 0
        static_dim = 0
        summaries = []
        for dataset in self.task_datasets:
            sample = dataset[0]
            target_dim = max(target_dim, int(sample.chronos_core.past_target.shape[-1]))
            hist_cov_dim = max(hist_cov_dim, int(sample.chronos_core.historical_covariates.shape[-1]))
            future_cov_dim = max(future_cov_dim, int(sample.chronos_core.future_covariates.shape[-1]))
            past_time_dim = max(past_time_dim, int(sample.chronos_core.past_time_features.shape[-1]))
            future_time_dim = max(future_time_dim, int(sample.chronos_core.future_time_features.shape[-1]))
            static_dim = max(static_dim, int(sample.chronos_core.static_features.shape[-1]))
            summaries.append(
                {
                    "task_name": dataset.task_name,
                    "resolution": dataset.task_spec.resolution,
                    "seq_len": dataset.seq_len,
                    "label_len": dataset.label_len,
                    "pred_len": dataset.pred_len,
                    "max_context_hours": dataset.task_spec.context_hours,
                    "history_choices_hours": (self.fixed_seq_len,) if self.fixed_seq_len > 0 else tuple(getattr(dataset.task_spec, "history_choices_hours", ()) or ()),
                    "num_samples": len(dataset),
                    "num_stations": len(dataset.station_records),
                }
            )
        self.task_summaries = summaries
        return {
            "target_dim": target_dim,
            "historical_covariate_dim": hist_cov_dim,
            "future_covariate_dim": future_cov_dim,
            "past_time_dim": past_time_dim,
            "future_time_dim": future_time_dim,
            "static_dim": static_dim,
            "context_input_dim": target_dim + hist_cov_dim + past_time_dim,
            "future_input_dim": future_cov_dim + future_time_dim,
            "output_dim": target_dim,
        }

    def _batch_indices(self, dataset):
        indices = list(range(len(dataset)))
        if (
            self.split != "train"
            and self.eval_sample_stride > 1
            and not self.has_strict_sample_index
        ):
            indices = indices[::self.eval_sample_stride]
        if self.distributed_world_size > 1:
            indices = indices[self.distributed_rank::self.distributed_world_size]
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            indices = indices[worker_info.id::worker_info.num_workers]
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            yield batch_indices

    def _extract_region_key(self, station_record):
        station_dir = str((station_record or {}).get("station_dir", "") or "")
        parts = station_dir.split("/")
        if len(parts) >= 3 and parts[2]:
            return parts[2]
        region = str((station_record or {}).get("region", "") or "").strip()
        if region:
            return region
        return "UNKNOWN"

    def _build_region_sampling_state(self):
        state = []
        if self.train_sampler_mode != "region_balanced":
            return state
        for dataset in self.task_datasets:
            buckets = defaultdict(list)
            for sample_index, (station_index, _sample_start) in enumerate(dataset.index):
                if self.distributed_world_size > 1 and (sample_index % self.distributed_world_size) != self.distributed_rank:
                    continue
                station_record = dataset.station_records[station_index]
                region_key = self._extract_region_key(station_record)
                buckets[region_key].append(sample_index)
            if not buckets:
                state.append(None)
                continue
            region_keys = sorted(buckets.keys())
            region_weights = []
            for region_key in region_keys:
                sample_count = len(buckets[region_key])
                weight = float(sample_count) ** (-self.region_balance_alpha) if sample_count > 0 else 0.0
                region_weights.append(weight)
            total_weight = sum(region_weights)
            if total_weight <= 0:
                state.append(None)
                continue
            region_probs = [weight / total_weight for weight in region_weights]
            state.append(
                {
                    "region_keys": region_keys,
                    "region_probs": region_probs,
                    "region_to_indices": {key: buckets[key] for key in region_keys},
                }
            )
        return state

    def _sample_region_balanced_batch_indices(self, dataset_idx):
        state = self._region_sampling_state[dataset_idx]
        if state is None:
            return []
        region_key = random.choices(state["region_keys"], weights=state["region_probs"], k=1)[0]
        candidates = state["region_to_indices"][region_key]
        if not candidates:
            return []
        if len(candidates) >= self.batch_size:
            return random.sample(candidates, self.batch_size)
        return random.choices(candidates, k=self.batch_size)

    def _sample_is_valid(self, sample):
        past_mask = sample.chronos_core.past_observed_mask
        future_mask = sample.chronos_core.future_observed_mask
        if past_mask.size == 0 or future_mask.size == 0:
            return False
        past_valid = float(np.sum(past_mask))
        future_valid = float(np.sum(future_mask))
        if past_valid <= 0 or future_valid <= 0:
            return False
        total_count = float(past_mask.size + future_mask.size)
        invalid_ratio = 1.0 - ((past_valid + future_valid) / total_count)
        if invalid_ratio > self.sample_nan_ratio_threshold:
            return False
        return True

    def _build_batch(self, samples, history_length=None):
        batch_size = len(samples)
        seq_len = int(history_length or samples[0].seq_len)
        pred_len = samples[0].pred_len
        dims = self.model_dims

        past_target = np.zeros((batch_size, seq_len, dims["target_dim"]), dtype=np.float32)
        past_observed_mask = np.zeros((batch_size, seq_len, dims["target_dim"]), dtype=np.float32)
        historical_covariates = np.zeros((batch_size, seq_len, dims["historical_covariate_dim"]), dtype=np.float32)
        historical_covariates_mask = np.zeros((batch_size, seq_len, dims["historical_covariate_dim"]), dtype=np.float32)
        future_covariates = np.zeros((batch_size, pred_len, dims["future_covariate_dim"]), dtype=np.float32)
        future_covariates_mask = np.zeros((batch_size, pred_len, dims["future_covariate_dim"]), dtype=np.float32)
        future_target = np.zeros((batch_size, pred_len, dims["output_dim"]), dtype=np.float32)
        future_target_mask = np.zeros((batch_size, pred_len, dims["output_dim"]), dtype=np.float32)
        past_time_features = np.zeros((batch_size, seq_len, dims["past_time_dim"]), dtype=np.float32)
        future_time_features = np.zeros((batch_size, pred_len, dims["future_time_dim"]), dtype=np.float32)
        static_features = np.zeros((batch_size, dims["static_dim"]), dtype=np.float32)
        site_features = np.zeros((batch_size, 5), dtype=np.float32)
        native_hist_len = max((sample.chronos_core.historical_covariates_native.shape[0] for sample in samples), default=0)
        native_fut_len = max((sample.chronos_core.future_covariates_native.shape[0] for sample in samples), default=0)
        native_hist_time_dim = max((sample.chronos_core.historical_covariates_native_time_features.shape[-1] if sample.chronos_core.historical_covariates_native_time_features.ndim == 2 else 0 for sample in samples), default=0)
        native_fut_time_dim = max((sample.chronos_core.future_covariates_native_time_features.shape[-1] if sample.chronos_core.future_covariates_native_time_features.ndim == 2 else 0 for sample in samples), default=0)
        historical_covariates_native = np.zeros((batch_size, native_hist_len, dims["historical_covariate_dim"]), dtype=np.float32)
        historical_covariates_native_mask = np.zeros((batch_size, native_hist_len, dims["historical_covariate_dim"]), dtype=np.float32)
        future_covariates_native = np.zeros((batch_size, native_fut_len, dims["future_covariate_dim"]), dtype=np.float32)
        future_covariates_native_mask = np.zeros((batch_size, native_fut_len, dims["future_covariate_dim"]), dtype=np.float32)
        historical_covariates_native_time_features = np.zeros((batch_size, native_hist_len, native_hist_time_dim), dtype=np.float32)
        future_covariates_native_time_features = np.zeros((batch_size, native_fut_len, native_fut_time_dim), dtype=np.float32)

        task_names = []
        resolutions = []
        station_ids = []
        seq_lens = []
        pred_lens = []
        metadata = []

        for idx, sample in enumerate(samples):
            past_target[idx] = _pad_last_dim(sample.chronos_core.past_target[-seq_len:], dims["target_dim"])
            past_observed_mask[idx] = _pad_last_dim(sample.chronos_core.past_observed_mask[-seq_len:], dims["target_dim"])
            future_target[idx] = _pad_last_dim(sample.chronos_core.future_target, dims["output_dim"])
            future_target_mask[idx] = _pad_last_dim(sample.chronos_core.future_observed_mask, dims["output_dim"])

            if dims["historical_covariate_dim"] > 0:
                historical_covariates[idx] = _pad_last_dim(
                    sample.chronos_core.historical_covariates[-seq_len:],
                    dims["historical_covariate_dim"],
                )
                historical_covariates_mask[idx] = _pad_last_dim(
                    sample.chronos_core.historical_covariates_mask[-seq_len:],
                    dims["historical_covariate_dim"],
                )
            if dims["future_covariate_dim"] > 0:
                future_covariates[idx] = _pad_last_dim(
                    sample.chronos_core.future_covariates,
                    dims["future_covariate_dim"],
                )
                future_covariates_mask[idx] = _pad_last_dim(
                    sample.chronos_core.future_covariates_mask,
                    dims["future_covariate_dim"],
                )
            if dims["past_time_dim"] > 0:
                past_time_features[idx] = _pad_last_dim(
                    _align_chronos_time_features(
                        sample.chronos_core.past_time_features[-seq_len:],
                        resolution=sample.resolution,
                        target_dim=dims["past_time_dim"],
                        schema=self.time_feature_schema,
                    ),
                    dims["past_time_dim"],
                )
            if dims["future_time_dim"] > 0:
                future_time_features[idx] = _pad_last_dim(
                    _align_chronos_time_features(
                        sample.chronos_core.future_time_features,
                        resolution=sample.resolution,
                        target_dim=dims["future_time_dim"],
                        schema=self.time_feature_schema,
                    ),
                    dims["future_time_dim"],
                )
            if dims["historical_covariate_dim"] > 0 and native_hist_len > 0:
                hist_native_len = sample.chronos_core.historical_covariates_native.shape[0]
                if hist_native_len > 0:
                    historical_covariates_native[idx, :hist_native_len] = _pad_last_dim(
                        sample.chronos_core.historical_covariates_native,
                        dims["historical_covariate_dim"],
                    )
                    historical_covariates_native_mask[idx, :hist_native_len] = _pad_last_dim(
                        sample.chronos_core.historical_covariates_native_mask,
                        dims["historical_covariate_dim"],
                    )
                    if native_hist_time_dim > 0:
                        historical_covariates_native_time_features[idx, :hist_native_len] = _pad_last_dim(
                            sample.chronos_core.historical_covariates_native_time_features,
                            native_hist_time_dim,
                        )
            if dims["future_covariate_dim"] > 0 and native_fut_len > 0:
                fut_native_len = sample.chronos_core.future_covariates_native.shape[0]
                if fut_native_len > 0:
                    future_covariates_native[idx, :fut_native_len] = _pad_last_dim(
                        sample.chronos_core.future_covariates_native,
                        dims["future_covariate_dim"],
                    )
                    future_covariates_native_mask[idx, :fut_native_len] = _pad_last_dim(
                        sample.chronos_core.future_covariates_native_mask,
                        dims["future_covariate_dim"],
                    )
                    if native_fut_time_dim > 0:
                        future_covariates_native_time_features[idx, :fut_native_len] = _pad_last_dim(
                            sample.chronos_core.future_covariates_native_time_features,
                            native_fut_time_dim,
                        )
            static_value = sample.chronos_core.static_features.copy()
            if static_value.shape[-1] > 0:
                static_value[0] = float(seq_len)
            static_features[idx, : static_value.shape[-1]] = static_value

            row = sample.metadata.get("station_record", {}) or {}
            lat = row.get("lat", np.nan)
            lon = row.get("lon", np.nan)
            try:
                lat = float(lat)
            except Exception:
                lat = np.nan
            try:
                lon = float(lon)
            except Exception:
                lon = np.nan
            cap_kw = sample.metadata.get("capacity_used_kw", np.nan)
            try:
                cap_kw = float(cap_kw)
            except Exception:
                cap_kw = np.nan
            timezone_offset_hours = sample.metadata.get("timezone_offset_hours", np.nan)
            try:
                timezone_offset_hours = float(timezone_offset_hours)
            except Exception:
                timezone_offset_hours = np.nan
            reliability = 0.0 if (not np.isfinite(lat) or not np.isfinite(lon)) else 1.0
            site_features[idx] = np.array(
                [
                    0.0 if not np.isfinite(lat) else lat,
                    0.0 if not np.isfinite(lon) else lon,
                    reliability,
                    0.0 if not np.isfinite(cap_kw) else cap_kw,
                    0.0 if not np.isfinite(timezone_offset_hours) else timezone_offset_hours,
                ],
                dtype=np.float32,
            )

            task_names.append(sample.task_name)
            resolutions.append(sample.resolution)
            station_ids.append(sample.station_id)
            seq_lens.append(seq_len)
            pred_lens.append(sample.pred_len)
            sample_metadata = dict(sample.metadata)
            sample_metadata["sampled_history_length"] = seq_len
            metadata.append(sample_metadata)

        context_padding_mask = (past_observed_mask.sum(axis=-1) > 0).astype(np.float32)

        return {
            "past_target": torch.from_numpy(past_target),
            "past_observed_mask": torch.from_numpy(past_observed_mask),
            "historical_covariates": torch.from_numpy(historical_covariates),
            "historical_covariates_mask": torch.from_numpy(historical_covariates_mask),
            "future_covariates": torch.from_numpy(future_covariates),
            "future_covariates_mask": torch.from_numpy(future_covariates_mask),
            "historical_covariates_native": torch.from_numpy(historical_covariates_native),
            "historical_covariates_native_mask": torch.from_numpy(historical_covariates_native_mask),
            "future_covariates_native": torch.from_numpy(future_covariates_native),
            "future_covariates_native_mask": torch.from_numpy(future_covariates_native_mask),
            "future_target": torch.from_numpy(future_target),
            "future_target_mask": torch.from_numpy(future_target_mask),
            "past_time_features": torch.from_numpy(past_time_features),
            "future_time_features": torch.from_numpy(future_time_features),
            "historical_covariates_native_time_features": torch.from_numpy(historical_covariates_native_time_features),
            "future_covariates_native_time_features": torch.from_numpy(future_covariates_native_time_features),
            "static_features": torch.from_numpy(static_features),
            "site_features": torch.from_numpy(site_features),
            "context_padding_mask": torch.from_numpy(context_padding_mask),
            "task_names": task_names,
            "resolutions": resolutions,
            "station_ids": station_ids,
            "seq_lens": seq_lens,
            "pred_lens": pred_lens,
            "metadata": metadata,
        }

    def __iter__(self):
        dataset_order = list(range(len(self.task_datasets)))
        batch_id = 0
        if self.split == "train":
            if self.train_sampler_mode == "region_balanced":
                while True:
                    if self.shuffle:
                        random.shuffle(dataset_order)
                    for dataset_idx in dataset_order:
                        dataset = self.task_datasets[dataset_idx]
                        batch_indices = self._sample_region_balanced_batch_indices(dataset_idx)
                        if not batch_indices:
                            continue
                        samples = [dataset[index] for index in batch_indices]
                        samples = [sample for sample in samples if self._sample_is_valid(sample)]
                        if not samples:
                            continue
                        history_length = self._sample_history_length(dataset)
                        yield self._build_batch(samples, history_length=history_length)
                        batch_id += 1
            else:
                while True:
                    active = {}
                    if self.shuffle:
                        random.shuffle(dataset_order)
                    for dataset_idx in dataset_order:
                        dataset = self.task_datasets[dataset_idx]
                        active[dataset_idx] = iter(self._batch_indices(dataset))
                    while active:
                        round_order = list(active.keys())
                        if self.shuffle:
                            random.shuffle(round_order)
                        for dataset_idx in round_order:
                            dataset = self.task_datasets[dataset_idx]
                            try:
                                batch_indices = next(active[dataset_idx])
                            except StopIteration:
                                del active[dataset_idx]
                                continue
                            samples = [dataset[index] for index in batch_indices]
                            samples = [sample for sample in samples if self._sample_is_valid(sample)]
                            if not samples:
                                continue
                            history_length = self._sample_history_length(dataset)
                            yield self._build_batch(samples, history_length=history_length)
                            batch_id += 1
        else:
            if self.shuffle:
                random.shuffle(dataset_order)
            for dataset_idx in dataset_order:
                dataset = self.task_datasets[dataset_idx]
                for batch_indices in self._batch_indices(dataset):
                    samples = [dataset[index] for index in batch_indices]
                    samples = [sample for sample in samples if self._sample_is_valid(sample)]
                    if not samples:
                        continue
                    yield self._build_batch(samples, history_length=dataset.seq_len)
                    batch_id += 1

    def __len__(self):
        total = 0
        for dataset in self.task_datasets:
            dataset_len = len(dataset)
            if self.distributed_world_size > 1:
                dataset_len = (dataset_len + self.distributed_world_size - 1 - self.distributed_rank) // self.distributed_world_size
            if self.drop_last:
                total += dataset_len // self.batch_size
            else:
                total += math.ceil(dataset_len / self.batch_size)
        return total


class UnifiedPVForecastBatchDataset(IterableDataset):
    """Unified 1h PV forecasting dataset with online horizon/seq/start sampling."""

    def __init__(
        self,
        base_dataset,
        batch_size,
        split="train",
        shuffle=True,
        drop_last=False,
        horizon_hours=(1, 4, 24, 72),
        history_choices_by_horizon=None,
        sample_nan_ratio_threshold=0.0,
        fixed_seq_len=0,
        distributed_rank=0,
        distributed_world_size=1,
        eval_sample_stride=6,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.batch_size = int(batch_size)
        self.split = split
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.horizon_hours = tuple(int(x) for x in horizon_hours)
        self.horizon_to_id = {hours: idx for idx, hours in enumerate(self.horizon_hours)}
        self.history_choices_by_horizon = history_choices_by_horizon or {
            1: (8, 16, 24),
            4: (16, 24, 48),
            24: (24, 48, 72),
            72: (72, 96, 144),
        }
        self.sample_nan_ratio_threshold = float(sample_nan_ratio_threshold)
        self.fixed_seq_len = int(fixed_seq_len or 0)
        self.distributed_rank = int(distributed_rank or 0)
        self.distributed_world_size = max(1, int(distributed_world_size or 1))
        self.eval_sample_stride = int(6 if eval_sample_stride is None else eval_sample_stride)
        if self.eval_sample_stride < 1:
            raise ValueError("eval_sample_stride must be a positive integer")
        self.has_strict_sample_index = bool(
            getattr(self.base_dataset, "sample_index_filter", None)
            and self.base_dataset.sample_index_filter.strict
        )
        self.station_indices = list(range(len(self.base_dataset.station_records)))
        if self.distributed_world_size > 1:
            self.station_indices = self.station_indices[self.distributed_rank::self.distributed_world_size]
            if not self.station_indices:
                self.station_indices = list(range(len(self.base_dataset.station_records)))
        self.max_context_len = int(base_dataset.seq_len)
        self.max_pred_len = int(max(self.horizon_hours))
        self.task_summaries = [
            {
                "task_name": "pv_forecasting",
                "resolution": base_dataset.task_spec.resolution,
                "seq_len": self.max_context_len,
                "label_len": self.max_pred_len,
                "pred_len": self.max_pred_len,
                "max_context_hours": self.max_context_len,
                "history_choices_hours": (self.fixed_seq_len,) if self.fixed_seq_len > 0 else tuple(sorted({h for values in self.history_choices_by_horizon.values() for h in values})),
                "horizon_hours": self.horizon_hours,
                "num_samples": sum(max(0, len(payload["target_values"]) - self.max_context_len - self.max_pred_len + 1) for payload in base_dataset.station_payloads),
                "num_stations": len(base_dataset.station_records),
            }
        ]
        self.model_dims = self._infer_dims()

    def _owned_batch(self, batch_id):
        return self.distributed_world_size <= 1 or (batch_id % self.distributed_world_size) == self.distributed_rank

    def _infer_dims(self):
        sample = self.base_dataset[0]
        return {
            "target_dim": int(sample.chronos_core.past_target.shape[-1]),
            "historical_covariate_dim": int(sample.chronos_core.historical_covariates.shape[-1]),
            "future_covariate_dim": int(sample.chronos_core.future_covariates.shape[-1]),
            "past_time_dim": int(sample.chronos_core.past_time_features.shape[-1]),
            "future_time_dim": int(sample.chronos_core.future_time_features.shape[-1]),
            "static_dim": int(sample.chronos_core.static_features.shape[-1]) + 2,
            "context_input_dim": int(sample.chronos_core.past_target.shape[-1])
            + int(sample.chronos_core.historical_covariates.shape[-1])
            + int(sample.chronos_core.past_time_features.shape[-1]),
            "future_input_dim": int(sample.chronos_core.future_covariates.shape[-1])
            + int(sample.chronos_core.future_time_features.shape[-1]),
            "output_dim": int(sample.chronos_core.future_target.shape[-1]),
            "num_horizons": len(self.horizon_hours),
        }

    def _sample_horizon(self):
        return random.choice(self.horizon_hours)

    def _sample_history_length(self, horizon_hours):
        if self.fixed_seq_len > 0:
            return min(self.max_context_len, self.fixed_seq_len)
        choices = tuple(int(x) for x in self.history_choices_by_horizon[int(horizon_hours)])
        return random.choice(choices)

    def _sample_station_index(self):
        return random.choice(self.station_indices)

    def _sample_start(self, payload, pred_len):
        series_length = len(payload["target_values"])
        max_start = series_length - self.max_context_len - pred_len
        if max_start < 0:
            return None
        if self.split == "train":
            return random.randint(0, max_start)
        return 0

    def _iter_eval_starts(self, payload, pred_len):
        valid_starts = payload.get("valid_start_indices")
        if valid_starts is not None:
            selected_starts = (
                valid_starts
                if self.has_strict_sample_index
                else valid_starts[::self.eval_sample_stride]
            )
            for start in selected_starts:
                yield int(start)
            return

        series_length = len(payload["target_values"])
        max_start = series_length - self.max_context_len - pred_len
        if max_start < 0:
            return
        stride = 1 if self.has_strict_sample_index else self.eval_sample_stride
        for start in range(0, max_start + 1, stride):
            yield start

    def _is_valid_window(self, payload, start, seq_len, pred_len):
        s_begin = int(start)
        s_end = s_begin + self.max_context_len
        f_begin = s_end
        f_end = f_begin + pred_len

        past_mask = payload["target_mask"][s_end - seq_len:s_end]
        future_mask = payload["target_mask"][f_begin:f_end]

        if past_mask.size == 0 or future_mask.size == 0:
            return False
        if float(np.sum(past_mask)) <= 0:
            return False
        if float(np.sum(future_mask)) <= 0:
            return False
        total_count = past_mask.size + future_mask.size
        invalid_ratio = 1.0 - (float(np.sum(past_mask)) + float(np.sum(future_mask))) / float(total_count)
        if invalid_ratio > self.sample_nan_ratio_threshold:
            return False
        return True

    def _build_sample_from_payload(self, station_index, start, seq_len, pred_len, horizon_hours):
        row = self.base_dataset.station_records[station_index]
        payload = self.base_dataset.station_payloads[station_index]
        s_begin = int(start)
        s_end = s_begin + self.max_context_len
        f_begin = s_end
        f_end = f_begin + pred_len

        full_past_target = payload["target_values"][s_begin:s_end]
        full_past_mask = payload["target_mask"][s_begin:s_end]
        full_history_cov = payload["history_values"][s_begin:s_end]
        full_history_cov_mask = payload["history_mask"][s_begin:s_end]
        full_past_time = payload["time_values"][s_begin:s_end]

        future_target = payload["target_values"][f_begin:f_end]
        future_mask = payload["target_mask"][f_begin:f_end]
        future_cov = payload["future_values"][f_begin:f_end]
        future_cov_mask = payload["future_mask"][f_begin:f_end]
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

        full_history_cov, full_history_cov_mask = _soft_mask_covariate_block(
            full_history_cov,
            full_history_cov_mask,
            min_valid_ratio=self.base_dataset.min_history_covariate_valid_ratio,
            min_std=self.base_dataset.min_history_covariate_std,
        )
        future_cov, future_cov_mask = _soft_mask_covariate_block(
            future_cov,
            future_cov_mask,
            min_valid_ratio=self.base_dataset.min_future_covariate_valid_ratio,
            min_std=self.base_dataset.min_future_covariate_std,
        )

        static_features = np.array(
            [
                float(seq_len),
                float(pred_len),
                float(pred_len),
                float(horizon_hours),
                float(self.max_context_len),
                float(self.max_pred_len),
            ],
            dtype=np.float32,
        )

        return {
            "task_name": "pv_forecasting",
            "resolution": self.base_dataset.task_spec.resolution,
            "station_id": row["station_dir"],
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
            "horizon_hours": int(horizon_hours),
            "horizon_id": int(self.horizon_to_id[int(horizon_hours)]),
            "past_target": full_past_target[-seq_len:],
            "past_observed_mask": full_past_mask[-seq_len:],
            "historical_covariates": full_history_cov[-seq_len:],
            "historical_covariates_mask": full_history_cov_mask[-seq_len:],
            "historical_covariates_native": hist_native_values,
            "historical_covariates_native_mask": hist_native_mask,
            "future_covariates": future_cov,
            "future_covariates_mask": future_cov_mask,
            "future_covariates_native": fut_native_values,
            "future_covariates_native_mask": fut_native_mask,
            "future_target": future_target,
            "future_target_mask": future_mask,
            "past_time_features": full_past_time[-seq_len:],
            "future_time_features": future_time,
            "historical_covariates_native_time_features": hist_native_time,
            "future_covariates_native_time_features": fut_native_time,
            "static_features": static_features,
            "metadata": {
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
                "historical_covariates_native_resolution": hist_native_resolution,
                "future_covariates_native_resolution": fut_native_resolution,
                "historical_covariates_native_len": int(hist_native_values.shape[0]),
                "future_covariates_native_len": int(fut_native_values.shape[0]),
            },
        }

    def _sample_batch_samples(self):
        samples = []
        attempts = 0
        max_attempts = max(100, self.batch_size * 20)
        while len(samples) < self.batch_size and attempts < max_attempts:
            attempts += 1
            horizon_hours = self._sample_horizon()
            pred_len = int(horizon_hours)
            seq_len = self._sample_history_length(horizon_hours)
            station_index = self._sample_station_index()
            payload = self.base_dataset.station_payloads[station_index]
            start = self._sample_start(payload, pred_len)
            if start is None:
                continue
            if not self._is_valid_window(payload, start, seq_len, pred_len):
                continue
            samples.append(self._build_sample_from_payload(station_index, start, seq_len, pred_len, horizon_hours))
        if len(samples) < self.batch_size and self.drop_last:
            return None
        return samples

    def _build_batch(self, samples):
        if not samples:
            return None
        batch_size = len(samples)
        dims = self.model_dims

        past_target = np.zeros((batch_size, self.max_context_len, dims["target_dim"]), dtype=np.float32)
        past_observed_mask = np.zeros((batch_size, self.max_context_len, dims["target_dim"]), dtype=np.float32)
        historical_covariates = np.zeros((batch_size, self.max_context_len, dims["historical_covariate_dim"]), dtype=np.float32)
        historical_covariates_mask = np.zeros((batch_size, self.max_context_len, dims["historical_covariate_dim"]), dtype=np.float32)
        future_covariates = np.zeros((batch_size, self.max_pred_len, dims["future_covariate_dim"]), dtype=np.float32)
        future_covariates_mask = np.zeros((batch_size, self.max_pred_len, dims["future_covariate_dim"]), dtype=np.float32)
        future_target = np.zeros((batch_size, self.max_pred_len, dims["output_dim"]), dtype=np.float32)
        future_target_mask = np.zeros((batch_size, self.max_pred_len, dims["output_dim"]), dtype=np.float32)
        past_time_features = np.zeros((batch_size, self.max_context_len, dims["past_time_dim"]), dtype=np.float32)
        future_time_features = np.zeros((batch_size, self.max_pred_len, dims["future_time_dim"]), dtype=np.float32)
        static_features = np.zeros((batch_size, dims["static_dim"]), dtype=np.float32)
        horizon_ids = np.zeros((batch_size,), dtype=np.int64)
        horizon_hours = np.zeros((batch_size,), dtype=np.int64)
        native_hist_len = max((sample.get("historical_covariates_native", np.zeros((0, 0), dtype=np.float32)).shape[0] for sample in samples), default=0)
        native_fut_len = max((sample.get("future_covariates_native", np.zeros((0, 0), dtype=np.float32)).shape[0] for sample in samples), default=0)
        native_hist_time_dim = max((sample.get("historical_covariates_native_time_features", np.zeros((0, 0))).shape[-1] if sample.get("historical_covariates_native_time_features", np.zeros((0, 0))).ndim == 2 else 0 for sample in samples), default=0)
        native_fut_time_dim = max((sample.get("future_covariates_native_time_features", np.zeros((0, 0))).shape[-1] if sample.get("future_covariates_native_time_features", np.zeros((0, 0))).ndim == 2 else 0 for sample in samples), default=0)
        historical_covariates_native = np.zeros((batch_size, native_hist_len, dims["historical_covariate_dim"]), dtype=np.float32)
        historical_covariates_native_mask = np.zeros((batch_size, native_hist_len, dims["historical_covariate_dim"]), dtype=np.float32)
        future_covariates_native = np.zeros((batch_size, native_fut_len, dims["future_covariate_dim"]), dtype=np.float32)
        future_covariates_native_mask = np.zeros((batch_size, native_fut_len, dims["future_covariate_dim"]), dtype=np.float32)
        historical_covariates_native_time_features = np.zeros((batch_size, native_hist_len, native_hist_time_dim), dtype=np.float32)
        future_covariates_native_time_features = np.zeros((batch_size, native_fut_len, native_fut_time_dim), dtype=np.float32)

        task_names = []
        resolutions = []
        station_ids = []
        seq_lens = []
        pred_lens = []
        metadata = []

        for idx, sample in enumerate(samples):
            seq_len = int(sample["seq_len"])
            pred_len = int(sample["pred_len"])

            past_target[idx, -seq_len:] = _pad_last_dim(sample["past_target"], dims["target_dim"])
            past_observed_mask[idx, -seq_len:] = _pad_last_dim(sample["past_observed_mask"], dims["target_dim"])
            if dims["historical_covariate_dim"] > 0:
                historical_covariates[idx, -seq_len:] = _pad_last_dim(sample["historical_covariates"], dims["historical_covariate_dim"])
                historical_covariates_mask[idx, -seq_len:] = _pad_last_dim(sample["historical_covariates_mask"], dims["historical_covariate_dim"])
            if dims["past_time_dim"] > 0:
                past_time_features[idx, -seq_len:] = _pad_last_dim(sample["past_time_features"], dims["past_time_dim"])

            future_target[idx, :pred_len] = _pad_last_dim(sample["future_target"], dims["output_dim"])
            future_target_mask[idx, :pred_len] = _pad_last_dim(sample["future_target_mask"], dims["output_dim"])
            if dims["future_covariate_dim"] > 0:
                future_covariates[idx, :pred_len] = _pad_last_dim(sample["future_covariates"], dims["future_covariate_dim"])
                future_covariates_mask[idx, :pred_len] = _pad_last_dim(sample["future_covariates_mask"], dims["future_covariate_dim"])
            if dims["future_time_dim"] > 0:
                future_time_features[idx, :pred_len] = _pad_last_dim(sample["future_time_features"], dims["future_time_dim"])
            if dims["historical_covariate_dim"] > 0 and native_hist_len > 0:
                hist_native = sample.get("historical_covariates_native")
                hist_native_mask = sample.get("historical_covariates_native_mask")
                if hist_native is not None and hist_native.shape[0] > 0:
                    hist_native_len = hist_native.shape[0]
                    historical_covariates_native[idx, :hist_native_len] = _pad_last_dim(hist_native, dims["historical_covariate_dim"])
                    historical_covariates_native_mask[idx, :hist_native_len] = _pad_last_dim(hist_native_mask, dims["historical_covariate_dim"])
                    if native_hist_time_dim > 0:
                        historical_covariates_native_time_features[idx, :hist_native_len] = _pad_last_dim(
                            sample.get("historical_covariates_native_time_features", np.zeros((hist_native_len, native_hist_time_dim), dtype=np.float32)),
                            native_hist_time_dim,
                        )
            if dims["future_covariate_dim"] > 0 and native_fut_len > 0:
                fut_native = sample.get("future_covariates_native")
                fut_native_mask = sample.get("future_covariates_native_mask")
                if fut_native is not None and fut_native.shape[0] > 0:
                    fut_native_len = fut_native.shape[0]
                    future_covariates_native[idx, :fut_native_len] = _pad_last_dim(fut_native, dims["future_covariate_dim"])
                    future_covariates_native_mask[idx, :fut_native_len] = _pad_last_dim(fut_native_mask, dims["future_covariate_dim"])
                    if native_fut_time_dim > 0:
                        future_covariates_native_time_features[idx, :fut_native_len] = _pad_last_dim(
                            sample.get("future_covariates_native_time_features", np.zeros((fut_native_len, native_fut_time_dim), dtype=np.float32)),
                            native_fut_time_dim,
                        )

            static_features[idx, : sample["static_features"].shape[-1]] = sample["static_features"]
            horizon_ids[idx] = int(sample["horizon_id"])
            horizon_hours[idx] = int(sample["horizon_hours"])

            task_names.append(sample["task_name"])
            resolutions.append(sample["resolution"])
            station_ids.append(sample["station_id"])
            seq_lens.append(seq_len)
            pred_lens.append(pred_len)
            metadata.append(sample["metadata"])

        context_padding_mask = (past_observed_mask.sum(axis=-1) > 0).astype(np.float32)

        return {
            "past_target": torch.from_numpy(past_target),
            "past_observed_mask": torch.from_numpy(past_observed_mask),
            "historical_covariates": torch.from_numpy(historical_covariates),
            "historical_covariates_mask": torch.from_numpy(historical_covariates_mask),
            "future_covariates": torch.from_numpy(future_covariates),
            "future_covariates_mask": torch.from_numpy(future_covariates_mask),
            "historical_covariates_native": torch.from_numpy(historical_covariates_native),
            "historical_covariates_native_mask": torch.from_numpy(historical_covariates_native_mask),
            "future_covariates_native": torch.from_numpy(future_covariates_native),
            "future_covariates_native_mask": torch.from_numpy(future_covariates_native_mask),
            "future_target": torch.from_numpy(future_target),
            "future_target_mask": torch.from_numpy(future_target_mask),
            "past_time_features": torch.from_numpy(past_time_features),
            "future_time_features": torch.from_numpy(future_time_features),
            "historical_covariates_native_time_features": torch.from_numpy(historical_covariates_native_time_features),
            "future_covariates_native_time_features": torch.from_numpy(future_covariates_native_time_features),
            "static_features": torch.from_numpy(static_features),
            "context_padding_mask": torch.from_numpy(context_padding_mask),
            "horizon_ids": torch.from_numpy(horizon_ids),
            "horizon_hours": torch.from_numpy(horizon_hours),
            "task_names": task_names,
            "resolutions": resolutions,
            "station_ids": station_ids,
            "seq_lens": seq_lens,
            "pred_lens": pred_lens,
            "metadata": metadata,
        }

    def __iter__(self):
        batch_id = 0
        if self.split == "train":
            while True:
                batch = self._build_batch(self._sample_batch_samples())
                if batch is not None:
                    yield batch
                    batch_id += 1
        else:
            station_order = list(self.station_indices)
            worker_info = get_worker_info()
            if worker_info is not None and worker_info.num_workers > 1:
                station_order = station_order[worker_info.id::worker_info.num_workers]
            if self.shuffle:
                random.shuffle(station_order)
            current = []
            for station_index in station_order:
                for horizon_hours in self.horizon_hours:
                    pred_len = int(horizon_hours)
                    seq_len = min(self.max_context_len, self.fixed_seq_len) if self.fixed_seq_len > 0 else max(self.history_choices_by_horizon[int(horizon_hours)])
                    payload = self.base_dataset.station_payloads[station_index]
                    for start in self._iter_eval_starts(payload, pred_len):
                        if not self._is_valid_window(payload, start, seq_len, pred_len):
                            continue
                        current.append(self._build_sample_from_payload(station_index, start, seq_len, pred_len, horizon_hours))
                        if len(current) >= self.batch_size:
                            yield self._build_batch(current)
                            batch_id += 1
                            current = []
            if current and not self.drop_last:
                yield self._build_batch(current)

    def __len__(self):
        base = self.task_summaries[0]["num_samples"]
        if self.split == "train":
            return base
        total = 0
        for payload in self.base_dataset.station_payloads:
            series_length = len(payload["target_values"])
            for horizon_hours in self.horizon_hours:
                pred_len = int(horizon_hours)
                max_start = series_length - self.max_context_len - pred_len
                if max_start < 0:
                    continue
                seq_len = min(self.max_context_len, self.fixed_seq_len) if self.fixed_seq_len > 0 else max(self.history_choices_by_horizon[int(horizon_hours)])
                for start in range(max_start + 1):
                    if self._is_valid_window(payload, start, seq_len, pred_len):
                        total += 1
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)


def build_chronos_batch_dataset(
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
    batch_size=8,
    shuffle=None,
    drop_last=False,
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
    fixed_seq_len=0,
    train_window_sample_stride=1,
    distributed_rank=0,
    distributed_world_size=1,
    target_transform="",
    target_normalization="none",
    target_standardization="auto",
    capacity_proxy_quantile=99.5,
    target_negative_sentinel=-1e5,
    night_small_negative_abs_kw=10.0,
    night_small_negative_capacity_frac=0.05,
    target_extreme_positive_capacity_frac=1.5,
    sample_index_csv="",
    train_sampler_mode="default",
    region_balance_alpha=0.5,
    region_balance_max_prob=0.0,
    region_balance_max_repeat_per_epoch=0.0,
    region_loss_alpha=0.0,
    dataset_cache_dir="",
    use_dataset_cache=False,
    binary_cache_registry="",
    binary_index_overlay_dir="",
    binary_dataset_dir="",
    eval_sample_stride=6,
    dataset_pipeline="legacy",
    include_native_covariates=False,
    time_feature_schema=None,
):
    dataset_pipeline = str(dataset_pipeline or "legacy").strip().lower()
    if dataset_pipeline not in {"legacy", "indexed", "binary_indexed", "multires_binary_indexed"}:
        raise ValueError(
            f"Unsupported dataset_pipeline={dataset_pipeline!r}; "
            "expected 'legacy', 'indexed', or 'binary_indexed'."
        )
    if len(task_names) == 1 and task_names[0] == "pv_forecasting":
        if dataset_pipeline in {"indexed", "binary_indexed", "multires_binary_indexed"}:
            raise ValueError(f"dataset_pipeline={dataset_pipeline} is not implemented for task_name=pv_forecasting yet.")
        base_dataset = PVTaskDataset(
            manifest_path=manifest_path,
            station_data_root=station_data_root,
            task_name="pv_72h_ahead",
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
            strict_hourly_resample=strict_hourly_resample,
            hourly_resample_mode=hourly_resample_mode,
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
            dataset_cache_dir=dataset_cache_dir,
            use_dataset_cache=use_dataset_cache,
        )
        if shuffle is None:
            shuffle = split == "train"
        return UnifiedPVForecastBatchDataset(
            base_dataset=base_dataset,
            batch_size=batch_size,
            split=split,
            shuffle=shuffle,
            drop_last=drop_last,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            fixed_seq_len=fixed_seq_len,
            train_window_sample_stride=train_window_sample_stride,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            eval_sample_stride=eval_sample_stride,
        )
    if dataset_pipeline == "binary_indexed":
        from foundation.data.binary_indexed_chronos import build_binary_indexed_chronos_dataset

        return build_binary_indexed_chronos_dataset(
            binary_dataset_dir=binary_dataset_dir,
            manifest_path=manifest_path,
            task_names=task_names,
            split=split,
            regions=regions,
            station_dirs=station_dirs,
            max_stations=max_stations,
            batch_size=batch_size,
            shuffle=shuffle if shuffle is not None else split == "train",
            drop_last=drop_last,
            fixed_seq_len=fixed_seq_len,
            train_window_sample_stride=train_window_sample_stride,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            train_sampler_mode=train_sampler_mode if split == "train" else "default",
            region_balance_alpha=region_balance_alpha,
            region_balance_max_prob=region_balance_max_prob,
            region_balance_max_repeat_per_epoch=region_balance_max_repeat_per_epoch,
            region_loss_alpha=region_loss_alpha if split == "train" else 0.0,
            eval_sample_stride=eval_sample_stride,
            include_native_covariates=include_native_covariates,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            min_history_covariate_valid_ratio=min_history_covariate_valid_ratio,
            min_future_covariate_valid_ratio=min_future_covariate_valid_ratio,
            min_history_covariate_std=min_history_covariate_std,
            min_future_covariate_std=min_future_covariate_std,
            sample_index_csv=sample_index_csv,
            time_feature_schema=time_feature_schema,
        )
    if dataset_pipeline == "multires_binary_indexed":
        from foundation.data.multires_binary_indexed_chronos import build_multires_binary_indexed_chronos_dataset

        return build_multires_binary_indexed_chronos_dataset(
            binary_cache_registry=binary_cache_registry,
            binary_index_overlay_dir=binary_index_overlay_dir,
            manifest_path=manifest_path,
            task_names=task_names,
            split=split,
            regions=regions,
            station_dirs=station_dirs,
            max_stations=max_stations,
            batch_size=batch_size,
            shuffle=shuffle if shuffle is not None else split == "train",
            drop_last=drop_last,
            fixed_seq_len=fixed_seq_len,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            train_sampler_mode=train_sampler_mode if split == "train" else "default",
            region_balance_alpha=region_balance_alpha,
            region_balance_max_prob=region_balance_max_prob,
            region_balance_max_repeat_per_epoch=region_balance_max_repeat_per_epoch,
            region_loss_alpha=region_loss_alpha if split == "train" else 0.0,
            eval_sample_stride=eval_sample_stride,
            include_native_covariates=include_native_covariates,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            min_history_covariate_valid_ratio=min_history_covariate_valid_ratio,
            min_future_covariate_valid_ratio=min_future_covariate_valid_ratio,
            min_history_covariate_std=min_history_covariate_std,
            min_future_covariate_std=min_future_covariate_std,
            time_feature_schema=time_feature_schema,
        )
    task_datasets = build_task_datasets(
        manifest_path=manifest_path,
        station_data_root=station_data_root,
        task_names=task_names,
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
        strict_hourly_resample=strict_hourly_resample,
        hourly_resample_mode=hourly_resample_mode,
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
        dataset_cache_dir=dataset_cache_dir,
        use_dataset_cache=use_dataset_cache,
    )
    if shuffle is None:
        shuffle = split == "train"
    if dataset_pipeline == "indexed":
        from foundation.data.indexed_chronos import build_indexed_chronos_dataset

        return build_indexed_chronos_dataset(
            task_datasets=task_datasets,
            batch_size=batch_size,
            split=split,
            shuffle=shuffle,
            drop_last=drop_last,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            fixed_seq_len=fixed_seq_len,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            train_sampler_mode=train_sampler_mode,
            region_balance_alpha=region_balance_alpha,
            region_balance_max_prob=region_balance_max_prob,
            region_balance_max_repeat_per_epoch=region_balance_max_repeat_per_epoch,
            region_loss_alpha=region_loss_alpha,
            eval_sample_stride=eval_sample_stride,
            include_native_covariates=include_native_covariates,
            time_feature_schema=time_feature_schema,
        )
    return ChronosLikePVBatchDataset(
        task_datasets=task_datasets,
        batch_size=batch_size,
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        sample_nan_ratio_threshold=sample_nan_ratio_threshold,
        fixed_seq_len=fixed_seq_len,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        train_sampler_mode=train_sampler_mode,
        region_balance_alpha=region_balance_alpha,
        region_balance_max_prob=region_balance_max_prob,
        region_balance_max_repeat_per_epoch=region_balance_max_repeat_per_epoch,
        eval_sample_stride=eval_sample_stride,
        time_feature_schema=time_feature_schema,
    )
