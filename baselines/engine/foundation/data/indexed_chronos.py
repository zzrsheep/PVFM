import math

import numpy as np
import torch
from torch.utils.data import Dataset

from foundation.data.chronos_batch import ChronosLikePVBatchDataset, _pad_last_dim
from foundation.data.datasets import _slice_native_covariate_window, _soft_mask_covariate_block
from foundation.data.region_balanced_sampler import canonical_region_key, sample_weights_from_region_keys


# Keep the indexed wrapper's calendar contract explicit.  The binary/multires
# implementation defines the same versioned names, but importing it here would
# create a cycle (that module imports ``FastChronosCollator`` from this file).
LEGACY_TIME_FEATURE_SCHEMA = "legacy_right_pad_v1"
CANONICAL_TIME_FEATURE_SCHEMA = "minute_hour_weekday_day_dayofyear_v1"
SUPPORTED_TIME_FEATURE_SCHEMAS = {
    LEGACY_TIME_FEATURE_SCHEMA,
    CANONICAL_TIME_FEATURE_SCHEMA,
}


def _normalize_time_feature_schema(value):
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return None
    if text in {"legacy", LEGACY_TIME_FEATURE_SCHEMA}:
        return LEGACY_TIME_FEATURE_SCHEMA
    if text in {"canonical", CANONICAL_TIME_FEATURE_SCHEMA}:
        return CANONICAL_TIME_FEATURE_SCHEMA
    raise ValueError(f"unsupported indexed time_feature_schema={value!r}.")


def _align_indexed_time_features(values, resolution, target_dim, schema):
    """Align one indexed task's calendar channels to the joint model width.

    ``PVTaskDataset`` emits four channels for 1h and five for sub-hourly
    tasks.  A mixed indexed batch therefore needs the same minute-zero
    insertion used by the registry-backed multires wrapper; plain collator
    right-padding would shift hour/weekday/day-of-year semantics for 1h rows.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"indexed time features must be rank-2, got shape={array.shape}.")
    target_dim = int(target_dim)
    if target_dim <= 0 or array.shape[-1] > target_dim:
        raise ValueError(
            f"cannot align indexed time features shape={array.shape} to target_dim={target_dim}."
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


class FastChronosCollator:
    """Collate lightweight PV windows into the Chronos-style batch contract."""

    def __init__(self, model_dims, split="train", metadata_mode="minimal", include_native_covariates=False):
        self.model_dims = dict(model_dims)
        self.split = str(split or "train")
        self.metadata_mode = str(metadata_mode or "minimal").strip().lower()
        self.include_native_covariates = bool(include_native_covariates)

    def _sample_is_valid(self, item):
        past_mask = item["past_observed_mask"]
        future_mask = item["future_observed_mask"]
        if past_mask.size == 0 or future_mask.size == 0:
            return False
        return float(np.sum(past_mask)) > 0.0 and float(np.sum(future_mask)) > 0.0

    def _can_stack_core(self, items, seq_len, pred_len):
        dims = self.model_dims
        history_shapes = {
            "past_target": dims["target_dim"],
            "past_observed_mask": dims["target_dim"],
            "historical_covariates": dims["historical_covariate_dim"],
            "historical_covariates_mask": dims["historical_covariate_dim"],
            "past_time_features": dims["past_time_dim"],
        }
        future_shapes = {
            "future_target": (pred_len, dims["output_dim"]),
            "future_observed_mask": (pred_len, dims["output_dim"]),
            "future_covariates": (pred_len, dims["future_covariate_dim"]),
            "future_covariates_mask": (pred_len, dims["future_covariate_dim"]),
            "future_time_features": (pred_len, dims["future_time_dim"]),
            "static_features": (dims["static_dim"],),
            "site_features": (5,),
        }
        for item in items:
            for key, dim in history_shapes.items():
                value = item[key]
                if value.ndim != 2 or value.shape[0] < seq_len or value.shape[-1] != dim:
                    return False
            for key, shape in future_shapes.items():
                if tuple(item[key].shape) != tuple(shape):
                    return False
        return True

    def _stack_core_batch(self, items, seq_len, pred_len):
        static_features = np.stack([item["static_features"].copy() for item in items]).astype(np.float32, copy=False)
        if static_features.shape[-1] > 0:
            static_features[:, 0] = float(seq_len)
        return {
            "past_target": np.stack([item["past_target"][-seq_len:] for item in items]).astype(np.float32, copy=False),
            "past_observed_mask": np.stack([item["past_observed_mask"][-seq_len:] for item in items]).astype(np.float32, copy=False),
            "historical_covariates": np.stack([item["historical_covariates"][-seq_len:] for item in items]).astype(np.float32, copy=False),
            "historical_covariates_mask": np.stack([item["historical_covariates_mask"][-seq_len:] for item in items]).astype(np.float32, copy=False),
            "future_covariates": np.stack([item["future_covariates"] for item in items]).astype(np.float32, copy=False),
            "future_covariates_mask": np.stack([item["future_covariates_mask"] for item in items]).astype(np.float32, copy=False),
            "future_target": np.stack([item["future_target"] for item in items]).astype(np.float32, copy=False),
            "future_target_mask": np.stack([item["future_observed_mask"] for item in items]).astype(np.float32, copy=False),
            "past_time_features": np.stack([item["past_time_features"][-seq_len:] for item in items]).astype(np.float32, copy=False),
            "future_time_features": np.stack([item["future_time_features"] for item in items]).astype(np.float32, copy=False),
            "static_features": static_features,
            "site_features": np.stack([item["site_features"] for item in items]).astype(np.float32, copy=False),
        }

    def __call__(self, items, history_length=None):
        original_items = list(items)
        items = [item for item in original_items if self._sample_is_valid(item)]
        if not items and original_items:
            items = [original_items[0]]
        if not items:
            raise ValueError("FastChronosCollator received no valid samples.")

        dims = self.model_dims
        batch_size = len(items)
        seq_len = int(history_length or items[0]["seq_len"])
        pred_len = int(items[0]["pred_len"])

        task_names = []
        resolutions = []
        station_ids = []
        region_keys = []
        sample_weights = []
        seq_lens = []
        pred_lens = []
        metadata = []

        stack_core = self._can_stack_core(items, seq_len, pred_len)
        collator_mode = "stack" if stack_core else "fallback"
        if stack_core:
            arrays = self._stack_core_batch(items, seq_len, pred_len)
            past_target = arrays["past_target"]
            past_observed_mask = arrays["past_observed_mask"]
            historical_covariates = arrays["historical_covariates"]
            historical_covariates_mask = arrays["historical_covariates_mask"]
            future_covariates = arrays["future_covariates"]
            future_covariates_mask = arrays["future_covariates_mask"]
            future_target = arrays["future_target"]
            future_target_mask = arrays["future_target_mask"]
            past_time_features = arrays["past_time_features"]
            future_time_features = arrays["future_time_features"]
            static_features = arrays["static_features"]
            site_features = arrays["site_features"]
        else:
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

            for idx, item in enumerate(items):
                past_target[idx] = _pad_last_dim(item["past_target"][-seq_len:], dims["target_dim"])
                past_observed_mask[idx] = _pad_last_dim(item["past_observed_mask"][-seq_len:], dims["target_dim"])
                future_target[idx] = _pad_last_dim(item["future_target"], dims["output_dim"])
                future_target_mask[idx] = _pad_last_dim(item["future_observed_mask"], dims["output_dim"])

                if dims["historical_covariate_dim"] > 0:
                    historical_covariates[idx] = _pad_last_dim(
                        item["historical_covariates"][-seq_len:],
                        dims["historical_covariate_dim"],
                    )
                    historical_covariates_mask[idx] = _pad_last_dim(
                        item["historical_covariates_mask"][-seq_len:],
                        dims["historical_covariate_dim"],
                    )
                if dims["future_covariate_dim"] > 0:
                    future_covariates[idx] = _pad_last_dim(item["future_covariates"], dims["future_covariate_dim"])
                    future_covariates_mask[idx] = _pad_last_dim(item["future_covariates_mask"], dims["future_covariate_dim"])
                if dims["past_time_dim"] > 0:
                    past_time_features[idx] = _pad_last_dim(item["past_time_features"][-seq_len:], dims["past_time_dim"])
                if dims["future_time_dim"] > 0:
                    future_time_features[idx] = _pad_last_dim(item["future_time_features"], dims["future_time_dim"])

                static_value = item["static_features"].copy()
                if static_value.shape[-1] > 0:
                    static_value[0] = float(seq_len)
                static_features[idx, : static_value.shape[-1]] = static_value
                site_features[idx] = item["site_features"]

        for item in items:
            task_names.append(item["task_name"])
            resolutions.append(item["resolution"])
            station_ids.append(item["station_id"])
            region_keys.append(item.get("region_key", "UNKNOWN"))
            sample_weights.append(float(item.get("sample_weight", 1.0)))
            seq_lens.append(seq_len)
            pred_lens.append(item["pred_len"])
            if self.metadata_mode == "full":
                sample_metadata = dict(item["metadata"])
                sample_metadata["sampled_history_length"] = seq_len
                metadata.append(sample_metadata)

        context_padding_mask = (past_observed_mask.sum(axis=-1) > 0).astype(np.float32)
        historical_covariate_mean = np.stack([
            np.asarray(item.get("historical_covariate_mean", np.zeros(dims["historical_covariate_dim"])), dtype=np.float32)
            for item in items
        ])
        historical_covariate_scale = np.stack([
            np.asarray(item.get("historical_covariate_scale", np.ones(dims["historical_covariate_dim"])), dtype=np.float32)
            for item in items
        ])
        batch = {
            "past_target": torch.from_numpy(past_target),
            "past_observed_mask": torch.from_numpy(past_observed_mask),
            "historical_covariates": torch.from_numpy(historical_covariates),
            "historical_covariates_mask": torch.from_numpy(historical_covariates_mask),
            "historical_covariate_mean": torch.from_numpy(historical_covariate_mean),
            "historical_covariate_scale": torch.from_numpy(historical_covariate_scale),
            "future_covariates": torch.from_numpy(future_covariates),
            "future_covariates_mask": torch.from_numpy(future_covariates_mask),
            "future_target": torch.from_numpy(future_target),
            "future_target_mask": torch.from_numpy(future_target_mask),
            "past_time_features": torch.from_numpy(past_time_features),
            "future_time_features": torch.from_numpy(future_time_features),
            "static_features": torch.from_numpy(static_features),
            "site_features": torch.from_numpy(site_features),
            "context_padding_mask": torch.from_numpy(context_padding_mask),
            "collator_mode": collator_mode,
            "task_names": task_names,
            "resolutions": resolutions,
            "station_ids": station_ids,
            "region_keys": region_keys,
            "seq_lens": seq_lens,
            "pred_lens": pred_lens,
        }
        if any(abs(weight - 1.0) > 1e-7 for weight in sample_weights):
            batch["sample_weights"] = torch.tensor(sample_weights, dtype=torch.float32)
        if self.include_native_covariates:
            native_hist_len = max((item["historical_covariates_native"].shape[0] for item in items), default=0)
            native_fut_len = max((item["future_covariates_native"].shape[0] for item in items), default=0)
            native_hist_time_dim = max(
                (
                    item["historical_covariates_native_time_features"].shape[-1]
                    if item["historical_covariates_native_time_features"].ndim == 2
                    else 0
                    for item in items
                ),
                default=0,
            )
            native_fut_time_dim = max(
                (
                    item["future_covariates_native_time_features"].shape[-1]
                    if item["future_covariates_native_time_features"].ndim == 2
                    else 0
                    for item in items
                ),
                default=0,
            )
            historical_covariates_native = np.zeros((batch_size, native_hist_len, dims["historical_covariate_dim"]), dtype=np.float32)
            historical_covariates_native_mask = np.zeros((batch_size, native_hist_len, dims["historical_covariate_dim"]), dtype=np.float32)
            future_covariates_native = np.zeros((batch_size, native_fut_len, dims["future_covariate_dim"]), dtype=np.float32)
            future_covariates_native_mask = np.zeros((batch_size, native_fut_len, dims["future_covariate_dim"]), dtype=np.float32)
            historical_covariates_native_time_features = np.zeros((batch_size, native_hist_len, native_hist_time_dim), dtype=np.float32)
            future_covariates_native_time_features = np.zeros((batch_size, native_fut_len, native_fut_time_dim), dtype=np.float32)
            for idx, item in enumerate(items):
                if dims["historical_covariate_dim"] > 0 and native_hist_len > 0:
                    hist_native_len = item["historical_covariates_native"].shape[0]
                    if hist_native_len > 0:
                        historical_covariates_native[idx, :hist_native_len] = _pad_last_dim(
                            item["historical_covariates_native"],
                            dims["historical_covariate_dim"],
                        )
                        historical_covariates_native_mask[idx, :hist_native_len] = _pad_last_dim(
                            item["historical_covariates_native_mask"],
                            dims["historical_covariate_dim"],
                        )
                        if native_hist_time_dim > 0:
                            historical_covariates_native_time_features[idx, :hist_native_len] = _pad_last_dim(
                                item["historical_covariates_native_time_features"],
                                native_hist_time_dim,
                            )
                if dims["future_covariate_dim"] > 0 and native_fut_len > 0:
                    fut_native_len = item["future_covariates_native"].shape[0]
                    if fut_native_len > 0:
                        future_covariates_native[idx, :fut_native_len] = _pad_last_dim(
                            item["future_covariates_native"],
                            dims["future_covariate_dim"],
                        )
                        future_covariates_native_mask[idx, :fut_native_len] = _pad_last_dim(
                            item["future_covariates_native_mask"],
                            dims["future_covariate_dim"],
                        )
                        if native_fut_time_dim > 0:
                            future_covariates_native_time_features[idx, :fut_native_len] = _pad_last_dim(
                                item["future_covariates_native_time_features"],
                                native_fut_time_dim,
                            )
            batch.update(
                {
                    "historical_covariates_native": torch.from_numpy(historical_covariates_native),
                    "historical_covariates_native_mask": torch.from_numpy(historical_covariates_native_mask),
                    "future_covariates_native": torch.from_numpy(future_covariates_native),
                    "future_covariates_native_mask": torch.from_numpy(future_covariates_native_mask),
                    "historical_covariates_native_time_features": torch.from_numpy(historical_covariates_native_time_features),
                    "future_covariates_native_time_features": torch.from_numpy(future_covariates_native_time_features),
                }
            )
        if self.metadata_mode == "full":
            batch["metadata"] = metadata
        return batch


class IndexedChronosPVDataset(Dataset):
    """Map-style PV window dataset with a lightweight TSFM-style collator."""

    yields_batches = False

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
        metadata_mode=None,
        include_native_covariates=False,
        region_loss_alpha=0.0,
        time_feature_schema=None,
    ):
        super().__init__()
        self.task_datasets = list(task_datasets)
        self.batch_size = int(batch_size)
        self.split = split
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
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
        self.has_strict_sample_index = any(
            bool(getattr(dataset, "sample_index_filter", None) and dataset.sample_index_filter.strict)
            for dataset in self.task_datasets
        )
        self.metadata_mode = str(metadata_mode or ("minimal" if self.split == "train" else "full")).strip().lower()
        self.include_native_covariates = bool(include_native_covariates)
        self.region_loss_alpha = max(0.0, float(region_loss_alpha or 0.0))
        self.native_fields_in_batch = self.include_native_covariates
        self.collator_mode_hint = "stack_or_fallback"
        if self.metadata_mode not in {"minimal", "full"}:
            raise ValueError("metadata_mode must be 'minimal' or 'full'.")
        if self.train_sampler_mode not in {"default", "region_balanced"}:
            raise ValueError("train_sampler_mode must be 'default' or 'region_balanced'.")

        self._batch_builder = ChronosLikePVBatchDataset(
            task_datasets=self.task_datasets,
            batch_size=self.batch_size,
            split=split,
            shuffle=shuffle,
            drop_last=drop_last,
            sample_nan_ratio_threshold=sample_nan_ratio_threshold,
            fixed_seq_len=fixed_seq_len,
            distributed_rank=0,
            distributed_world_size=1,
            train_sampler_mode="default",
            region_balance_alpha=region_balance_alpha,
            region_balance_max_prob=region_balance_max_prob,
            region_balance_max_repeat_per_epoch=region_balance_max_repeat_per_epoch,
            eval_sample_stride=eval_sample_stride,
        )
        self.model_dims = self._batch_builder.model_dims
        requested_schema = _normalize_time_feature_schema(time_feature_schema)
        if requested_schema is None:
            # Indexed task datasets do not carry cache metadata.  Resolution
            # is therefore the reliable discriminator: any sub-hourly task
            # emits minute/hour/... channels and requires the canonical
            # contract, while an all-1h collection retains the legacy layout.
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
        if self.time_feature_schema == CANONICAL_TIME_FEATURE_SCHEMA:
            # Reserve the minute channel even when the first task is 1h and
            # the builder inferred its dimensions from a four-channel sample.
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
        self.index = self._build_index()
        self.region_keys = self._build_region_keys()
        self.sample_weights = sample_weights_from_region_keys(self.region_keys, self.region_loss_alpha)
        self.task_summaries = self._build_task_summaries()
        self._collator = FastChronosCollator(
            model_dims=self.model_dims,
            split=self.split,
            metadata_mode=self.metadata_mode,
            include_native_covariates=self.include_native_covariates,
        )

    def _build_index(self):
        index = []
        for dataset_idx, dataset in enumerate(self.task_datasets):
            sample_indices = list(range(len(dataset)))
            if self.split != "train" and self.eval_sample_stride > 1 and not self.has_strict_sample_index:
                # An exact origin manifest is already the complete index.
                sample_indices = sample_indices[:: self.eval_sample_stride]
            if self.distributed_world_size > 1:
                sample_indices = sample_indices[self.distributed_rank :: self.distributed_world_size]
            for sample_idx in sample_indices:
                station_index, sample_start = dataset.index[int(sample_idx)]
                index.append((dataset_idx, int(sample_idx), int(station_index), int(sample_start)))
        return index

    def _build_task_summaries(self):
        summaries = []
        counts_by_dataset = [0 for _ in self.task_datasets]
        for dataset_idx, _sample_idx, _station_index, _sample_start in self.index:
            counts_by_dataset[dataset_idx] += 1
        for dataset_idx, summary in enumerate(self._batch_builder.task_summaries):
            item = dict(summary)
            item["num_samples"] = counts_by_dataset[dataset_idx]
            summaries.append(item)
        return summaries

    def _build_region_keys(self):
        region_keys = []
        for dataset_idx, _sample_idx, station_index, _sample_start in self.index:
            dataset = self.task_datasets[int(dataset_idx)]
            row = dataset.station_records[int(station_index)]
            region_keys.append(canonical_region_key(row))
        return region_keys

    def __len__(self):
        return len(self.index)

    def _site_features(self, row, payload):
        try:
            lat = float(row.get("lat", np.nan))
        except Exception:
            lat = np.nan
        try:
            lon = float(row.get("lon", np.nan))
        except Exception:
            lon = np.nan
        try:
            cap_kw = float(payload.get("capacity_used_kw", np.nan))
        except Exception:
            cap_kw = np.nan
        try:
            timezone_offset_hours = float(payload.get("timezone_offset_hours", np.nan))
        except Exception:
            timezone_offset_hours = np.nan
        reliability = 0.0 if (not np.isfinite(lat) or not np.isfinite(lon)) else 1.0
        return np.array(
            [
                0.0 if not np.isfinite(lat) else lat,
                0.0 if not np.isfinite(lon) else lon,
                reliability,
                0.0 if not np.isfinite(cap_kw) else cap_kw,
                0.0 if not np.isfinite(timezone_offset_hours) else timezone_offset_hours,
            ],
            dtype=np.float32,
        )

    def _full_metadata(self, dataset, row, payload, s_begin, f_end, hist_native_resolution, fut_native_resolution):
        return {
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
            "timezone_offset_hours": payload.get("timezone_offset_hours", np.nan),
            "timezone_name": payload.get("timezone_name", ""),
        }

    def __getitem__(self, idx):
        dataset_idx, sample_idx, station_index, sample_start = self.index[int(idx)]
        dataset = self.task_datasets[dataset_idx]
        row = dataset.station_records[station_index]
        payload = dataset.station_payloads[station_index]

        s_begin = sample_start
        s_end = s_begin + dataset.seq_len
        f_begin = s_end
        f_end = f_begin + dataset.pred_len

        history_cov = payload["history_values"][s_begin:s_end]
        history_cov_mask = payload["history_mask"][s_begin:s_end]
        future_cov = payload["future_values"][f_begin:f_end]
        future_cov_mask = payload["future_mask"][f_begin:f_end]
        hist_native_resolution = ""
        fut_native_resolution = ""
        if self.include_native_covariates:
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
        else:
            hist_dim = int(payload["history_values"].shape[-1])
            fut_dim = int(payload["future_values"].shape[-1])
            time_dim = int(payload["time_values"].shape[-1])
            hist_native_values = np.zeros((0, hist_dim), dtype=np.float32)
            hist_native_mask = np.zeros((0, hist_dim), dtype=np.float32)
            hist_native_time = np.zeros((0, time_dim), dtype=np.float32)
            fut_native_values = np.zeros((0, fut_dim), dtype=np.float32)
            fut_native_mask = np.zeros((0, fut_dim), dtype=np.float32)
            fut_native_time = np.zeros((0, time_dim), dtype=np.float32)
        history_cov, history_cov_mask = _soft_mask_covariate_block(
            history_cov,
            history_cov_mask,
            min_valid_ratio=dataset.min_history_covariate_valid_ratio,
            min_std=dataset.min_history_covariate_std,
        )
        future_cov, future_cov_mask = _soft_mask_covariate_block(
            future_cov,
            future_cov_mask,
            min_valid_ratio=dataset.min_future_covariate_valid_ratio,
            min_std=dataset.min_future_covariate_std,
        )

        history_scaler = payload["history_scaler"]
        history_mean = np.asarray(getattr(history_scaler, "mean_", np.zeros(history_cov.shape[-1])), dtype=np.float32).reshape(-1)
        history_scale = np.asarray(getattr(history_scaler, "scale_", np.ones(history_cov.shape[-1])), dtype=np.float32).reshape(-1)
        if history_mean.size < history_cov.shape[-1]:
            history_mean = np.pad(history_mean, (0, history_cov.shape[-1] - history_mean.size))
        if history_scale.size < history_cov.shape[-1]:
            history_scale = np.pad(history_scale, (0, history_cov.shape[-1] - history_scale.size), constant_values=1.0)

        item = {
            "dataset_idx": dataset_idx,
            "sample_idx": sample_idx,
            "station_index": station_index,
            "sample_start": sample_start,
            "task_name": dataset.task_name,
            "station_id": row["station_dir"],
            "region_key": self.region_keys[int(idx)] if int(idx) < len(self.region_keys) else canonical_region_key(row),
            "sample_weight": float(self.sample_weights[int(idx)]) if int(idx) < len(self.sample_weights) else 1.0,
            "resolution": dataset.task_spec.resolution,
            "seq_len": dataset.seq_len,
            "label_len": dataset.label_len,
            "pred_len": dataset.pred_len,
            "past_target": payload["target_values"][s_begin:s_end],
            "past_observed_mask": payload["target_mask"][s_begin:s_end],
            "future_target": payload["target_values"][f_begin:f_end],
            "future_observed_mask": payload["target_mask"][f_begin:f_end],
            "historical_covariates": history_cov,
            "historical_covariates_mask": history_cov_mask,
            "historical_covariate_mean": history_mean[: history_cov.shape[-1]],
            "historical_covariate_scale": history_scale[: history_cov.shape[-1]],
            "historical_covariates_native": hist_native_values,
            "historical_covariates_native_mask": hist_native_mask,
            "future_covariates": future_cov,
            "future_covariates_mask": future_cov_mask,
            "future_covariates_native": fut_native_values,
            "future_covariates_native_mask": fut_native_mask,
            "past_time_features": payload["time_values"][s_begin:s_end],
            "future_time_features": payload["time_values"][f_begin:f_end],
            "historical_covariates_native_time_features": hist_native_time,
            "future_covariates_native_time_features": fut_native_time,
            "static_features": np.array(
                [
                    float(dataset.seq_len),
                    float(dataset.pred_len),
                    float(dataset.label_len),
                    float(dataset.task_spec.horizon_hours),
                ],
                dtype=np.float32,
            ),
            "site_features": self._site_features(row, payload),
        }
        # Normalize the task-specific calendar layout before the collator sees
        # it.  This matters for mixed 1h/sub-hourly indexed batches: generic
        # right-padding a 1h row would make the zero pad look like day-of-year
        # to the solar-regime weighting code.
        item["past_time_features"] = _align_indexed_time_features(
            item["past_time_features"],
            resolution=dataset.task_spec.resolution,
            target_dim=self.model_dims["past_time_dim"],
            schema=self.time_feature_schema,
        )
        item["future_time_features"] = _align_indexed_time_features(
            item["future_time_features"],
            resolution=dataset.task_spec.resolution,
            target_dim=self.model_dims["future_time_dim"],
            schema=self.time_feature_schema,
        )
        if self.metadata_mode == "full":
            item["metadata"] = self._full_metadata(
                dataset,
                row,
                payload,
                s_begin,
                f_end,
                hist_native_resolution,
                fut_native_resolution,
            )
        return item

    def _sample_history_length_for_batch(self, items):
        if self.split != "train" or not items:
            return None
        dataset_idx = int(items[0]["dataset_idx"])
        return self._batch_builder._sample_history_length(self.task_datasets[dataset_idx])

    def collate_fn(self, items):
        history_length = self._sample_history_length_for_batch(items)
        if self.split != "train" and items:
            dataset_idx = int(items[0]["dataset_idx"])
            history_length = self.task_datasets[dataset_idx].seq_len
        return self._collator(items, history_length=history_length)

    def logical_epoch_steps(self):
        if self.drop_last:
            return len(self.index) // self.batch_size
        return math.ceil(len(self.index) / self.batch_size)


def build_indexed_chronos_dataset(**kwargs):
    return IndexedChronosPVDataset(**kwargs)
