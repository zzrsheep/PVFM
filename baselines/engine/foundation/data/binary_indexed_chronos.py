import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from foundation.data.datasets import _soft_mask_covariate_block
from foundation.data.indexed_chronos import FastChronosCollator
from foundation.data.region_balanced_sampler import canonical_region_key, sample_weights_from_region_keys
from foundation.task_specs import resolve_task_lengths, resolve_task_spec
from utils.sample_index import SampleIndexFilter


# Keep these names in sync with the registry-routed multi-resolution dataset,
# but define them locally to avoid importing that module here (it imports this
# binary dataset while constructing its child views).
LEGACY_TIME_FEATURE_SCHEMA = "legacy_right_pad_v1"
CANONICAL_TIME_FEATURE_SCHEMA = "minute_hour_weekday_day_dayofyear_v1"


def _normalize_time_feature_schema(value):
    """Normalize persisted aliases to the versioned cache contract."""
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return None
    if text in {"legacy", LEGACY_TIME_FEATURE_SCHEMA}:
        return LEGACY_TIME_FEATURE_SCHEMA
    if text in {"canonical", CANONICAL_TIME_FEATURE_SCHEMA}:
        return CANONICAL_TIME_FEATURE_SCHEMA
    raise ValueError(f"unsupported binary-cache time_feature_schema={value!r}.")


def _align_binary_time_features(values, resolution, target_dim, schema):
    """Align cache time channels to the declared binary model contract."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"binary time features must be rank-2, got shape={array.shape}.")
    target_dim = int(target_dim)
    if target_dim <= 0 or array.shape[-1] > target_dim:
        raise ValueError(
            f"cannot align binary time features shape={array.shape} to target_dim={target_dim}."
        )
    schema = _normalize_time_feature_schema(schema) or LEGACY_TIME_FEATURE_SCHEMA
    if schema == CANONICAL_TIME_FEATURE_SCHEMA:
        if target_dim != 5:
            raise ValueError(
                f"{CANONICAL_TIME_FEATURE_SCHEMA} requires target_dim=5, got {target_dim}."
            )
        resolution = str(resolution or "1h").strip().lower()
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


class _ArrayScaler:
    """Small scaler object matching the attributes eval_one_task expects."""

    def __init__(self, mean=None, scale=None, var=None):
        mean = np.asarray(mean if mean is not None else [0.0], dtype=np.float64).reshape(-1)
        scale = np.asarray(scale if scale is not None else np.ones_like(mean), dtype=np.float64).reshape(-1)
        var = np.asarray(var if var is not None else scale * scale, dtype=np.float64).reshape(-1)
        self.mean_ = mean
        self.scale_ = scale
        self.var_ = var
        self.n_features_in_ = int(mean.shape[0])

    def inverse_transform(self, values):
        """Match the sklearn scaler interface used by prediction exporters."""
        values = np.asarray(values, dtype=np.float64)
        return values * self.scale_ + self.mean_


@dataclass
class _BinaryTaskDataset:
    task_name: str
    split: str
    station_records: list
    seq_len: int
    label_len: int
    pred_len: int
    task_spec: object
    min_history_covariate_valid_ratio: float
    min_future_covariate_valid_ratio: float
    min_history_covariate_std: float
    min_future_covariate_std: float


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_station_dir(value):
    return str(value or "").replace("\\", "/")


def _as_float_array(value, default):
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


class BinaryIndexedChronosPVDataset(Dataset):
    """Map-style PV dataset backed by station tensors and Parquet window indices."""

    yields_batches = False

    def __init__(
        self,
        binary_dataset_dir,
        manifest_path,
        task_names,
        split="train",
        regions=None,
        station_dirs=None,
        max_stations=0,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        fixed_seq_len=0,
        distributed_rank=0,
        distributed_world_size=1,
        train_sampler_mode="default",
        region_balance_alpha=0.5,
        region_balance_max_prob=0.0,
        region_balance_max_repeat_per_epoch=0.0,
        region_loss_alpha=0.0,
        eval_sample_stride=6,
        train_window_sample_stride=1,
        metadata_mode=None,
        include_native_covariates=False,
        min_history_covariate_valid_ratio=1.0,
        min_future_covariate_valid_ratio=1.0,
        min_history_covariate_std=0.0,
        min_future_covariate_std=0.0,
        window_index_root="",
        sample_index_csv="",
        time_feature_schema=None,
        **_unused,
    ):
        super().__init__()
        self.binary_dataset_dir = os.path.abspath(str(binary_dataset_dir or ""))
        if not self.binary_dataset_dir or not os.path.isdir(self.binary_dataset_dir):
            raise FileNotFoundError(f"binary_dataset_dir not found: {self.binary_dataset_dir}")
        self.manifest_path = os.path.abspath(str(manifest_path or ""))
        self.task_names = list(task_names or [])
        if not self.task_names:
            raise ValueError("BinaryIndexedChronosPVDataset requires at least one task name.")
        self.split = str(split or "train")
        self.regions = list(regions or [])
        self.station_dirs = [_normalize_station_dir(item) for item in (station_dirs or []) if str(item or "").strip()]
        self.max_stations = int(max_stations or 0)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.fixed_seq_len = int(fixed_seq_len or 0)
        self.distributed_rank = int(distributed_rank or 0)
        self.distributed_world_size = max(1, int(distributed_world_size or 1))
        self.train_sampler_mode = str(train_sampler_mode or "default").strip().lower()
        self.region_balance_alpha = max(0.0, float(region_balance_alpha or 0.0))
        self.region_balance_max_prob = max(0.0, float(region_balance_max_prob or 0.0))
        self.region_balance_max_repeat_per_epoch = max(0.0, float(region_balance_max_repeat_per_epoch or 0.0))
        self.region_loss_alpha = max(0.0, float(region_loss_alpha or 0.0))
        self.eval_sample_stride = int(6 if eval_sample_stride is None else eval_sample_stride)
        if self.eval_sample_stride < 1:
            raise ValueError("eval_sample_stride must be a positive integer")
        self.train_window_sample_stride = max(1, int(train_window_sample_stride or 1))
        self.metadata_mode = str(metadata_mode or ("minimal" if self.split == "train" else "full")).strip().lower()
        self.include_native_covariates = bool(include_native_covariates)
        self.window_index_root = os.path.abspath(str(window_index_root or "")) if window_index_root else ""
        self.native_fields_in_batch = self.include_native_covariates
        self.collator_mode_hint = "stack_or_fallback"
        self.min_history_covariate_valid_ratio = float(min_history_covariate_valid_ratio)
        self.min_future_covariate_valid_ratio = float(min_future_covariate_valid_ratio)
        self.min_history_covariate_std = float(min_history_covariate_std)
        self.min_future_covariate_std = float(min_future_covariate_std)
        self.sample_index_csv = str(sample_index_csv or "")
        self.sample_index_filter = SampleIndexFilter(self.sample_index_csv, split=self.split)
        if self.train_sampler_mode not in {"default", "region_balanced"}:
            raise ValueError("train_sampler_mode must be 'default' or 'region_balanced'.")
        if self.sample_index_filter.strict and len(self.task_names) != 1:
            raise ValueError("strict sample-index manifests require exactly one evaluation task")

        self.metadata = _read_json(os.path.join(self.binary_dataset_dir, "metadata.json"))
        # Newer cache metadata records the exact calendar-channel contract.
        # Older binary caches do not, so defer a width-based fallback until
        # model dimensions have been inferred below.
        metadata_schema = _normalize_time_feature_schema(
            self.metadata.get("time_feature_schema")
        )
        requested_schema = _normalize_time_feature_schema(time_feature_schema)
        # A checkpoint can predate cache metadata or intentionally preserve
        # the legacy right-padded contract.  When supplied, its explicit
        # schema is authoritative for emitted tensors; otherwise use the
        # cache declaration and finally infer from observed width below.
        self.time_feature_schema = requested_schema or metadata_schema
        self.stations = self._load_stations()
        self.station_records = self._build_station_records()
        self._station_meta_cache = {}
        self._station_array_cache = {}
        self._allowed_station_indices = self._resolve_allowed_station_indices()
        self.task_datasets = []
        self.index = []
        self._build_task_indices()
        self.sample_index_filter.assert_complete()
        if not self.index:
            raise ValueError(
                f"BinaryIndexedChronosPVDataset has no samples for split={self.split} "
                f"tasks={self.task_names} regions={self.regions} station_dirs={self.station_dirs[:3]}"
            )
        self.region_keys = self._build_region_keys()
        self.sample_weights = sample_weights_from_region_keys(self.region_keys, self.region_loss_alpha)
        self.task_summaries = self._build_task_summaries()
        self.model_dims = self._infer_model_dims()
        if self.time_feature_schema is None:
            time_width = max(
                int(self.model_dims.get("past_time_dim", 0) or 0),
                int(self.model_dims.get("future_time_dim", 0) or 0),
            )
            self.time_feature_schema = (
                CANONICAL_TIME_FEATURE_SCHEMA
                if time_width >= 5
                else LEGACY_TIME_FEATURE_SCHEMA
            )
        if self.time_feature_schema == CANONICAL_TIME_FEATURE_SCHEMA:
            # Canonical metadata may accompany a legacy 1h cache whose raw
            # arrays still have four channels.  ``__getitem__`` inserts the
            # minute-zero channel, so reserve width five here rather than
            # rejecting an otherwise valid cache.
            self.model_dims["past_time_dim"] = max(5, int(self.model_dims.get("past_time_dim", 0) or 0))
            self.model_dims["future_time_dim"] = max(5, int(self.model_dims.get("future_time_dim", 0) or 0))
            self.model_dims["context_input_dim"] = (
                int(self.model_dims.get("target_dim", 1))
                + int(self.model_dims.get("historical_covariate_dim", 0))
                + int(self.model_dims["past_time_dim"])
            )
            self.model_dims["future_input_dim"] = (
                int(self.model_dims.get("future_covariate_dim", 0))
                + int(self.model_dims["future_time_dim"])
            )
        self._collator = FastChronosCollator(
            model_dims=self.model_dims,
            split=self.split,
            metadata_mode=self.metadata_mode,
            include_native_covariates=self.include_native_covariates,
        )
        print(
            f"[BinaryIndexedChronosPVDataset] split={self.split} tasks={','.join(self.task_names)} "
            f"stations={len({int(item[2]) for item in self.index})} samples={len(self.index)} "
            f"dir={self.binary_dataset_dir}",
            flush=True,
        )

    def _load_stations(self):
        path = os.path.join(self.binary_dataset_dir, "stations.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"stations.parquet not found in binary dataset: {path}")
        frame = pd.read_parquet(path)
        frame = frame.fillna("")
        frame["station_dir"] = frame["station_dir"].map(_normalize_station_dir)
        return frame.sort_values("station_index").reset_index(drop=True)

    def _build_station_records(self):
        records = []
        for _, row in self.stations.iterrows():
            records.append({key: row[key] for key in self.stations.columns})
        return records

    def _manifest_station_dirs(self):
        if not self.manifest_path or not os.path.exists(self.manifest_path):
            return None
        frame = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        if "station_dir" not in frame.columns:
            return None
        frame["station_dir"] = frame["station_dir"].map(_normalize_station_dir)
        if self.station_dirs:
            frame = frame[frame["station_dir"].isin(set(self.station_dirs))]
        if self.regions:
            records = frame.to_dict(orient="records")
            keep = [
                canonical_region_key(row) in self.regions or str(row.get("region", "")) in self.regions
                for row in records
            ]
            frame = frame[keep]
        if self.max_stations > 0:
            frame = frame.head(self.max_stations)
        return set(frame["station_dir"].tolist())

    def _resolve_allowed_station_indices(self):
        allowed_dirs = self._manifest_station_dirs()
        if self.station_dirs:
            station_filter = set(self.station_dirs)
            allowed_dirs = station_filter if allowed_dirs is None else allowed_dirs & station_filter
        allowed = []
        has_filter = allowed_dirs is not None or bool(self.station_dirs) or bool(self.regions) or self.max_stations > 0
        for idx, row in self.stations.iterrows():
            station_dir = _normalize_station_dir(row.get("station_dir", ""))
            if allowed_dirs is not None and station_dir not in allowed_dirs:
                continue
            if self.regions:
                region_key = canonical_region_key(row.to_dict())
                if region_key not in self.regions and row.get("region", "") not in self.regions:
                    continue
            allowed.append(int(idx))
        if self.max_stations > 0:
            allowed = allowed[: self.max_stations]
        if has_filter:
            return set(allowed)
        return set(int(idx) for idx in self.stations.index)

    def _task_lengths(self, task_name):
        spec = resolve_task_spec(task_name)
        if spec is None:
            raise ValueError(f"Unknown task_name in binary dataset: {task_name}")
        seq_len, label_len, pred_len = resolve_task_lengths(spec)
        return spec, int(seq_len), int(label_len), int(pred_len)

    def _task_resolution(self, task_name):
        spec = resolve_task_spec(task_name)
        if spec is None:
            raise ValueError(f"Unknown task_name in binary dataset: {task_name}")
        return str(spec.resolution)

    def _window_index_path(self, task_name):
        resolution = self._task_resolution(task_name)
        if self.window_index_root:
            return os.path.join(self.window_index_root, "window_index", resolution, task_name, f"{self.split}.parquet")
        return os.path.join(self.binary_dataset_dir, "window_index", resolution, task_name, f"{self.split}.parquet")

    def _load_window_index(self, task_name):
        """Read immutable base index plus an optional lightweight dual-view override."""
        base_path = os.path.join(
            self.binary_dataset_dir, "window_index", self._task_resolution(task_name), task_name, f"{self.split}.parquet"
        )
        if not self.window_index_root:
            return pd.read_parquet(base_path, columns=["station_index", "sample_start"])
        extension_path = os.path.join(
            self.window_index_root, "window_index", self._task_resolution(task_name), task_name, f"{self.split}.parquet"
        )
        has_extension = os.path.exists(extension_path)

        overlay_root = self.window_index_root
        audit_path = os.path.join(overlay_root, "time_boundary_audit.json")
        if not os.path.exists(audit_path):
            raise FileNotFoundError(f"multires overlay audit missing: {audit_path}")
        audit = _read_json(audit_path)
        base_overlay_dir = str(audit.get("base_overlay_dir", "")).strip()
        if base_overlay_dir:
            if not os.path.isabs(base_overlay_dir):
                base_overlay_dir = os.path.join(overlay_root, base_overlay_dir)
            overlay_root = os.path.abspath(base_overlay_dir)
            audit_path = os.path.join(overlay_root, "time_boundary_audit.json")
            if not os.path.exists(audit_path):
                raise FileNotFoundError(f"task-coverage base overlay audit missing: {audit_path}")
            audit = _read_json(audit_path)
        cache_name = None
        for name, cache_hash in (audit.get("cache_metadata_hashes") or {}).items():
            # Metadata hashes make a stale overlay fail closed rather than silently mixing versions.
            import hashlib
            digest = hashlib.sha256(open(os.path.join(self.binary_dataset_dir, "metadata.json"), "rb").read()).hexdigest()
            if digest == cache_hash:
                cache_name = name
                break
        if cache_name is None:
            raise ValueError("Overlay registry does not match this binary cache metadata.")
        # A task-coverage extension owns complete indices only for newly added
        # tasks. The metadata hash check above prevents stale cache mixing.
        if has_extension:
            return pd.read_parquet(extension_path, columns=["station_index", "sample_start"])
        dual_indices = set(int(value) for value in (audit.get("dual_station_indices_by_cache") or {}).get(cache_name, []))
        override_path = os.path.join(
            overlay_root, "window_overrides", self._task_resolution(task_name), task_name, f"{self.split}.parquet"
        )
        if not dual_indices:
            return pd.read_parquet(base_path, columns=["station_index", "sample_start"])
        base = pd.read_parquet(base_path, columns=["station_index", "sample_start"])
        base = base[~base["station_index"].isin(dual_indices)]
        if os.path.exists(override_path):
            override = pd.read_parquet(override_path, columns=["station_index", "sample_start"])
            return pd.concat([base, override], ignore_index=True)
        return base.reset_index(drop=True)

    def _build_task_indices(self):
        for dataset_idx, task_name in enumerate(self.task_names):
            spec, seq_len, label_len, pred_len = self._task_lengths(task_name)
            path = self._window_index_path(task_name)
            if not self.window_index_root and not os.path.exists(path):
                raise FileNotFoundError(f"binary window index not found: {path}")
            if self.sample_index_filter.strict:
                # A virtual boundary changes the chronological split, so the
                # old immutable test index is no longer authoritative. Rebuild
                # exact starts from canonical timestamps in frozen tensors.
                frame = self._strict_index_from_manifest(spec.resolution, seq_len, pred_len)
            else:
                frame = self._load_window_index(task_name)
                if self._allowed_station_indices:
                    frame = frame[frame["station_index"].isin(self._allowed_station_indices)]
            if self.sample_index_filter.enabled and not self.sample_index_filter.strict:
                frame = self._filter_by_sample_index(frame, spec.resolution, seq_len, pred_len)
            if self.split == "train" and self.train_window_sample_stride > 1:
                # Candidate-window thinning only. Tensor values remain exact
                # cache slices and are never transformed or aggregated.
                frame = frame.iloc[:: self.train_window_sample_stride]
            if self.split != "train" and self.eval_sample_stride > 1 and not self.sample_index_filter.strict:
                # Strict manifests are exact origin whitelists. Applying a
                # second positional stride would silently drop requested
                # origins, so the manifest remains authoritative.
                frame = frame.iloc[:: self.eval_sample_stride]
            if self.distributed_world_size > 1:
                frame = frame.iloc[self.distributed_rank :: self.distributed_world_size]
            frame = frame.reset_index(drop=True)
            task_dataset = _BinaryTaskDataset(
                task_name=task_name,
                split=self.split,
                station_records=self.station_records,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                task_spec=spec,
                min_history_covariate_valid_ratio=self.min_history_covariate_valid_ratio,
                min_future_covariate_valid_ratio=self.min_future_covariate_valid_ratio,
                min_history_covariate_std=self.min_history_covariate_std,
                min_future_covariate_std=self.min_future_covariate_std,
            )
            self.task_datasets.append(task_dataset)
            station_indices = frame["station_index"].to_numpy(dtype=np.int32, copy=False)
            sample_starts = frame["sample_start"].to_numpy(dtype=np.int64, copy=False)
            for sample_idx, (station_index, sample_start) in enumerate(zip(station_indices, sample_starts)):
                self.index.append((dataset_idx, int(sample_idx), int(station_index), int(sample_start)))

    def _strict_index_from_manifest(self, resolution, seq_len, pred_len):
        expected_by_station = {}
        for key in self.sample_index_filter.expected_keys:
            expected_by_station.setdefault(key[0], []).append(key)

        rows = []
        for station_index in sorted(self._allowed_station_indices):
            station_index = int(station_index)
            station_row = self.stations.iloc[station_index]
            station_dir = _normalize_station_dir(station_row.get("station_dir", ""))
            station_id = str(station_row.get("station_id", "") or "").strip()
            station_keys = [f"dir:{station_dir}"] if station_dir else []
            if station_id:
                station_keys.append(f"id:{station_id}")
            expected = []
            for station_key in station_keys:
                expected.extend(expected_by_station.get(station_key, []))
            if not expected:
                continue

            arrays = self._station_arrays(str(resolution), station_index)
            timestamps = np.asarray(arrays["timestamps"])
            target_mask = np.asarray(arrays["target_mask"])
            history_mask = np.asarray(arrays["history_covariate_mask"])
            future_mask = np.asarray(arrays["future_covariate_mask"])
            for _station_key, input_end_time, future_start_time in sorted(expected):
                future_timestamp = np.datetime64(future_start_time.replace(" ", "T"), "ns")
                future_index = int(np.searchsorted(timestamps, future_timestamp))
                sample_start = future_index - int(seq_len)
                target_end = future_index + int(pred_len)
                if (
                    future_index >= len(timestamps)
                    or timestamps[future_index] != future_timestamp
                    or sample_start < 0
                    or target_end > len(timestamps)
                ):
                    continue
                actual_input_end = np.datetime_as_string(timestamps[future_index - 1], unit="s").replace("T", " ")
                actual_future_start = np.datetime_as_string(timestamps[future_index], unit="s").replace("T", " ")
                if input_end_time and actual_input_end != input_end_time:
                    continue
                if not (
                    np.all(target_mask[sample_start:target_end] > 0)
                    and np.all(history_mask[sample_start:future_index] > 0)
                    and np.all(future_mask[future_index:target_end] > 0)
                ):
                    continue
                if not self.sample_index_filter.accepts_normalized(
                    station_dir=station_dir,
                    station_id=station_id,
                    input_end_time=actual_input_end,
                    future_start_time=actual_future_start,
                ):
                    continue
                rows.append({"station_index": station_index, "sample_start": sample_start})
        return pd.DataFrame(rows, columns=["station_index", "sample_start"])

    def _filter_by_sample_index(self, frame, resolution, seq_len, pred_len):
        """Filter immutable cache indices by exact canonical timestamps."""
        kept_parts = []
        for station_index, group in frame.groupby("station_index", sort=False):
            station_index = int(station_index)
            starts = group["sample_start"].to_numpy(dtype=np.int64, copy=False)
            arrays = self._station_arrays(str(resolution), station_index)
            timestamps = arrays["timestamps"]
            in_bounds = starts + int(seq_len) + int(pred_len) <= len(timestamps)
            if not bool(np.all(in_bounds)):
                starts = starts[in_bounds]
                group = group.iloc[np.flatnonzero(in_bounds)]
            if not len(starts):
                continue

            input_end = np.datetime_as_string(timestamps[starts + int(seq_len) - 1], unit="s")
            future_start = np.datetime_as_string(timestamps[starts + int(seq_len)], unit="s")
            station_row = self.stations.iloc[station_index]
            station_dir = _normalize_station_dir(station_row.get("station_dir", ""))
            station_id = str(station_row.get("station_id", "") or "").strip()
            accepted = [
                self.sample_index_filter.accepts_normalized(
                    station_dir=station_dir,
                    station_id=station_id,
                    input_end_time=str(input_time).replace("T", " "),
                    future_start_time=str(future_time).replace("T", " "),
                )
                for input_time, future_time in zip(input_end, future_start)
            ]
            if any(accepted):
                kept_parts.append(group.iloc[np.flatnonzero(accepted)])
        if not kept_parts:
            return frame.iloc[0:0].copy()
        return pd.concat(kept_parts, ignore_index=True)

    def _build_region_keys(self):
        keys = []
        for _dataset_idx, _sample_idx, station_index, _sample_start in self.index:
            keys.append(canonical_region_key(self.station_records[int(station_index)]))
        return keys

    def _build_task_summaries(self):
        counts = [0 for _ in self.task_datasets]
        station_sets = [set() for _ in self.task_datasets]
        for dataset_idx, _sample_idx, station_index, _sample_start in self.index:
            counts[int(dataset_idx)] += 1
            station_sets[int(dataset_idx)].add(int(station_index))
        summaries = []
        for dataset_idx, task_dataset in enumerate(self.task_datasets):
            summaries.append(
                {
                    "task_name": task_dataset.task_name,
                    "resolution": task_dataset.task_spec.resolution,
                    "seq_len": task_dataset.seq_len,
                    "label_len": task_dataset.label_len,
                    "pred_len": task_dataset.pred_len,
                    "max_context_hours": task_dataset.task_spec.context_hours,
                    "history_choices_hours": tuple(task_dataset.task_spec.history_choices_hours),
                    "compatible_granularities": tuple(task_dataset.task_spec.compatible_granularities),
                    "horizon_hours": task_dataset.task_spec.horizon_hours,
                    "num_samples": counts[dataset_idx],
                    "num_stations": len(station_sets[dataset_idx]),
                }
            )
        return summaries

    def _infer_model_dims(self):
        # A binary cache may expose multiple task resolutions.  Infer the
        # global dimensions rather than using whichever task happens to be the
        # first index row; otherwise a 1h-first/15min-second dataset would
        # truncate the latter's five-channel time features in the collator.
        first_station_by_dataset = {}
        for dataset_idx, _sample_idx, station_index, _sample_start in self.index:
            first_station_by_dataset.setdefault(int(dataset_idx), int(station_index))
        if not first_station_by_dataset:
            raise ValueError("Cannot infer binary model dimensions from an empty index.")
        hist_dim = fut_dim = time_dim = 0
        for dataset_idx, station_index in sorted(first_station_by_dataset.items()):
            resolution = self.task_datasets[dataset_idx].task_spec.resolution
            arrays = self._station_arrays(resolution, station_index)
            hist_dim = max(hist_dim, int(arrays["history_covariates"].shape[-1]))
            fut_dim = max(fut_dim, int(arrays["future_covariates"].shape[-1]))
            time_dim = max(time_dim, int(arrays["time_features"].shape[-1]))
        return {
            "target_dim": 1,
            "historical_covariate_dim": hist_dim,
            "future_covariate_dim": fut_dim,
            "past_time_dim": time_dim,
            "future_time_dim": time_dim,
            "static_dim": 4,
            "context_input_dim": 1 + hist_dim + time_dim,
            "future_input_dim": fut_dim + time_dim,
            "output_dim": 1,
            "num_horizons": 0,
        }

    def _station_key(self, station_index):
        return str(self.stations.iloc[int(station_index)]["station_key"])

    def _station_dir(self, station_index):
        return _normalize_station_dir(self.stations.iloc[int(station_index)]["station_dir"])

    def _station_view_dir(self, resolution, station_index):
        return os.path.join(self.binary_dataset_dir, "views", str(resolution), self._station_key(station_index))

    def _station_arrays(self, resolution, station_index):
        resolution = str(resolution)
        station_index = int(station_index)
        cache_key = (resolution, station_index)
        cached = self._station_array_cache.get(cache_key)
        if cached is not None:
            return cached
        view_dir = self._station_view_dir(resolution, station_index)
        arrays = {
            "target": np.load(os.path.join(view_dir, "target.npy"), mmap_mode="r"),
            "target_mask": np.load(os.path.join(view_dir, "target_mask.npy"), mmap_mode="r"),
            "history_covariates": np.load(os.path.join(view_dir, "history_covariates.npy"), mmap_mode="r"),
            "history_covariate_mask": np.load(os.path.join(view_dir, "history_covariate_mask.npy"), mmap_mode="r"),
            "future_covariates": np.load(os.path.join(view_dir, "future_covariates.npy"), mmap_mode="r"),
            "future_covariate_mask": np.load(os.path.join(view_dir, "future_covariate_mask.npy"), mmap_mode="r"),
            "time_features": np.load(os.path.join(view_dir, "time_features.npy"), mmap_mode="r"),
            "timestamps": np.load(os.path.join(view_dir, "timestamps.npy"), mmap_mode="r"),
            "site_features": np.load(os.path.join(view_dir, "site_features.npy"), mmap_mode="r"),
        }
        self._station_array_cache[cache_key] = arrays
        return arrays

    def _station_meta(self, resolution, station_index):
        resolution = str(resolution)
        station_index = int(station_index)
        cache_key = (resolution, station_index)
        cached = self._station_meta_cache.get(cache_key)
        if cached is not None:
            return cached
        path = os.path.join(self._station_view_dir(resolution, station_index), "metadata.json")
        meta = _read_json(path)
        self._station_meta_cache[cache_key] = meta
        return meta

    def _scaler(self, station_meta, name):
        state = station_meta.get(name, {}) or {}
        return _ArrayScaler(
            mean=_as_float_array(state.get("mean"), [0.0]),
            scale=_as_float_array(state.get("scale"), [1.0]),
            var=_as_float_array(state.get("var"), [1.0]),
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx, station_index, sample_start = self.index[int(idx)]
        task_dataset = self.task_datasets[int(dataset_idx)]
        station_index = int(station_index)
        sample_start = int(sample_start)
        resolution = str(task_dataset.task_spec.resolution)
        s_begin = sample_start
        s_end = s_begin + int(task_dataset.seq_len)
        f_begin = s_end
        f_end = f_begin + int(task_dataset.pred_len)

        arrays = self._station_arrays(resolution, station_index)
        station_meta = self._station_meta(resolution, station_index)
        history_cov = np.asarray(arrays["history_covariates"][s_begin:s_end], dtype=np.float32)
        history_cov_mask = np.asarray(arrays["history_covariate_mask"][s_begin:s_end], dtype=np.float32)
        future_cov = np.asarray(arrays["future_covariates"][f_begin:f_end], dtype=np.float32)
        future_cov_mask = np.asarray(arrays["future_covariate_mask"][f_begin:f_end], dtype=np.float32)
        history_cov, history_cov_mask = _soft_mask_covariate_block(
            history_cov,
            history_cov_mask,
            min_valid_ratio=task_dataset.min_history_covariate_valid_ratio,
            min_std=task_dataset.min_history_covariate_std,
        )
        future_cov, future_cov_mask = _soft_mask_covariate_block(
            future_cov,
            future_cov_mask,
            min_valid_ratio=task_dataset.min_future_covariate_valid_ratio,
            min_std=task_dataset.min_future_covariate_std,
        )

        hist_dim = int(arrays["history_covariates"].shape[-1])
        fut_dim = int(arrays["future_covariates"].shape[-1])
        time_dim = int(arrays["time_features"].shape[-1])
        # Keep the station-level standardization statistics in the lightweight
        # batch contract.  V7 uses them only to reconstruct physical GHI for a
        # history-only auxiliary label; they are never passed into the forecast
        # model as predictive features.
        history_scaler = self._scaler(station_meta, "history_scaler")
        history_mean = np.asarray(history_scaler.mean_, dtype=np.float32).reshape(-1)
        history_scale = np.asarray(history_scaler.scale_, dtype=np.float32).reshape(-1)
        if history_mean.size < hist_dim:
            history_mean = np.pad(history_mean, (0, hist_dim - history_mean.size))
        if history_scale.size < hist_dim:
            history_scale = np.pad(history_scale, (0, hist_dim - history_scale.size), constant_values=1.0)
        item = {
            "dataset_idx": int(dataset_idx),
            "sample_idx": int(sample_idx),
            "station_index": station_index,
            "sample_start": sample_start,
            "task_name": task_dataset.task_name,
            "station_id": self._station_dir(station_index),
            "region_key": self.region_keys[int(idx)] if int(idx) < len(self.region_keys) else canonical_region_key(self.station_records[station_index]),
            "sample_weight": float(self.sample_weights[int(idx)]) if int(idx) < len(self.sample_weights) else 1.0,
            "resolution": task_dataset.task_spec.resolution,
            "seq_len": int(task_dataset.seq_len),
            "label_len": int(task_dataset.label_len),
            "pred_len": int(task_dataset.pred_len),
            "past_target": np.asarray(arrays["target"][s_begin:s_end], dtype=np.float32),
            "past_observed_mask": np.asarray(arrays["target_mask"][s_begin:s_end], dtype=np.float32),
            "future_target": np.asarray(arrays["target"][f_begin:f_end], dtype=np.float32),
            "future_observed_mask": np.asarray(arrays["target_mask"][f_begin:f_end], dtype=np.float32),
            "historical_covariates": history_cov,
            "historical_covariates_mask": history_cov_mask,
            "historical_covariate_mean": history_mean[:hist_dim],
            "historical_covariate_scale": history_scale[:hist_dim],
            "historical_covariates_native": np.zeros((0, hist_dim), dtype=np.float32),
            "historical_covariates_native_mask": np.zeros((0, hist_dim), dtype=np.float32),
            "future_covariates": future_cov,
            "future_covariates_mask": future_cov_mask,
            "future_covariates_native": np.zeros((0, fut_dim), dtype=np.float32),
            "future_covariates_native_mask": np.zeros((0, fut_dim), dtype=np.float32),
            "past_time_features": np.asarray(arrays["time_features"][s_begin:s_end], dtype=np.float32),
            "future_time_features": np.asarray(arrays["time_features"][f_begin:f_end], dtype=np.float32),
            "historical_covariates_native_time_features": np.zeros((0, time_dim), dtype=np.float32),
            "future_covariates_native_time_features": np.zeros((0, time_dim), dtype=np.float32),
            "static_features": np.array(
                [
                    float(task_dataset.seq_len),
                    float(task_dataset.pred_len),
                    float(task_dataset.label_len),
                    float(task_dataset.task_spec.horizon_hours),
                ],
                dtype=np.float32,
            ),
            "site_features": np.asarray(arrays["site_features"], dtype=np.float32).reshape(-1)[:5],
        }
        # Align mixed-resolution rows before the generic collator.  Its
        # fallback path right-pads dimensions, which would shift the calendar
        # semantics of a 1h row in a canonical 1h+15min batch.
        item["past_time_features"] = _align_binary_time_features(
            item["past_time_features"],
            resolution=resolution,
            target_dim=self.model_dims["past_time_dim"],
            schema=self.time_feature_schema,
        )
        item["future_time_features"] = _align_binary_time_features(
            item["future_time_features"],
            resolution=resolution,
            target_dim=self.model_dims["future_time_dim"],
            schema=self.time_feature_schema,
        )
        if self.metadata_mode == "full":
            target_scaler = self._scaler(station_meta, "target_scaler")
            future_scaler = self._scaler(station_meta, "future_scaler")
            item["metadata"] = {
                "station_record": self.station_records[station_index],
                "timestamps": np.asarray(arrays["timestamps"][s_begin:f_end]),
                "target_scaler": target_scaler,
                "history_scaler": history_scaler,
                "future_scaler": future_scaler,
                "y_scaler": target_scaler,
                "feature_cols": tuple(station_meta.get("history_covariate_cols", [])),
                "history_covariate_cols": tuple(station_meta.get("history_covariate_cols", [])),
                "future_covariate_cols": tuple(station_meta.get("future_covariate_cols", [])),
                "data_file_name": station_meta.get("data_file_name", ""),
                "history_covariate_file_name": station_meta.get("history_covariate_file_name", ""),
                "future_covariate_file_name": station_meta.get("future_covariate_file_name", ""),
                "target_transform": station_meta.get("target_transform", ""),
                "historical_covariates_native_resolution": "",
                "future_covariates_native_resolution": "",
                "target_normalization_mode": station_meta.get("target_normalization_mode", "none"),
                "target_standardization": station_meta.get("target_standardization", "standard"),
                "power_semantics": station_meta.get("power_semantics", ""),
                "power_semantics_note": station_meta.get("power_semantics_note", ""),
                "power_unit_scale_to_kw": station_meta.get("power_unit_scale_to_kw", 1.0),
                "cap_meta_kw": station_meta.get("cap_meta_kw", np.nan),
                "cap_meta_field": station_meta.get("cap_meta_field", ""),
                "cap_meta_source": station_meta.get("cap_meta_source", ""),
                "ratio_p99_5_to_cap": station_meta.get("ratio_p99_5_to_cap", np.nan),
                "cap_status": station_meta.get("cap_status", ""),
                "cap_note": station_meta.get("cap_note", ""),
                "capacity_used_kw": station_meta.get("capacity_used_kw", np.nan),
                "capacity_used_source": station_meta.get("capacity_used_source", ""),
                "target_to_power_scale": station_meta.get("target_to_power_scale", 1.0),
                "timezone_offset_hours": station_meta.get("timezone_offset_hours", np.nan),
                "timezone_name": station_meta.get("timezone_name", ""),
                "binary_dataset_dir": self.binary_dataset_dir,
            }
        return item

    def _sample_history_length_for_batch(self, items):
        if self.split != "train" or not items:
            return None
        if self.fixed_seq_len > 0:
            return int(items[0]["seq_len"])
        return None

    def collate_fn(self, items):
        history_length = self._sample_history_length_for_batch(items)
        if self.split != "train" and items:
            history_length = int(items[0]["seq_len"])
        return self._collator(items, history_length=history_length)

    def logical_epoch_steps(self):
        if self.drop_last:
            return len(self.index) // self.batch_size
        return int(math.ceil(float(len(self.index)) / float(self.batch_size)))


def build_binary_indexed_chronos_dataset(**kwargs):
    return BinaryIndexedChronosPVDataset(**kwargs)
