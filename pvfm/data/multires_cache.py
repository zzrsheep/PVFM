from __future__ import annotations

import json

import hashlib

import math

import os

from pathlib import Path

import numpy as np

from torch.utils.data import Dataset

from pvfm.data.binary_cache import BinaryCacheDataset

from pvfm.data.collate import WindowCollator

from pvfm.data.regions import sample_weights_from_region_keys

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
    raise ValueError(f"unsupported multires time_feature_schema={value!r}.")


def align_multires_time_features(
    values, resolution, target_dim, schema=LEGACY_TIME_FEATURE_SCHEMA
):
    """Align cache-specific calendar channels without modifying cache arrays.

    The frozen 1h cache stores ``[hour, weekday, day, dayofyear]`` while the
    native 15min cache stores ``[minute, hour, weekday, day, dayofyear]``.
    The canonical opt-in contract inserts the normalized minute-zero value at
    the front of 1h rows.  Legacy registries retain the historical right-pad
    behavior for checkpoint reproducibility.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"time features must be rank-2, got shape={array.shape}.")
    target_dim = int(target_dim)
    if target_dim <= 0 or array.shape[-1] > target_dim:
        raise ValueError(
            f"cannot align time features shape={array.shape} to target_dim={target_dim}."
        )
    schema = _normalize_time_feature_schema(schema) or LEGACY_TIME_FEATURE_SCHEMA
    if schema not in SUPPORTED_TIME_FEATURE_SCHEMAS:
        raise ValueError(f"unsupported multires time_feature_schema={schema!r}.")
    if schema == CANONICAL_TIME_FEATURE_SCHEMA:
        if target_dim != 5:
            raise ValueError(
                f"{CANONICAL_TIME_FEATURE_SCHEMA} requires target_dim=5, got {target_dim}."
            )
        if str(resolution) == "1h" and array.shape[-1] == 4:
            aligned = np.empty((array.shape[0], 5), dtype=np.float32)
            aligned[:, 0] = -0.5
            aligned[:, 1:] = array
            return aligned
        if array.shape[-1] != 5:
            raise ValueError(
                f"{CANONICAL_TIME_FEATURE_SCHEMA} expects 1h/4D or canonical/5D rows, "
                f"got resolution={resolution!r} shape={array.shape}."
            )
        return array
    if array.shape[-1] == target_dim:
        return array
    padded = np.zeros((array.shape[0], target_dim), dtype=np.float32)
    padded[:, : array.shape[-1]] = array
    return padded


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(value):
    return os.path.abspath(str(value))


def _validate_cache_identity(cache_name, config):
    """Validate optional immutable cache identity fields without changing old registries."""
    cache_dir = _resolve_path(config["binary_dataset_dir"])
    metadata_path = os.path.join(cache_dir, "metadata.json")
    expected_hash = str(config.get("metadata_sha256", "")).strip()
    expected_version = str(config.get("metadata_dataset_version", "")).strip()
    if not expected_hash and not expected_version:
        return
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"registry cache metadata not found for {cache_name}: {metadata_path}"
        )
    if expected_hash:
        digest = hashlib.sha256()
        with open(metadata_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(
                f"registry cache metadata hash mismatch for {cache_name}: {metadata_path}"
            )
    if expected_version:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("dataset_version") != expected_version:
            raise ValueError(
                f"registry cache dataset version mismatch for {cache_name}: "
                f"{metadata.get('dataset_version')!r} != {expected_version!r}"
            )


def _registry_tensor_dims(registry):
    """Read only array headers so a single-task eval keeps joint-model dims.

    A multires checkpoint is trained with the union of all routed cache
    feature dimensions.  During per-task evaluation we may instantiate only a
    1h child, whose legacy cache has four calendar channels.  Looking at every
    cache's first station header preserves the joint 1h+15min input contract
    without constructing inactive child datasets or copying tensors.
    """
    dims = {
        "historical_covariate_dim": 0,
        "future_covariate_dim": 0,
        "past_time_dim": 0,
        "future_time_dim": 0,
    }
    seen_caches = set()
    for config in (registry.get("caches") or {}).values():
        cache_dir = Path(_resolve_path(config["binary_dataset_dir"]))
        if cache_dir in seen_caches:
            continue
        seen_caches.add(cache_dir)
        resolution = str(config.get("resolution", ""))
        view_dir = cache_dir / "views" / resolution
        if not view_dir.is_dir():
            raise FileNotFoundError(f"registry cache view not found: {view_dir}")
        station_dirs = sorted(path for path in view_dir.iterdir() if path.is_dir())
        if not station_dirs:
            raise ValueError(f"registry cache has no station tensors: {view_dir}")
        station_dir = station_dirs[0]
        dims["historical_covariate_dim"] = max(
            dims["historical_covariate_dim"],
            int(
                np.load(station_dir / "history_covariates.npy", mmap_mode="r").shape[-1]
            ),
        )
        dims["future_covariate_dim"] = max(
            dims["future_covariate_dim"],
            int(
                np.load(station_dir / "future_covariates.npy", mmap_mode="r").shape[-1]
            ),
        )
        time_dim = int(
            np.load(station_dir / "time_features.npy", mmap_mode="r").shape[-1]
        )
        dims["past_time_dim"] = max(dims["past_time_dim"], time_dim)
        dims["future_time_dim"] = max(dims["future_time_dim"], time_dim)
    return dims


class MultiResolutionDataset(Dataset):
    """Compose task-specific binary caches without copying their arrays.

    Child datasets keep the established binary tensor contract.  This wrapper
    only owns a lightweight global index of ``(child_id, child_sample_id)``.
    """

    yields_batches = False

    def __init__(
        self,
        binary_cache_registry,
        binary_index_overlay_dir,
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
        time_feature_schema=None,
        **_unused,
    ):
        self.binary_cache_registry = _resolve_path(binary_cache_registry)
        self.binary_index_overlay_dir = (
            _resolve_path(binary_index_overlay_dir)
            if str(binary_index_overlay_dir or "").strip()
            else ""
        )
        if not os.path.exists(self.binary_cache_registry):
            raise FileNotFoundError(
                f"binary cache registry not found: {self.binary_cache_registry}"
            )
        if self.binary_index_overlay_dir and not os.path.exists(
            os.path.join(self.binary_index_overlay_dir, "time_boundary_audit.json")
        ):
            raise FileNotFoundError(
                f"multires overlay missing time_boundary_audit.json: {self.binary_index_overlay_dir}"
            )
        self.registry = _read_json(self.binary_cache_registry)
        registry_schema = _normalize_time_feature_schema(
            self.registry.get("time_feature_schema", LEGACY_TIME_FEATURE_SCHEMA)
        )
        requested_schema = _normalize_time_feature_schema(time_feature_schema)
        # Checkpoint metadata can explicitly preserve the legacy layout when
        # evaluating a registry created before the canonical contract.  New
        # training/evaluation calls omit this override and use the registry.
        self.time_feature_schema = (
            requested_schema or registry_schema or LEGACY_TIME_FEATURE_SCHEMA
        )
        if self.time_feature_schema not in SUPPORTED_TIME_FEATURE_SCHEMAS:
            raise ValueError(
                f"unsupported registry time_feature_schema={self.time_feature_schema!r}."
            )
        self.cache_sampling_mode = (
            str(self.registry.get("cache_sampling_mode", "weighted")).strip().lower()
        )
        if self.cache_sampling_mode not in {"weighted", "pooled"}:
            raise ValueError(
                "cache_sampling_mode must be either 'weighted' or 'pooled', "
                f"got {self.cache_sampling_mode!r}."
            )
        self.overlay_audit = (
            _read_json(
                os.path.join(self.binary_index_overlay_dir, "time_boundary_audit.json")
            )
            if self.binary_index_overlay_dir
            else {}
        )
        self.split = str(split or "train")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.fixed_seq_len = int(fixed_seq_len or 0)
        self.metadata_mode = str(
            metadata_mode or ("minimal" if self.split == "train" else "full")
        )
        self.include_native_covariates = bool(include_native_covariates)
        self.native_fields_in_batch = self.include_native_covariates
        self.collator_mode_hint = "stack_or_fallback"
        self.task_names = list(task_names or [])
        if not self.task_names:
            raise ValueError("MultiResolutionDataset requires task_names.")
        self.task_datasets = []
        self.children = []
        self.index = []
        self._child_data_versions = []
        self._child_cache_names = []
        for task_name in self.task_names:
            task_config = (self.registry.get("tasks") or {}).get(task_name)
            if not task_config:
                raise ValueError(f"Task {task_name!r} is absent from cache registry.")
            cache_names = task_config.get("caches")
            if cache_names is None:
                cache_names = task_config.get("cache")
            if isinstance(cache_names, str):
                cache_names = [cache_names]
            cache_names = [
                str(name).strip() for name in (cache_names or []) if str(name).strip()
            ]
            if not cache_names:
                raise ValueError(f"Task {task_name!r} does not reference any cache.")
            for cache_name in cache_names:
                cache_config = (self.registry.get("caches") or {}).get(cache_name)
                if not cache_config:
                    raise ValueError(
                        f"Task {task_name!r} references unknown cache {cache_name!r}."
                    )
                _validate_cache_identity(cache_name, cache_config)
                child = BinaryCacheDataset(
                    binary_dataset_dir=cache_config["binary_dataset_dir"],
                    window_index_root=self.binary_index_overlay_dir,
                    manifest_path=manifest_path,
                    task_names=[task_name],
                    split=self.split,
                    regions=regions,
                    station_dirs=station_dirs,
                    max_stations=max_stations,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    fixed_seq_len=fixed_seq_len,
                    # The parent partitions global batches, so children must not shard again.
                    distributed_rank=0,
                    distributed_world_size=1,
                    train_sampler_mode=train_sampler_mode,
                    region_balance_alpha=region_balance_alpha,
                    region_balance_max_prob=region_balance_max_prob,
                    region_balance_max_repeat_per_epoch=region_balance_max_repeat_per_epoch,
                    region_loss_alpha=0.0,
                    eval_sample_stride=eval_sample_stride,
                    train_window_sample_stride=train_window_sample_stride,
                    metadata_mode=self.metadata_mode,
                    include_native_covariates=include_native_covariates,
                    min_history_covariate_valid_ratio=min_history_covariate_valid_ratio,
                    min_future_covariate_valid_ratio=min_future_covariate_valid_ratio,
                    min_history_covariate_std=min_history_covariate_std,
                    min_future_covariate_std=min_future_covariate_std,
                    time_feature_schema=self.time_feature_schema,
                )
                child_id = len(self.children)
                self.children.append(child)
                self.task_datasets.append(child.task_datasets[0])
                self._child_data_versions.append(
                    str(cache_config.get("data_version", ""))
                )
                self._child_cache_names.append(cache_name)
                for local_idx, (
                    _local_task,
                    _sample,
                    station_idx,
                    sample_start,
                ) in enumerate(child.index):
                    self.index.append(
                        (child_id, int(local_idx), int(station_idx), int(sample_start))
                    )
        if not self.index:
            raise ValueError(f"MultiResolutionDataset has no {self.split} samples.")
        self.region_keys = [
            self.children[int(child_id)].region_keys[int(local_idx)]
            for child_id, local_idx, _station_idx, _sample_start in self.index
        ]
        self.source_cache_names = list(dict.fromkeys(self._child_cache_names))
        self.source_cache_names_by_child = list(self._child_cache_names)
        configured_weights = self.registry.get("cache_sampling_weights") or {}
        self.source_cache_weights = {
            name: float(configured_weights.get(name, 0.0))
            for name in self.source_cache_names
        }
        if not any(weight > 0.0 for weight in self.source_cache_weights.values()):
            self.source_cache_weights = {name: 1.0 for name in self.source_cache_names}
        self.sample_weights = sample_weights_from_region_keys(
            self.region_keys, float(region_loss_alpha or 0.0)
        )
        # The dimensions are registry-wide, rather than limited to the task
        # requested by this process.  This is essential for loading a joint
        # 1h+15min checkpoint in a per-task evaluation command.
        registry_dims = _registry_tensor_dims(self.registry)
        self.model_dims = dict(self.children[0].model_dims)
        for child in self.children[1:]:
            for key in (
                "historical_covariate_dim",
                "future_covariate_dim",
                "past_time_dim",
                "future_time_dim",
            ):
                self.model_dims[key] = max(
                    int(self.model_dims[key]), int(child.model_dims[key])
                )
        for key, value in registry_dims.items():
            self.model_dims[key] = max(int(self.model_dims[key]), int(value))
        # The canonical calendar contract always has five channels
        # ``[minute, hour, weekday, day, dayofyear]``.  A registry can route a
        # single legacy 1h cache while its metadata opts into the canonical
        # contract; that cache stores four raw channels and is aligned in
        # ``__getitem__`` by inserting the minute-zero channel.  Reserve the
        # extra slot in the public model dimensions before constructing the
        # collator, otherwise alignment would reject the valid 1h rows with a
        # target width of four.
        if self.time_feature_schema == CANONICAL_TIME_FEATURE_SCHEMA:
            self.model_dims["past_time_dim"] = max(
                5, int(self.model_dims["past_time_dim"])
            )
            self.model_dims["future_time_dim"] = max(
                5, int(self.model_dims["future_time_dim"])
            )
        self.model_dims["context_input_dim"] = (
            1
            + self.model_dims["historical_covariate_dim"]
            + self.model_dims["past_time_dim"]
        )
        self.model_dims["future_input_dim"] = (
            self.model_dims["future_covariate_dim"] + self.model_dims["future_time_dim"]
        )
        self._collator = WindowCollator(
            model_dims=self.model_dims,
            split=self.split,
            metadata_mode=self.metadata_mode,
            include_native_covariates=self.include_native_covariates,
        )
        self.task_summaries = self._build_task_summaries()
        print(
            f"[MultiResolutionDataset] split={self.split} tasks={','.join(self.task_names)} "
            f"samples={len(self.index)} registry={self.binary_cache_registry}",
            flush=True,
        )

    def _build_task_summaries(self):
        result = []
        for child, data_version, cache_name in zip(
            self.children, self._child_data_versions, self._child_cache_names
        ):
            item = dict(child.task_summaries[0])
            item["data_version"] = data_version
            item["cache_name"] = cache_name
            result.append(item)
        return result

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        child_id, local_idx, _station_idx, _sample_start = self.index[int(idx)]
        item = dict(self.children[int(child_id)][int(local_idx)])
        item["dataset_idx"] = int(child_id)
        item["sample_weight"] = float(self.sample_weights[int(idx)])
        item["data_version"] = self._child_data_versions[int(child_id)]
        cache_name = self._child_cache_names[int(child_id)]
        item["cache_name"] = cache_name
        item["weather_policy"] = self.registry["caches"][cache_name].get(
            "weather_policy", ""
        )
        resolution = str(self.task_datasets[int(child_id)].task_spec.resolution)
        for key, dim_key in (
            ("past_time_features", "past_time_dim"),
            ("future_time_features", "future_time_dim"),
        ):
            item[key] = align_multires_time_features(
                item[key],
                resolution=resolution,
                target_dim=self.model_dims[dim_key],
                schema=self.time_feature_schema,
            )
        return item

    def collate_fn(self, items):
        history_length = int(items[0]["seq_len"]) if items else None
        return self._collator(items, history_length=history_length)

    def logical_epoch_steps(self):
        if self.drop_last:
            return len(self.index) // self.batch_size
        return int(math.ceil(float(len(self.index)) / float(self.batch_size)))
