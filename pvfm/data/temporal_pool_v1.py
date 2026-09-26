from __future__ import annotations

import hashlib

import json

import math

import os

import random

from collections import Counter, defaultdict, deque

from pathlib import Path

import numpy as np

import torch

import torch.distributed as dist

from torch.utils.data import Sampler

from pvfm.data.window_shapes import (
    WindowMaterializer,
    DynamicPoolUnavailable,
    sample_native_ratio_context_horizon,
    sample_ratio_context_horizon,
    TYPICAL_WEIGHTS,
    NATIVE_TYPICAL_HORIZONS,
    NATIVE_TYPICAL_WEIGHTS,
    DEFAULT_NATIVE_HORIZON_MIN_HOURS,
    DEFAULT_NATIVE_HORIZON_MAX_HOURS,
    DEFAULT_NATIVE_CONTEXT_MIN_RATIO,
    DEFAULT_NATIVE_CONTEXT_MAX_RATIO,
    DEFAULT_NATIVE_CONTEXT_MAX_HOURS,
    DEFAULT_CONTEXT_MIN_RATIO,
    DEFAULT_CONTEXT_MAX_RATIO,
    DEFAULT_CONTEXT_MAX_HOURS,
)

from pvfm.data.cache_contract import (
    _array_contract_reason,
    cache_fingerprint,
    split_bounds,
)

from pvfm.data.regions import (
    normalize_region_weights,
    _stable_probability_caps,
    canonical_region_key,
)

from pvfm.task_specs.pv_tasks import PV_STEPS_PER_HOUR

PROTOCOL = "temporal_pool_v1"


METADATA_FILE = "metadata.json"


SAMPLER_STATE_VERSION = 3


ORIGIN_SUBSET_PROTOCOL = "nested_lcg32_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(payload):
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _station_records(child):
    records = getattr(child, "station_records", None)
    if records:
        return records
    stations = getattr(child, "stations", None)
    if stations is None:
        return []
    try:
        return stations.to_dict("records")
    except AttributeError:
        return list(stations)


def _cadence_segments(arrays, resolution):
    """Split discontinuous edges without dropping their valid endpoint rows."""
    times = np.asarray(arrays["timestamps"])
    if times.ndim != 1 or len(times) == 0:
        return []
    try:
        ns = times.astype("datetime64[ns]").astype(np.int64)
        step = int(round(3600 * 1_000_000_000 / PV_STEPS_PER_HOUR[str(resolution)]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return []
    # Validity belongs to a row; cadence belongs to the edge between rows.
    # A gap closes the preceding segment and starts one at the next valid
    # row. Only NaT rows are excluded, not the first observation after a gap.
    good = ns != np.iinfo(np.int64).min
    breaks = np.diff(ns) != step
    starts = np.flatnonzero(good & np.r_[True, ~good[:-1] | breaks])
    ends = np.flatnonzero(good & np.r_[~good[1:] | breaks, True]) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def _pool_source_signature(base_dataset, rows):
    children = getattr(base_dataset, "children", ())
    signatures = {}
    for row in rows:
        key = (int(row["child_index"]), int(row["station_index"]))
        if key in signatures:
            continue
        child = children[key[0]]
        try:
            signatures[f"{key[0]}:{key[1]}"] = cache_fingerprint(child)
        except Exception:
            signatures[f"{key[0]}:{key[1]}"] = str(
                getattr(child, "binary_dataset_dir", "")
            )
    return signatures


def build_temporal_pool(
    base_dataset, output_dir, *, split="train", config=None, rebuild=False
):
    """Build a station/segment descriptor pool from full station timelines.

    Existing destinations are never overwritten.  ``rebuild=True`` writes to
    a new destination only; callers should choose a new run-local directory.
    Segmentation fixes affect new pools only; loading an existing pool keeps
    its saved boundaries and fingerprint for reproducible resume.
    """
    if str(split) != "train":
        raise ValueError("temporal_pool_v1 currently supports the training split only")
    destination = Path(output_dir).resolve()
    metadata_path = destination / METADATA_FILE
    if metadata_path.exists():
        return load_temporal_pool(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty temporal pool: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    children = getattr(base_dataset, "children", ())
    cache_names = list(getattr(base_dataset, "source_cache_names_by_child", ()))
    for child_index, child in enumerate(children):
        task = base_dataset.task_datasets[child_index]
        resolution = str(task.task_spec.resolution)
        if resolution not in {"1h", "15min"}:
            raise ValueError(
                f"temporal_pool_v1 supports only 1h and 15min, got {resolution!r}"
            )
        cache_name = str(
            cache_names[child_index] if child_index < len(cache_names) else "default"
        )
        records = _station_records(child)
        stations = getattr(child, "stations", None)
        if stations is not None:
            try:
                station_count = len(stations)
            except TypeError:
                station_count = len(records)
        else:
            station_count = len(records)
        # The binary child may keep the complete stations table even when the
        # caller requested a station subset.  The immutable child index is
        # used only to recover that already-filtered station set; no candidate
        # cutoff is taken from its task-specific rows.
        allowed_station_indices = sorted(
            int(value) for value in getattr(child, "_allowed_station_indices", ())
        )
        if not allowed_station_indices:
            # Lightweight test/fallback datasets do not expose the binary
            # station filter; recover only its station membership, never its
            # task-specific cutoff rows.
            allowed_station_indices = sorted(
                {int(row[2]) for row in getattr(child, "index", ())}
            )
        if not allowed_station_indices:
            allowed_station_indices = list(range(station_count))
        for station_index in allowed_station_indices:
            record = records[station_index] if station_index < len(records) else {}
            try:
                station_key = str(child._station_key(station_index))
                station_dir = str(child._station_dir(station_index))
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                station_key = str(record.get("station_key", station_index))
                station_dir = str(record.get("station_dir", station_key))
            region = canonical_region_key(record)
            unique_key = (cache_name, resolution, station_key, region)
            if unique_key in seen:
                continue
            seen.add(unique_key)
            try:
                arrays = child._station_arrays(resolution, station_index)
                reason = _array_contract_reason(arrays)
                segments = _cadence_segments(arrays, resolution) if not reason else []
                length = int(np.asarray(arrays["target"]).shape[0])
            except (
                AttributeError,
                FileNotFoundError,
                IndexError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ):
                continue
            if not segments or length <= 0:
                continue
            quality = {}
            for mask_name in (
                "target_mask",
                "history_covariate_mask",
                "future_covariate_mask",
            ):
                try:
                    mask = np.asarray(arrays[mask_name])
                    quality[f"{mask_name}_valid_ratio"] = (
                        float(np.mean(mask > 0.0)) if mask.size else 1.0
                    )
                except (KeyError, TypeError, ValueError):
                    quality[f"{mask_name}_valid_ratio"] = 0.0
            rows.append(
                {
                    "child_index": int(child_index),
                    "station_index": int(station_index),
                    "cache": cache_name,
                    "resolution": resolution,
                    "region": region,
                    "station_key": station_key,
                    "station_dir": station_dir,
                    "length": length,
                    "segments": [[int(start), int(end)] for start, end in segments],
                    "quality": quality,
                }
            )
    if not rows:
        raise ValueError("temporal_pool_v1 found no valid station timelines")
    config = dict(config or {})
    metadata = {
        "protocol": PROTOCOL,
        "split": "train",
        "schema_version": 1,
        "config": config,
        "rows": len(rows),
        "source_cache_names": sorted({row["cache"] for row in rows}),
        "source_resolutions": sorted({row["resolution"] for row in rows}),
        "source_signature": _pool_source_signature(base_dataset, rows),
        "pool_fingerprint": _fingerprint(rows),
    }
    temporary_path = destination / f"{METADATA_FILE}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "rows": rows}, handle, indent=2, sort_keys=True
        )
        handle.write("\n")
    os.replace(temporary_path, metadata_path)
    return load_temporal_pool(destination)


def load_temporal_pool(output_dir):
    path = Path(output_dir).resolve() / METADATA_FILE
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata") or {}
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Unsupported temporal pool protocol: {metadata.get('protocol')!r}"
        )
    rows = payload.get("rows") or []
    if metadata.get("pool_fingerprint") != _fingerprint(rows):
        raise ValueError("temporal_pool_v1 metadata fingerprint mismatch")
    return metadata, rows


def validate_temporal_pool(pool, base_dataset):
    """Fail closed when a reusable pool is pointed at different source data."""
    metadata, rows = pool
    expected = metadata.get("source_signature") or {}
    actual = _pool_source_signature(base_dataset, rows)
    for key, value in expected.items():
        if str(actual.get(key, "")) != str(value):
            raise ValueError(
                "temporal_pool_v1 source fingerprint mismatch; build a new pool directory"
            )
    return pool


class TemporalPoolDataset(WindowMaterializer):
    """Dynamic materializer backed by station/segment rows."""

    def __init__(
        self,
        base_dataset,
        pool,
        context_max_hours=DEFAULT_CONTEXT_MAX_HOURS,
        horizon_max_hours=168,
        dynamic_native_task_names=(),
    ):
        self.pool_metadata, self.pool_rows = pool
        super().__init__(
            base_dataset,
            context_max_hours,
            horizon_max_hours,
            dynamic_native_task_names=dynamic_native_task_names,
        )
        self.index = list(range(len(self.pool_rows)))
        self.region_keys = [str(row["region"]) for row in self.pool_rows]
        self.source_cache_names_by_child = [str(row["cache"]) for row in self.pool_rows]
        self.source_cache_names = sorted({str(row["cache"]) for row in self.pool_rows})
        self.task_datasets = list(base_dataset.task_datasets)
        self.task_summaries = []
        self.quality_index = None

    def ensure_quality_index(self, directory=None):
        if self.quality_index is None or directory is not None:
            from pvfm.data.temporal_quality_index import build_or_load_quality_index

            self.quality_index = build_or_load_quality_index(self, directory)
        return self.quality_index

    def _parts(self, descriptor):
        if isinstance(descriptor, (tuple, list)) and len(descriptor) >= 4:
            return (
                int(descriptor[0]),
                int(descriptor[1]),
                int(descriptor[2]),
                int(descriptor[3]),
            )
        raise TypeError(
            "TemporalPoolDataset expects (pool_row, context_len, pred_len, cutoff) descriptors"
        )

    def _window(self, sample_index, context_len, pred_len, cutoff=None):
        row = self.pool_rows[int(sample_index)]
        child_index = int(row["child_index"])
        station_index = int(row["station_index"])
        resolution = str(row["resolution"])
        if cutoff is None:
            raise ValueError("temporal_pool_v1 requires an explicit cutoff")
        cutoff = int(cutoff)
        history_start = cutoff - int(context_len)
        future_end = cutoff + int(pred_len)
        arrays = self._source_dataset(child_index)._station_arrays(
            resolution, station_index
        )
        if history_start < 0 or future_end > int(np.asarray(arrays["target"]).shape[0]):
            return None, "out_of_bounds"
        return (
            child_index,
            station_index,
            history_start,
            cutoff,
            future_end,
            arrays,
            resolution,
        ), None

    def validity_reason(self, sample_index, context_len, pred_len, cutoff=None):
        row = self.pool_rows[int(sample_index)]
        resolution = str(row["resolution"])
        if resolution == "1h":
            if int(context_len) < 6 or int(context_len) > self.context_max_hours:
                return "context_bounds"
            if int(pred_len) < 1 or int(pred_len) > self.horizon_max_hours:
                return "horizon_bounds"
        else:
            if int(context_len) < 1 or int(pred_len) < 1:
                return "native_shape_bounds"
        window, reason = self._window(sample_index, context_len, pred_len, cutoff)
        if reason:
            return reason
        child_index, _station, start, actual_cutoff, end, arrays, _ = window
        split_start, split_end = split_bounds(
            int(row["length"]), context_len, pred_len, self.split
        )
        if start < split_start or end > split_end:
            return "split_boundary"
        target_mask = np.asarray(arrays["target_mask"][start:end])
        if target_mask.shape[0] != int(context_len) + int(pred_len) or not np.all(
            target_mask > 0
        ):
            return "target_mask"
        task = self.task_datasets[child_index]
        for key, lo, phase in (
            (
                "history_covariate_mask",
                task.min_history_covariate_valid_ratio,
                "history_covariates",
            ),
            (
                "future_covariate_mask",
                task.min_future_covariate_valid_ratio,
                "future_covariates",
            ),
        ):
            values = np.asarray(
                arrays[key][start:actual_cutoff]
                if phase.startswith("history")
                else arrays[key][actual_cutoff:end]
            )
            if values.size and float(np.mean(values > 0.0)) < float(lo):
                return f"{phase}_mask"
        return None

    def __getitem__(self, descriptor):
        sample_index, context_len, pred_len, cutoff = self._parts(descriptor)
        window, reason = self._window(sample_index, context_len, pred_len, cutoff)
        if reason:
            raise RuntimeError(f"Temporal C/H descriptor is out of bounds: {reason}")
        # Reuse the stable materializer in WindowMaterializer while passing
        # the explicit cutoff through a temporary four-field descriptor.
        return _materialize_temporal_item(
            self, sample_index, context_len, pred_len, cutoff, window
        )

    def collate_fn(self, items):
        batch = super().collate_fn(items)
        # DataLoader calls this with materialized item dictionaries.  The
        # cutoff is included for sampler resume/audit without changing any
        # model-facing tensor fields.
        batch["temporal_cutoff_indices"] = [
            int(item["temporal_cutoff_index"]) for item in items
        ]
        return batch


def _materialize_temporal_item(
    dataset, sample_index, context_len, pred_len, cutoff, window
):
    """Materialize one item using the same fields as WindowMaterializer."""
    (
        child_index,
        station_index,
        history_start,
        _cutoff,
        future_end,
        arrays,
        resolution,
    ) = window
    task = dataset.task_datasets[child_index]
    source_dataset = dataset._source_dataset(child_index)
    history_end = int(cutoff)
    from pvfm.data.cache_contract import _soft_mask_covariate_block

    history_cov, history_cov_mask = _soft_mask_covariate_block(
        np.asarray(
            arrays["history_covariates"][history_start:history_end], dtype=np.float32
        ),
        np.asarray(
            arrays["history_covariate_mask"][history_start:history_end],
            dtype=np.float32,
        ),
        min_valid_ratio=task.min_history_covariate_valid_ratio,
        min_std=task.min_history_covariate_std,
    )
    future_cov, future_cov_mask = _soft_mask_covariate_block(
        np.asarray(arrays["future_covariates"][cutoff:future_end], dtype=np.float32),
        np.asarray(
            arrays["future_covariate_mask"][cutoff:future_end], dtype=np.float32
        ),
        min_valid_ratio=task.min_future_covariate_valid_ratio,
        min_std=task.min_future_covariate_std,
    )
    steps_per_hour = PV_STEPS_PER_HOUR.get(str(resolution), 1)
    horizon_hours = (
        float(pred_len)
        if resolution == "1h"
        else float(pred_len) / float(steps_per_hour)
    )
    return {
        "source_sample_index": int(sample_index),
        "source_task_name": str(task.task_name),
        "source_cache_name": str(dataset.pool_rows[sample_index]["cache"]),
        "task_name": f"dynamic_{resolution}_c{int(context_len)}_h{int(pred_len)}",
        "resolution": resolution,
        "station_index": int(station_index),
        "station_id": source_dataset._station_dir(station_index),
        "region_key": str(dataset.pool_rows[sample_index]["region"]),
        "seq_len": int(context_len),
        "label_len": int(pred_len),
        "pred_len": int(pred_len),
        "temporal_cutoff_index": int(cutoff),
        "past_target": np.asarray(
            arrays["target"][history_start:history_end], dtype=np.float32
        ),
        "past_observed_mask": np.asarray(
            arrays["target_mask"][history_start:history_end], dtype=np.float32
        ),
        "historical_covariates": history_cov,
        "historical_covariates_mask": history_cov_mask,
        "future_covariates": future_cov,
        "future_covariates_mask": future_cov_mask,
        "future_target": np.asarray(
            arrays["target"][cutoff:future_end], dtype=np.float32
        ),
        "future_observed_mask": np.asarray(
            arrays["target_mask"][cutoff:future_end], dtype=np.float32
        ),
        "past_time_features": np.asarray(
            arrays["time_features"][history_start:history_end], dtype=np.float32
        ),
        "future_time_features": np.asarray(
            arrays["time_features"][cutoff:future_end], dtype=np.float32
        ),
        "static_features": np.asarray(
            [context_len, pred_len, pred_len, horizon_hours], dtype=np.float32
        ),
        "site_features": np.asarray(arrays["site_features"], dtype=np.float32).reshape(
            -1
        )[:5],
    }


class TemporalPoolSampler(Sampler):
    """C/H calibration and state machinery for AverageSupplyTemporalSampler.

    ``batches_per_epoch`` counts optimizer steps (batches *per rank*), not
    batches summed across ranks.  No step-expanded pool artifact is needed.
    This base does not provide an alternative station-equal sampling policy.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        *,
        seed=0,
        rank=0,
        world_size=1,
        batches_per_epoch=0,
        source_weights=None,
        region_balance_alpha=0.25,
        region_balance_max_prob=0.15,
        context_max_hours=DEFAULT_CONTEXT_MAX_HOURS,
        context_min_ratio=DEFAULT_CONTEXT_MIN_RATIO,
        context_max_ratio=DEFAULT_CONTEXT_MAX_RATIO,
        typical_probability=0.6,
        typical_weights=TYPICAL_WEIGHTS,
        native_typical_probability=0.6,
        native_typical_horizons=NATIVE_TYPICAL_HORIZONS,
        native_typical_weights=NATIVE_TYPICAL_WEIGHTS,
        native_horizon_min_hours=DEFAULT_NATIVE_HORIZON_MIN_HOURS,
        native_horizon_max_hours=DEFAULT_NATIVE_HORIZON_MAX_HOURS,
        native_context_min_ratio=DEFAULT_NATIVE_CONTEXT_MIN_RATIO,
        native_context_max_ratio=DEFAULT_NATIVE_CONTEXT_MAX_RATIO,
        native_context_max_hours=DEFAULT_NATIVE_CONTEXT_MAX_HOURS,
        dynamic_native_task_names=(),
        max_retries=64,
        enable_resume_state=False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed or 0)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if (
            self.batch_size < 1
            or self.world_size < 1
            or not 0 <= self.rank < self.world_size
        ):
            raise ValueError("Invalid temporal sampler batch_size/rank/world_size")
        self.local_batches = max(
            1, int(batches_per_epoch or math.ceil(len(dataset) / self.batch_size))
        )
        self.global_batches = self.local_batches * self.world_size
        self.region_balance_alpha = max(0.0, float(region_balance_alpha))
        self.region_balance_max_prob = max(0.0, float(region_balance_max_prob))
        self.context_max_hours = int(context_max_hours)
        self.context_min_ratio = float(context_min_ratio)
        self.context_max_ratio = float(context_max_ratio)
        self.typical_probability = float(typical_probability)
        self.typical_weights = tuple(float(x) for x in typical_weights)
        self.use_h6_special_contexts = (
            False  # Fixed shape-profile field, not a sampling switch.
        )
        self.dynamic_native_task_names = frozenset(
            str(x) for x in dynamic_native_task_names
        )
        self.native_typical_probability = float(native_typical_probability)
        self.native_typical_horizons = tuple(float(x) for x in native_typical_horizons)
        self.native_typical_weights = tuple(float(x) for x in native_typical_weights)
        self.native_horizon_min_hours = float(native_horizon_min_hours)
        self.native_horizon_max_hours = float(native_horizon_max_hours)
        self.native_context_min_ratio = float(native_context_min_ratio)
        self.native_context_max_ratio = float(native_context_max_ratio)
        self.native_context_max_hours = float(native_context_max_hours)
        self.max_retries = max(1, int(max_retries))
        self.enable_resume_state = bool(enable_resume_state)
        self.epoch = 0
        self.cache_sampling_mode = "temporal_source"
        self._resume_pending = deque()
        self._generated_local_batches = 0
        self._consumed_local_batches = 0
        self._iter_active = False
        self._resume_replay_pending = False
        self._schedule_rng = None
        self._selection_rng = None
        self.rejections = Counter()
        self.length_counts = Counter()
        self.region_counts = Counter()
        self.excluded_region_counts = Counter()
        self.origin_subset_counts = Counter()
        rows = dataset.pool_rows
        self._groups = defaultdict(list)
        for index, row in enumerate(rows):
            self._groups[
                (str(row["resolution"]), str(row["cache"]), str(row["region"]))
            ].append(index)
        configured = dict(source_weights or {"1h": 0.75, "15min": 0.25})
        available = sorted({resolution for resolution, _cache, _region in self._groups})
        weights = {res: max(0.0, float(configured.get(res, 0.0))) for res in available}
        if not any(weights.values()):
            weights = {res: 1.0 for res in available}
        total = sum(weights.values())
        self.source_probabilities = {
            res: weights[res] / total for res in sorted(weights)
        }
        configured_cache_weights = dict(
            getattr(dataset, "source_cache_weights", {}) or {}
        )
        self.cache_weights = {
            str(name): max(0.0, float(value))
            for name, value in configured_cache_weights.items()
        }
        self._regions = {}
        for res in sorted(available):
            keys = sorted({key for key in self._groups if key[0] == res})
            region_counts = Counter(
                {
                    (cache, region): len(self._groups[(res, cache, region)])
                    for _r, cache, region in keys
                }
            )
            grouped = defaultdict(list)
            for (cache, region), count in region_counts.items():
                grouped[cache].append((region, count))
            for cache, entries in grouped.items():
                raw = {
                    region: float(count) ** (-self.region_balance_alpha)
                    for region, count in entries
                }
                probs = normalize_region_weights(raw)
                if self.region_balance_max_prob > 0.0:
                    bounds = {region: self.region_balance_max_prob for region in raw}
                    if sum(bounds.values()) < 1.0:
                        bounds = {region: 1.0 for region in raw}
                    probs = _stable_probability_caps(probs, bounds)
                self._regions[(res, cache)] = (
                    sorted(raw),
                    [probs[r] for r in sorted(raw)],
                )
        self.pool_fingerprint = str(dataset.pool_metadata.get("pool_fingerprint", ""))
        self.station_subset_fingerprint = str(
            dataset.pool_metadata.get("station_subset_fingerprint", "")
        )
        # Fixed full-data fields preserve existing sampler checkpoint fingerprints.
        # Origin thinning is not implemented or configurable in this release.
        self.origin_subset_fingerprint = _fingerprint(
            {
                "protocol": ORIGIN_SUBSET_PROTOCOL,
                "pool_fingerprint": self.pool_fingerprint,
                "fraction": 1.0,
                "seed": 20260923,
            }
        )
        self.quality_index = dataset.ensure_quality_index()
        self.region_sampling_policy = "feasible_given_source_ch_v1"
        sampler_config = {
            "quality_index_fingerprint": self.quality_index.fingerprint,
            "region_sampling_policy": self.region_sampling_policy,
            "source_probabilities": self.source_probabilities,
            "cache_weights": self.cache_weights,
            "regions": [(key, value) for key, value in sorted(self._regions.items())],
            "shape_config": {
                key: getattr(self, key)
                for key in (
                    "context_max_hours",
                    "context_min_ratio",
                    "context_max_ratio",
                    "typical_probability",
                    "typical_weights",
                    "use_h6_special_contexts",
                    "native_typical_probability",
                    "native_typical_horizons",
                    "native_typical_weights",
                    "native_horizon_min_hours",
                    "native_horizon_max_hours",
                    "native_context_min_ratio",
                    "native_context_max_ratio",
                    "native_context_max_hours",
                )
            },
            "max_retries": self.max_retries,
            "qc": [
                {
                    key: getattr(task, key, None)
                    for key in (
                        "min_history_covariate_valid_ratio",
                        "min_future_covariate_valid_ratio",
                        "min_history_covariate_std",
                        "min_future_covariate_std",
                    )
                }
                for task in getattr(dataset, "task_datasets", ())
            ],
        }
        if self.station_subset_fingerprint:
            sampler_config["station_subset_fingerprint"] = (
                self.station_subset_fingerprint
            )
        self.sampler_config_fingerprint = _fingerprint(sampler_config)

    def __len__(self):
        return self.local_batches

    def _sample_shape(self, rng, resolution):
        if resolution == "15min":
            return sample_native_ratio_context_horizon(
                rng,
                typical_probability=self.native_typical_probability,
                typical_horizons=self.native_typical_horizons,
                typical_weights=self.native_typical_weights,
                horizon_min_steps=max(
                    1, int(math.ceil(self.native_horizon_min_hours * 4))
                ),
                horizon_max_steps=max(
                    1, int(math.floor(self.native_horizon_max_hours * 4))
                ),
                steps_per_hour=4,
                context_min_ratio=self.native_context_min_ratio,
                context_max_ratio=self.native_context_max_ratio,
                context_max_steps=max(
                    1, int(math.floor(self.native_context_max_hours * 4))
                ),
            )[:2]
        return sample_ratio_context_horizon(
            rng,
            typical_probability=self.typical_probability,
            typical_weights=self.typical_weights,
            context_max_hours=self.context_max_hours,
            context_min_ratio=self.context_min_ratio,
            context_max_ratio=self.context_max_ratio,
        )[:2]

    def _sample_schedule(self, rng):
        raise NotImplementedError("Use AverageSupplyTemporalSampler for training")

    def _draw_batch(self, resolution, cache, region, c, h, rng):
        raise NotImplementedError("Use AverageSupplyTemporalSampler for training")

    def _next_schedule(self, distributed_sync):
        if not distributed_sync:
            # Also supports deterministic CPU-only rank simulations.  Window
            # rejection never consumes this shared schedule RNG.
            return self._sample_schedule(self._schedule_rng)
        payload = [None]
        if self.rank == 0:
            try:
                payload[0] = {"schedule": self._sample_schedule(self._schedule_rng)}
            except Exception as exc:
                payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
        # BatchSampler runs in the DataLoader parent, not in its workers.
        dist.broadcast_object_list(payload, src=0)
        if "error" in payload[0]:
            raise DynamicPoolUnavailable(
                f"temporal_pool_v1 schedule failed: {payload[0]['error']}"
            )
        return tuple(payload[0]["schedule"])

    def _local_batch(self, schedule, distributed_sync):
        error = None
        batch = None
        try:
            batch = self._draw_batch(*schedule, self._selection_rng)
        except Exception as exc:
            error = exc
        if distributed_sync:
            # If one rank cannot fill this fixed C/H, fail all ranks before
            # any enters model collectives.  Never silently resample a shape.
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if dist.get_backend() == "nccl"
                else torch.device("cpu")
            )
            failed = torch.tensor(
                int(error is not None), dtype=torch.int32, device=device
            )
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item():
                errors = [None] * self.world_size
                dist.all_gather_object(
                    errors, f"{type(error).__name__}: {error}" if error else None
                )
                raise DynamicPoolUnavailable(
                    f"temporal_pool_v1 rank-local window failure: {errors}"
                )
        elif error is not None:
            raise error
        return batch

    def __iter__(self):
        distributed_sync = (
            self.world_size > 1 and dist.is_available() and dist.is_initialized()
        )
        if distributed_sync and (
            dist.get_world_size() != self.world_size or dist.get_rank() != self.rank
        ):
            raise ValueError(
                "temporal_pool_v1 sampler rank/world_size does not match process group"
            )
        if self._resume_replay_pending:
            self._resume_replay_pending = False
            for pending_batch in list(self._resume_pending):
                yield list(pending_batch)
            if not self._iter_active:
                return
        if not self._iter_active:
            if self._resume_pending:
                raise RuntimeError(
                    "temporal_pool_v1 previous epoch has unconsumed batches"
                )
            self._schedule_rng = random.Random(self.seed + self.epoch * 1000003)
            self._selection_rng = random.Random(
                self.seed + self.epoch * 1000003 + self.rank * 1000033
            )
            self._generated_local_batches = 0
            self._consumed_local_batches = 0
            self._iter_active = True
        while self._generated_local_batches < self.local_batches:
            schedule = self._next_schedule(distributed_sync)
            batch = self._local_batch(schedule, distributed_sync)
            resolution, _cache, _region, c, h = schedule
            self.length_counts[(resolution, c, h)] += self.batch_size
            self.region_counts[(resolution, _cache, _region)] += self.batch_size
            self._generated_local_batches += 1
            if self.enable_resume_state:
                self._resume_pending.append(list(batch))
            yield batch
        self._iter_active = False
        self._schedule_rng = self._selection_rng = None
        self.epoch += 1

    def mark_consumed(self, batch):
        if not self.enable_resume_state:
            return None
        actual = [
            (int(index), int(context), int(horizon), int(cutoff))
            for index, context, horizon, cutoff in zip(
                batch["source_sample_indices"],
                batch["seq_lens"],
                batch["pred_lens"],
                batch.get("temporal_cutoff_indices", ()),
            )
        ]
        if not self._resume_pending:
            raise RuntimeError("temporal_pool_v1 resume queue is empty")
        expected = [
            tuple(int(value) for value in descriptor)
            for descriptor in self._resume_pending[0]
        ]
        if actual != expected:
            raise RuntimeError(
                "temporal_pool_v1 consumed batch does not match sampler queue"
            )
        self._resume_pending.popleft()
        self._consumed_local_batches += 1

    def resume_sync_state(self):
        return {
            "version": SAMPLER_STATE_VERSION,
            "epoch": int(self.epoch),
            "pool_fingerprint": self.pool_fingerprint,
            "sampler_config_fingerprint": self.sampler_config_fingerprint,
            "quality_index_fingerprint": self.quality_index.fingerprint,
            "region_sampling_policy": self.region_sampling_policy,
            "pretrain_data_fraction": 1.0,
            "pretrain_data_fraction_seed": 20260923,
            "origin_subset_fingerprint": self.origin_subset_fingerprint,
            "iter_active": self._iter_active,
            "generated_local_batches": int(self._generated_local_batches),
            "consumed_local_batches": int(self._consumed_local_batches),
            "pending_shapes": [
                (
                    self.dataset.pool_rows[batch[0][0]]["resolution"],
                    batch[0][1],
                    batch[0][2],
                )
                for batch in self._resume_pending
            ],
        }

    def state_dict(self):
        if not self.enable_resume_state:
            raise RuntimeError("temporal_pool_v1 resume tracking is not enabled")
        return {
            "version": SAMPLER_STATE_VERSION,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
            "global_batches": self.global_batches,
            "local_batches": self.local_batches,
            "sampler_config_fingerprint": self.sampler_config_fingerprint,
            "quality_index_fingerprint": self.quality_index.fingerprint,
            "region_sampling_policy": self.region_sampling_policy,
            "pretrain_data_fraction": 1.0,
            "pretrain_data_fraction_seed": 20260923,
            "origin_subset_fingerprint": self.origin_subset_fingerprint,
            "pool_fingerprint": self.pool_fingerprint,
            "epoch": self.epoch,
            "iter_active": self._iter_active,
            "schedule_rng_state": (
                self._schedule_rng.getstate() if self._schedule_rng else None
            ),
            "selection_rng_state": (
                self._selection_rng.getstate() if self._selection_rng else None
            ),
            "generated_local_batches": self._generated_local_batches,
            "consumed_local_batches": self._consumed_local_batches,
            "length_counts": dict(self.length_counts),
            "rejections": dict(self.rejections),
            "region_counts": dict(self.region_counts),
            "excluded_region_counts": dict(self.excluded_region_counts),
            "origin_subset_counts": dict(self.origin_subset_counts),
            "pending_batches": [list(batch) for batch in self._resume_pending],
        }

    def load_state_dict(self, state):
        if not self.enable_resume_state:
            raise RuntimeError("temporal_pool_v1 resume tracking is not enabled")
        if state.get("pretrain_data_fraction", 1.0) != 1.0:
            raise ValueError(
                "Origin-subset checkpoints are not supported by the full-data loader"
            )
        if state.get("version") != SAMPLER_STATE_VERSION:
            raise ValueError(
                "Unsupported temporal sampler state version; pre-fix rank-strided or unconditional-region states cannot "
                "be exactly resumed by the feasibility-conditioned sampler. The pool itself remains reusable."
            )
        for key, current in (
            ("seed", self.seed),
            ("rank", self.rank),
            ("world_size", self.world_size),
            ("batch_size", self.batch_size),
            ("global_batches", self.global_batches),
            ("local_batches", self.local_batches),
            ("sampler_config_fingerprint", self.sampler_config_fingerprint),
            ("quality_index_fingerprint", self.quality_index.fingerprint),
            ("region_sampling_policy", self.region_sampling_policy),
            ("pool_fingerprint", self.pool_fingerprint),
        ):
            if state.get(key) != current:
                raise ValueError(
                    f"temporal_pool_v1 resume mismatch for {key}: {state.get(key)!r} != {current!r}"
                )
        generated = int(state["generated_local_batches"])
        consumed = int(state["consumed_local_batches"])
        pending = [list(batch) for batch in state["pending_batches"]]
        if (
            not 0 <= consumed <= generated <= self.local_batches
            or len(pending) != generated - consumed
        ):
            raise ValueError("temporal_pool_v1 invalid resume cursor/prefetch queue")
        if any(len(batch) != self.batch_size for batch in pending):
            raise ValueError("temporal_pool_v1 invalid resume batch size")
        active = bool(state["iter_active"])
        rngs = []
        for key in ("schedule_rng_state", "selection_rng_state"):
            value = state.get(key)
            if active != (value is not None):
                raise ValueError(f"temporal_pool_v1 invalid resume RNG state: {key}")
            rng = random.Random() if value is not None else None
            if rng is not None:
                rng.setstate(value)
            rngs.append(rng)
        self.epoch = int(state.get("epoch", 0))
        self._iter_active = active
        self._generated_local_batches = generated
        self._consumed_local_batches = consumed
        self._resume_pending = deque(pending)
        self._resume_replay_pending = bool(pending)
        self._schedule_rng, self._selection_rng = rngs
        self.length_counts = Counter(state.get("length_counts", {}))
        self.rejections = Counter(state.get("rejections", {}))
        self.region_counts = Counter(state.get("region_counts", {}))
        self.excluded_region_counts = Counter(state.get("excluded_region_counts", {}))
        self.origin_subset_counts = Counter(state.get("origin_subset_counts", {}))

    def write_sampling_audit(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"sampling_audit_rank{self.rank}.json")
        csv_path = os.path.join(directory, f"sampling_audit_rank{self.rank}.csv")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol": PROTOCOL,
                    "pool_fingerprint": self.pool_fingerprint,
                    "sampler_state_version": SAMPLER_STATE_VERSION,
                    "sampler_config_fingerprint": self.sampler_config_fingerprint,
                    "quality_index_fingerprint": self.quality_index.fingerprint,
                    "region_sampling_policy": self.region_sampling_policy,
                    "pretrain_data_fraction": 1.0,
                    "pretrain_data_fraction_seed": 20260923,
                    "origin_subset_protocol": ORIGIN_SUBSET_PROTOCOL,
                    "origin_subset_fingerprint": self.origin_subset_fingerprint,
                    "quality_query_cache": {
                        "hits": self.quality_index.hits,
                        "misses": self.quality_index.misses,
                        "bytes": self.quality_index.cache_bytes,
                        "shapes": len(self.quality_index.cache),
                    },
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "source_probabilities": self.source_probabilities,
                    "length_counts": {str(k): v for k, v in self.length_counts.items()},
                    "region_counts": {str(k): v for k, v in self.region_counts.items()},
                    "excluded_region_counts": {
                        str(k): v for k, v in self.excluded_region_counts.items()
                    },
                    "origin_subset_counts": dict(self.origin_subset_counts),
                    "rejections": dict(self.rejections),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("resolution,context_len,horizon_len,window_draws\n")
            for (resolution, context_len, horizon_len), count in sorted(
                self.length_counts.items(), key=lambda item: str(item[0])
            ):
                handle.write(f"{resolution},{context_len},{horizon_len},{count}\n")
        return {"json": path, "csv": csv_path}
