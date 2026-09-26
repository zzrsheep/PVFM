from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pvfm.data.collate import _pad_last_dim
from pvfm.data.cache_contract import _soft_mask_covariate_block, DynamicPoolUnavailable
from pvfm.data.multires_cache import (
    CANONICAL_TIME_FEATURE_SCHEMA,
    LEGACY_TIME_FEATURE_SCHEMA,
    align_multires_time_features,
)
from pvfm.task_specs.pv_tasks import PV_STEPS_PER_HOUR

TYPICAL_HORIZONS = (6, 24, 168)
TYPICAL_WEIGHTS = (0.333333, 0.333333, 0.333334)
DEFAULT_CONTEXT_MAX_HOURS = 672
DEFAULT_CONTEXT_MIN_RATIO = 0.5
DEFAULT_CONTEXT_MAX_RATIO = 4.0
NATIVE_TYPICAL_HORIZONS = (1.0, 4.0, 6.0, 24.0)
NATIVE_TYPICAL_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
DEFAULT_NATIVE_HORIZON_MIN_HOURS = 0.25
DEFAULT_NATIVE_HORIZON_MAX_HOURS = 24.0
DEFAULT_NATIVE_CONTEXT_MIN_RATIO = 0.5
DEFAULT_NATIVE_CONTEXT_MAX_RATIO = 4.0
DEFAULT_NATIVE_CONTEXT_MAX_HOURS = 96.0


def sample_ratio_context_horizon(
    rng,
    typical_probability=0.6,
    typical_weights=TYPICAL_WEIGHTS,
    context_max_hours=DEFAULT_CONTEXT_MAX_HOURS,
    context_min_ratio=DEFAULT_CONTEXT_MIN_RATIO,
    context_max_ratio=DEFAULT_CONTEXT_MAX_RATIO,
):
    """Return ``(context_hours, horizon_hours, source)`` for the PVFM protocol."""
    context_max_hours = int(context_max_hours)
    if context_max_hours < 6:
        raise ValueError("context_max_hours must be at least 6.")
    context_min_ratio = float(context_min_ratio)
    context_max_ratio = float(context_max_ratio)
    if context_min_ratio <= 0.0 or context_max_ratio < context_min_ratio:
        raise ValueError("context ratios must satisfy 0 < min <= max.")
    if rng.random() < float(typical_probability):
        horizon = int(rng.choices(TYPICAL_HORIZONS, weights=typical_weights, k=1)[0])
        source = "typical"
    else:
        horizon = int(rng.randint(1, 168))
        source = "uniform"
    if horizon == 1:
        return (6, horizon, source)
    context_min = max(6, int(math.ceil(context_min_ratio * horizon)))
    context_max = min(context_max_hours, int(math.floor(context_max_ratio * horizon)))
    if context_max < context_min:
        raise ValueError(
            f"No valid context for H={horizon}: range=[{context_min}, {context_max}] under cap={context_max_hours}."
        )
    return (int(rng.randint(context_min, context_max)), horizon, source)


def sample_native_ratio_context_horizon(
    rng,
    *,
    typical_probability=0.6,
    typical_horizons=NATIVE_TYPICAL_HORIZONS,
    typical_weights=NATIVE_TYPICAL_WEIGHTS,
    horizon_min_steps=1,
    horizon_max_steps=96,
    steps_per_hour=4,
    context_min_ratio=DEFAULT_NATIVE_CONTEXT_MIN_RATIO,
    context_max_ratio=DEFAULT_NATIVE_CONTEXT_MAX_RATIO,
    context_max_steps=384,
):
    """Sample native-resolution ``(C, H)`` lengths in integer data steps.

    Native PV arrays are quarter-hourly. Both lengths are chosen before
    querying legal station/cutoff windows; no nominal task bounds the pool.
    """
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour <= 0:
        raise ValueError("steps_per_hour must be positive.")
    horizon_min_steps = int(horizon_min_steps)
    horizon_max_steps = int(horizon_max_steps)
    context_max_steps = int(context_max_steps)
    if horizon_min_steps < 1 or horizon_max_steps < horizon_min_steps:
        raise ValueError("native horizon step bounds are invalid.")
    if context_max_steps < 1:
        raise ValueError("native context_max_steps must be positive.")
    context_min_ratio = float(context_min_ratio)
    context_max_ratio = float(context_max_ratio)
    if context_min_ratio <= 0.0 or context_max_ratio < context_min_ratio:
        raise ValueError("native context ratios must satisfy 0 < min <= max.")
    horizons = tuple((float(value) for value in typical_horizons))
    weights = tuple((float(value) for value in typical_weights))
    if len(horizons) != len(weights) or not horizons:
        raise ValueError(
            "native typical horizons and weights must have equal non-zero length."
        )
    if any((value <= 0.0 for value in horizons)) or any(
        (value < 0.0 for value in weights)
    ):
        raise ValueError(
            "native typical horizons must be positive and weights non-negative."
        )
    if sum(weights) <= 0.0:
        raise ValueError("native typical horizon weights must have a positive sum.")
    if rng.random() < float(typical_probability):
        horizon_steps = int(
            round(rng.choices(horizons, weights=weights, k=1)[0] * steps_per_hour)
        )
        source = "native_typical"
    else:
        horizon_steps = int(rng.randint(horizon_min_steps, horizon_max_steps))
        source = "native_uniform"
    horizon_steps = max(horizon_min_steps, min(horizon_steps, horizon_max_steps))
    context_min_steps = max(1, int(math.ceil(context_min_ratio * horizon_steps)))
    context_max_for_horizon = int(math.floor(context_max_ratio * horizon_steps))
    context_max_steps = min(context_max_steps, context_max_for_horizon)
    if context_max_steps < context_min_steps:
        raise ValueError(
            f"No valid native context for H steps={horizon_steps}: range=[{context_min_steps}, {context_max_steps}]."
        )
    context_steps = int(rng.randint(context_min_steps, context_max_steps))
    return (context_steps, horizon_steps, source)


class WindowMaterializer(Dataset):
    """Collate variable-context/horizon windows at either supported resolution."""

    yields_batches = False

    def __init__(
        self,
        base_dataset,
        context_max_hours=DEFAULT_CONTEXT_MAX_HOURS,
        horizon_max_hours=168,
        dynamic_native_task_names=(),
    ):
        self.base_dataset = base_dataset
        self.context_max_hours = int(context_max_hours)
        self.horizon_max_hours = int(horizon_max_hours)
        self.dynamic_native_task_names = frozenset(
            (str(name) for name in dynamic_native_task_names or ())
        )
        self.index = base_dataset.index
        self.task_datasets = base_dataset.task_datasets
        self.source_cache_names_by_child = list(
            getattr(
                base_dataset,
                "source_cache_names_by_child",
                ["default"] * len(self.task_datasets),
            )
        )
        self.source_cache_names = list(
            getattr(
                base_dataset,
                "source_cache_names",
                sorted(set(self.source_cache_names_by_child)),
            )
        )
        self.source_cache_weights = dict(
            getattr(base_dataset, "source_cache_weights", {})
        )
        self.cache_sampling_mode = (
            str(getattr(base_dataset, "cache_sampling_mode", "weighted"))
            .strip()
            .lower()
        )
        self.station_records = getattr(base_dataset, "station_records", [])
        self.region_keys = base_dataset.region_keys
        self.model_dims = dict(base_dataset.model_dims)
        declared_schema = getattr(base_dataset, "time_feature_schema", None)
        schema_text = str(declared_schema or "").strip().lower()
        if schema_text in {"canonical", CANONICAL_TIME_FEATURE_SCHEMA}:
            self.time_feature_schema = CANONICAL_TIME_FEATURE_SCHEMA
        elif schema_text in {"legacy", LEGACY_TIME_FEATURE_SCHEMA, ""}:
            time_width = max(
                int(self.model_dims.get("past_time_dim", 0) or 0),
                int(self.model_dims.get("future_time_dim", 0) or 0),
            )
            self.time_feature_schema = (
                CANONICAL_TIME_FEATURE_SCHEMA
                if schema_text == "" and time_width >= 5
                else LEGACY_TIME_FEATURE_SCHEMA
            )
        elif schema_text == "auto":
            time_width = max(
                int(self.model_dims.get("past_time_dim", 0) or 0),
                int(self.model_dims.get("future_time_dim", 0) or 0),
            )
            self.time_feature_schema = (
                CANONICAL_TIME_FEATURE_SCHEMA
                if time_width >= 5
                else LEGACY_TIME_FEATURE_SCHEMA
            )
        else:
            raise ValueError(
                f"Unsupported dynamic time_feature_schema={declared_schema!r}."
            )
        if self.time_feature_schema == CANONICAL_TIME_FEATURE_SCHEMA:
            self.model_dims["past_time_dim"] = max(
                5, int(self.model_dims.get("past_time_dim", 0) or 0)
            )
            self.model_dims["future_time_dim"] = max(
                5, int(self.model_dims.get("future_time_dim", 0) or 0)
            )
            self.model_dims["context_input_dim"] = (
                int(self.model_dims.get("target_dim", 1))
                + int(self.model_dims.get("historical_covariate_dim", 0))
                + int(self.model_dims["past_time_dim"])
            )
            self.model_dims["future_input_dim"] = int(
                self.model_dims.get("future_covariate_dim", 0)
            ) + int(self.model_dims["future_time_dim"])
        self.split = str(base_dataset.split)
        self.metadata_mode = "minimal"
        self.include_native_covariates = False
        self.native_fields_in_batch = False
        self.collator_mode_hint = "dynamic_ratio_ch_padded"
        self.task_summaries = []

    def __len__(self):
        return len(self.index)

    def _source_dataset(self, dataset_idx):
        children = getattr(self.base_dataset, "children", None)
        return children[int(dataset_idx)] if children is not None else self.base_dataset

    def collate_fn(self, items):
        dims = self.model_dims
        batch_size = len(items)
        c_max = max((int(item["seq_len"]) for item in items))
        h_max = max((int(item["pred_len"]) for item in items))

        def zeros(length, dim):
            return np.zeros((batch_size, length, dim), dtype=np.float32)

        past_target = zeros(c_max, dims["target_dim"])
        past_mask = zeros(c_max, dims["target_dim"])
        hist_cov = zeros(c_max, dims["historical_covariate_dim"])
        hist_cov_mask = zeros(c_max, dims["historical_covariate_dim"])
        past_time = zeros(c_max, dims["past_time_dim"])
        future_cov = zeros(h_max, dims["future_covariate_dim"])
        future_cov_mask = zeros(h_max, dims["future_covariate_dim"])
        future_target = zeros(h_max, dims["output_dim"])
        future_mask = zeros(h_max, dims["output_dim"])
        future_time = zeros(h_max, dims["future_time_dim"])
        static = np.zeros((batch_size, dims["static_dim"]), dtype=np.float32)
        site = np.zeros((batch_size, 5), dtype=np.float32)
        names, regions, stations, seq_lens, pred_lens, source_tasks, source_caches = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        plan_ids, base_indices, input_indices, cutoff_indices, future_end_indices = (
            [],
            [],
            [],
            [],
            [],
        )
        input_times, origin_times, target_end_times, qc_versions, cache_fingerprints = (
            [],
            [],
            [],
            [],
            [],
        )
        (
            origin_tasks,
            origin_dataset_indices,
            origin_source_indices,
            origin_base_indices,
        ) = ([], [], [], [])
        routing_tasks, routing_dataset_indices, routing_caches, source_start_indices = (
            [],
            [],
            [],
            [],
        )
        for row, item in enumerate(items):
            c, h = (int(item["seq_len"]), int(item["pred_len"]))
            past_target[row, -c:] = item["past_target"]
            past_mask[row, -c:] = item["past_observed_mask"]
            hist_cov[row, -c:] = _pad_last_dim(
                item["historical_covariates"], dims["historical_covariate_dim"]
            )
            hist_cov_mask[row, -c:] = _pad_last_dim(
                item["historical_covariates_mask"], dims["historical_covariate_dim"]
            )
            past_time[row, -c:] = align_multires_time_features(
                item["past_time_features"],
                item["resolution"],
                dims["past_time_dim"],
                self.time_feature_schema,
            )
            future_cov[row, :h] = _pad_last_dim(
                item["future_covariates"], dims["future_covariate_dim"]
            )
            future_cov_mask[row, :h] = _pad_last_dim(
                item["future_covariates_mask"], dims["future_covariate_dim"]
            )
            future_target[row, :h] = item["future_target"]
            future_mask[row, :h] = item["future_observed_mask"]
            future_time[row, :h] = align_multires_time_features(
                item["future_time_features"],
                item["resolution"],
                dims["future_time_dim"],
                self.time_feature_schema,
            )
            static[row] = item["static_features"]
            site[row] = item["site_features"]
            names.append(item["task_name"])
            regions.append(item["region_key"])
            stations.append(item["station_id"])
            seq_lens.append(c)
            pred_lens.append(h)
            source_tasks.append(item["source_task_name"])
            source_caches.append(item.get("source_cache_name", "default"))
        batch = {
            "past_target": torch.from_numpy(past_target),
            "past_observed_mask": torch.from_numpy(past_mask),
            "historical_covariates": torch.from_numpy(hist_cov),
            "historical_covariates_mask": torch.from_numpy(hist_cov_mask),
            "future_covariates": torch.from_numpy(future_cov),
            "future_covariates_mask": torch.from_numpy(future_cov_mask),
            "future_target": torch.from_numpy(future_target),
            "future_target_mask": torch.from_numpy(future_mask),
            "past_time_features": torch.from_numpy(past_time),
            "future_time_features": torch.from_numpy(future_time),
            "static_features": torch.from_numpy(static),
            "site_features": torch.from_numpy(site),
            "context_padding_mask": torch.from_numpy(
                (past_mask.sum(axis=-1) > 0).astype(np.float32)
            ),
            "collator_mode": "dynamic_ratio_ch_batch_local",
            "task_names": names,
            "source_task_names": source_tasks,
            "source_cache_names": source_caches,
            "resolutions": [str(item["resolution"]) for item in items],
            "station_ids": stations,
            "region_keys": regions,
            "seq_lens": seq_lens,
            "pred_lens": pred_lens,
            "source_sample_indices": [
                int(item["source_sample_index"]) for item in items
            ],
        }
        return batch
