from __future__ import annotations

import numpy as np

import torch


def _pad_last_dim(values, target_dim):
    values = values.astype(np.float32)
    if values.shape[-1] == target_dim:
        return values
    padded = np.zeros((*values.shape[:-1], target_dim), dtype=np.float32)
    padded[..., : values.shape[-1]] = values
    return padded


class WindowCollator:
    """Collate cache windows into the shared PVFM tensor and metadata contract."""

    def __init__(
        self,
        model_dims,
        split="train",
        metadata_mode="minimal",
        include_native_covariates=False,
    ):
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
                if (
                    value.ndim != 2
                    or value.shape[0] < seq_len
                    or value.shape[-1] != dim
                ):
                    return False
            for key, shape in future_shapes.items():
                if tuple(item[key].shape) != tuple(shape):
                    return False
        return True

    def _stack_core_batch(self, items, seq_len, pred_len):
        static_features = np.stack(
            [item["static_features"].copy() for item in items]
        ).astype(np.float32, copy=False)
        if static_features.shape[-1] > 0:
            static_features[:, 0] = float(seq_len)
        return {
            "past_target": np.stack(
                [item["past_target"][-seq_len:] for item in items]
            ).astype(np.float32, copy=False),
            "past_observed_mask": np.stack(
                [item["past_observed_mask"][-seq_len:] for item in items]
            ).astype(np.float32, copy=False),
            "historical_covariates": np.stack(
                [item["historical_covariates"][-seq_len:] for item in items]
            ).astype(np.float32, copy=False),
            "historical_covariates_mask": np.stack(
                [item["historical_covariates_mask"][-seq_len:] for item in items]
            ).astype(np.float32, copy=False),
            "future_covariates": np.stack(
                [item["future_covariates"] for item in items]
            ).astype(np.float32, copy=False),
            "future_covariates_mask": np.stack(
                [item["future_covariates_mask"] for item in items]
            ).astype(np.float32, copy=False),
            "future_target": np.stack([item["future_target"] for item in items]).astype(
                np.float32, copy=False
            ),
            "future_target_mask": np.stack(
                [item["future_observed_mask"] for item in items]
            ).astype(np.float32, copy=False),
            "past_time_features": np.stack(
                [item["past_time_features"][-seq_len:] for item in items]
            ).astype(np.float32, copy=False),
            "future_time_features": np.stack(
                [item["future_time_features"] for item in items]
            ).astype(np.float32, copy=False),
            "static_features": static_features,
            "site_features": np.stack([item["site_features"] for item in items]).astype(
                np.float32, copy=False
            ),
        }

    def __call__(self, items, history_length=None):
        original_items = list(items)
        items = [item for item in original_items if self._sample_is_valid(item)]
        if not items and original_items:
            items = [original_items[0]]
        if not items:
            raise ValueError("WindowCollator received no valid samples.")

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
            past_target = np.zeros(
                (batch_size, seq_len, dims["target_dim"]), dtype=np.float32
            )
            past_observed_mask = np.zeros(
                (batch_size, seq_len, dims["target_dim"]), dtype=np.float32
            )
            historical_covariates = np.zeros(
                (batch_size, seq_len, dims["historical_covariate_dim"]),
                dtype=np.float32,
            )
            historical_covariates_mask = np.zeros(
                (batch_size, seq_len, dims["historical_covariate_dim"]),
                dtype=np.float32,
            )
            future_covariates = np.zeros(
                (batch_size, pred_len, dims["future_covariate_dim"]), dtype=np.float32
            )
            future_covariates_mask = np.zeros(
                (batch_size, pred_len, dims["future_covariate_dim"]), dtype=np.float32
            )
            future_target = np.zeros(
                (batch_size, pred_len, dims["output_dim"]), dtype=np.float32
            )
            future_target_mask = np.zeros(
                (batch_size, pred_len, dims["output_dim"]), dtype=np.float32
            )
            past_time_features = np.zeros(
                (batch_size, seq_len, dims["past_time_dim"]), dtype=np.float32
            )
            future_time_features = np.zeros(
                (batch_size, pred_len, dims["future_time_dim"]), dtype=np.float32
            )
            static_features = np.zeros(
                (batch_size, dims["static_dim"]), dtype=np.float32
            )
            site_features = np.zeros((batch_size, 5), dtype=np.float32)

            for idx, item in enumerate(items):
                past_target[idx] = _pad_last_dim(
                    item["past_target"][-seq_len:], dims["target_dim"]
                )
                past_observed_mask[idx] = _pad_last_dim(
                    item["past_observed_mask"][-seq_len:], dims["target_dim"]
                )
                future_target[idx] = _pad_last_dim(
                    item["future_target"], dims["output_dim"]
                )
                future_target_mask[idx] = _pad_last_dim(
                    item["future_observed_mask"], dims["output_dim"]
                )

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
                    future_covariates[idx] = _pad_last_dim(
                        item["future_covariates"], dims["future_covariate_dim"]
                    )
                    future_covariates_mask[idx] = _pad_last_dim(
                        item["future_covariates_mask"], dims["future_covariate_dim"]
                    )
                if dims["past_time_dim"] > 0:
                    past_time_features[idx] = _pad_last_dim(
                        item["past_time_features"][-seq_len:], dims["past_time_dim"]
                    )
                if dims["future_time_dim"] > 0:
                    future_time_features[idx] = _pad_last_dim(
                        item["future_time_features"], dims["future_time_dim"]
                    )

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
        historical_covariate_mean = np.stack(
            [
                np.asarray(
                    item.get(
                        "historical_covariate_mean",
                        np.zeros(dims["historical_covariate_dim"]),
                    ),
                    dtype=np.float32,
                )
                for item in items
            ]
        )
        historical_covariate_scale = np.stack(
            [
                np.asarray(
                    item.get(
                        "historical_covariate_scale",
                        np.ones(dims["historical_covariate_dim"]),
                    ),
                    dtype=np.float32,
                )
                for item in items
            ]
        )
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
            native_hist_len = max(
                (item["historical_covariates_native"].shape[0] for item in items),
                default=0,
            )
            native_fut_len = max(
                (item["future_covariates_native"].shape[0] for item in items), default=0
            )
            native_hist_time_dim = max(
                (
                    (
                        item["historical_covariates_native_time_features"].shape[-1]
                        if item["historical_covariates_native_time_features"].ndim == 2
                        else 0
                    )
                    for item in items
                ),
                default=0,
            )
            native_fut_time_dim = max(
                (
                    (
                        item["future_covariates_native_time_features"].shape[-1]
                        if item["future_covariates_native_time_features"].ndim == 2
                        else 0
                    )
                    for item in items
                ),
                default=0,
            )
            historical_covariates_native = np.zeros(
                (batch_size, native_hist_len, dims["historical_covariate_dim"]),
                dtype=np.float32,
            )
            historical_covariates_native_mask = np.zeros(
                (batch_size, native_hist_len, dims["historical_covariate_dim"]),
                dtype=np.float32,
            )
            future_covariates_native = np.zeros(
                (batch_size, native_fut_len, dims["future_covariate_dim"]),
                dtype=np.float32,
            )
            future_covariates_native_mask = np.zeros(
                (batch_size, native_fut_len, dims["future_covariate_dim"]),
                dtype=np.float32,
            )
            historical_covariates_native_time_features = np.zeros(
                (batch_size, native_hist_len, native_hist_time_dim), dtype=np.float32
            )
            future_covariates_native_time_features = np.zeros(
                (batch_size, native_fut_len, native_fut_time_dim), dtype=np.float32
            )
            for idx, item in enumerate(items):
                if dims["historical_covariate_dim"] > 0 and native_hist_len > 0:
                    hist_native_len = item["historical_covariates_native"].shape[0]
                    if hist_native_len > 0:
                        historical_covariates_native[idx, :hist_native_len] = (
                            _pad_last_dim(
                                item["historical_covariates_native"],
                                dims["historical_covariate_dim"],
                            )
                        )
                        historical_covariates_native_mask[idx, :hist_native_len] = (
                            _pad_last_dim(
                                item["historical_covariates_native_mask"],
                                dims["historical_covariate_dim"],
                            )
                        )
                        if native_hist_time_dim > 0:
                            historical_covariates_native_time_features[
                                idx, :hist_native_len
                            ] = _pad_last_dim(
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
                        future_covariates_native_mask[idx, :fut_native_len] = (
                            _pad_last_dim(
                                item["future_covariates_native_mask"],
                                dims["future_covariate_dim"],
                            )
                        )
                        if native_fut_time_dim > 0:
                            future_covariates_native_time_features[
                                idx, :fut_native_len
                            ] = _pad_last_dim(
                                item["future_covariates_native_time_features"],
                                native_fut_time_dim,
                            )
            batch.update(
                {
                    "historical_covariates_native": torch.from_numpy(
                        historical_covariates_native
                    ),
                    "historical_covariates_native_mask": torch.from_numpy(
                        historical_covariates_native_mask
                    ),
                    "future_covariates_native": torch.from_numpy(
                        future_covariates_native
                    ),
                    "future_covariates_native_mask": torch.from_numpy(
                        future_covariates_native_mask
                    ),
                    "historical_covariates_native_time_features": torch.from_numpy(
                        historical_covariates_native_time_features
                    ),
                    "future_covariates_native_time_features": torch.from_numpy(
                        future_covariates_native_time_features
                    ),
                }
            )
        if self.metadata_mode == "full":
            batch["metadata"] = metadata
        return batch
