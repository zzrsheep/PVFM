"""Portable pre-aligned tensor shards, without private cache dependencies.

The manifest fixes split, station cohort, row order and shard checksums.
Raw observations and original private cache readers are not distributed.
"""

from collections import OrderedDict
import bisect
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .model import BACKBONE_INPUTS, SATELLITE_INPUTS

TENSOR_KEYS = (
    *BACKBONE_INPUTS,
    *SATELLITE_INPUTS,
    "future_target",
    "future_target_mask",
)
HOUR_NS = 3_600_000_000_000


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_batch(size=2, seed=0):
    """Artificial smoke inputs only; never used as reported evaluation data."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    origin = torch.full((size,), 1_700_000_000_000_000_000, dtype=torch.int64)
    time = torch.zeros(size, 16, 5)
    time[:, :, 0] = -0.5
    time[:, :, 1] = torch.arange(16).float() / 23.0 - 0.5
    time[:, :, -1] = (200.0 - 1.0) / 365.0 - 0.5
    future_time = time[:, :4].clone()
    future_time[:, :, 1] = torch.arange(16, 20).float() / 23.0 - 0.5
    coord = torch.linspace(-0.01, 0.01, 8)
    grid = torch.stack(torch.meshgrid(coord, coord, indexing="ij"), -1).reshape(
        1, 64, 2
    )
    return {
        "past_target": rand(size, 16, 1),
        "past_observed_mask": torch.ones(size, 16, 1),
        "future_covariates": rand(size, 4, 6),
        "future_covariates_mask": torch.ones(size, 4, 6),
        "past_time_features": time,
        "future_time_features": future_time,
        "static_features": torch.tensor([16.0, 4.0, 4.0, 4.0]).repeat(size, 1),
        "site_features": torch.tensor([32.0, 117.0, 0.0, 0.0, 8.0]).repeat(size, 1),
        "satellite_history": rand(size, 16, 4, 64, 64) * 0.3,
        "satellite_frame_mask": torch.ones(size, 16, dtype=torch.bool),
        "satellite_patch_coords": grid.repeat(size, 1, 1),
        "satellite_frame_time_ns": origin[:, None]
        + torch.arange(-16, 0, dtype=torch.int64)[None, :] * HOUR_NS,
        "forecast_origin_time_ns": origin,
        "future_target": rand(size, 4, 1),
        "future_target_mask": torch.ones(size, 4, 1),
        "station_id": [f"site_{i:03d}" for i in range(size)],
    }


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in (*BACKBONE_INPUTS, *SATELLITE_INPUTS)}


class WindowDataset(Dataset):
    """Bounded two-shard cache; no interpolation, resampling or window selection."""

    def __init__(self, manifest, expected_split):
        self.path = Path(manifest).resolve()
        self.metadata = json.loads(self.path.read_text())
        if self.metadata.get("format") != "satellite_windows_v1":
            raise ValueError("Expected satellite_windows_v1 manifest")
        if self.metadata.get("split") != expected_split:
            raise ValueError("Train/validation/test split mismatch")
        if self.metadata.get("cohort") not in ("seen", "unseen"):
            raise ValueError("Explicit seen/unseen cohort required")
        if expected_split != "test" and self.metadata["cohort"] != "seen":
            raise ValueError("Unseen stations are test-only")
        if (
            self.metadata.get("context_steps") != 16
            or self.metadata.get("horizon_steps") != 4
            or self.metadata.get("resolution_minutes") != 60
        ):
            raise ValueError("Expected hourly C16/H4 data")
        self.shards = self.metadata["shards"]
        self.ends = []
        count = 0
        self.cache = OrderedDict()
        for shard in self.shards:
            path = (self.path.parent / shard["file"]).resolve()
            if not path.is_relative_to(self.path.parent) or path.suffix != ".npz":
                raise ValueError(
                    "Shard path must be a relative NPZ within manifest directory"
                )
            if file_hash(path) != shard["sha256"]:
                raise ValueError("Shard checksum mismatch")
            if int(shard["samples"]) <= 0:
                raise ValueError("Empty shard")
            count += int(shard["samples"])
            self.ends.append(count)
        if not count:
            raise ValueError("Empty data manifest")
        self.fingerprint = file_hash(self.path)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[shard_index - 1] if shard_index else 0)
        if shard_index not in self.cache:
            row = self.shards[shard_index]
            with np.load(self.path.parent / row["file"], allow_pickle=False) as data:
                arrays = {key: data[key] for key in (*TENSOR_KEYS, "station_id")}
            if any(len(v) != row["samples"] for v in arrays.values()):
                raise ValueError("Shard row count mismatch")
            self.cache[shard_index] = arrays
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard_index)
        arrays = self.cache[shard_index]
        result = {
            key: torch.from_numpy(np.asarray(arrays[key][local]).copy())
            for key in TENSOR_KEYS
        }
        result["station_id"] = str(arrays["station_id"][local])
        return result
