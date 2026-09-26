from __future__ import annotations

from collections import OrderedDict, defaultdict

import hashlib

import json

import os

from pathlib import Path

import numpy as np

from pvfm.data.cache_contract import _array_contract_reason, split_bounds

from pvfm.data.regions import canonical_region_key

PROTOCOL = "temporal_quality_intervals_v1"


ARRAY_NAMES = (
    "target",
    "target_mask",
    "history_covariates",
    "history_covariate_mask",
    "future_covariates",
    "future_covariate_mask",
    "time_features",
    "timestamps",
    "site_features",
)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _runs(good):
    changes = np.diff(np.r_[False, good, False].astype(np.int8))
    return np.column_stack(
        (np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
    ).tolist()


def _intersect(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        if max(a, c) < min(b, d):
            result.append([max(a, c), min(b, d)])
        if b < d:
            i += 1
        else:
            j += 1
    return result


def _bad_counts(mask, end):
    values = np.asarray(mask[:end])
    channels = int(np.prod(values.shape[1:])) if values.ndim > 1 else 1
    counts = (
        np.sum(~(values.reshape(end, channels) > 0), axis=1)
        if channels
        else np.zeros(end, dtype=np.int64)
    )
    starts = np.r_[0, np.flatnonzero(np.diff(counts)) + 1]
    ends = np.r_[starts[1:], end]
    return {
        "channels": channels,
        "runs": [
            [int(a), int(b), int(counts[a])]
            for a, b in zip(starts, ends)
            if b > a and counts[a]
        ],
    }


def _good_runs(bad, end):
    runs = []
    cursor = 0
    for start, stop, _count in bad["runs"]:
        if cursor < start:
            runs.append([cursor, start])
        cursor = stop
    if cursor < end:
        runs.append([cursor, end])
    return runs


def _cadence_runs(timestamps, resolution):
    times = np.asarray(timestamps).astype("datetime64[ns]").astype(np.int64)
    if not len(times):
        return []
    step = {"1h": 3600, "15min": 900}[resolution] * 1_000_000_000
    good = times != np.iinfo(np.int64).min
    breaks = np.diff(times) != step
    starts = np.flatnonzero(good & np.r_[True, ~good[:-1] | breaks])
    ends = np.flatnonzero(good & np.r_[~good[1:] | breaks, True]) + 1
    return np.column_stack((starts, ends)).tolist()


class _SparsePrefix:
    def __init__(self, bad):
        values = np.asarray(bad["runs"], dtype=np.int64).reshape(-1, 3)
        self.starts, self.ends, self.counts = values.T
        self.prefix = np.r_[0, np.cumsum((self.ends - self.starts) * self.counts)]

    def at(self, times):
        times = np.asarray(times, dtype=np.int64)
        if not len(self.starts):
            return np.zeros_like(times)
        index = np.searchsorted(self.starts, times, side="right") - 1
        valid_index = np.maximum(index, 0)
        value = (
            self.prefix[valid_index]
            + np.maximum(
                0, np.minimum(times, self.ends[valid_index]) - self.starts[valid_index]
            )
            * self.counts[valid_index]
        )
        return np.where(index >= 0, value, 0)


def source_signature(dataset):
    """Stat all referenced array files; hash small controls and memory fixtures.

    This is an immutable-file identity check, not a full tensor byte audit.
    """
    files = {}
    memory = {}
    children = dataset.base_dataset.children
    cache_names = dataset.base_dataset.source_cache_names_by_child
    for row in dataset.pool_rows:
        child_index, station_index = int(row["child_index"]), int(row["station_index"])
        child = children[child_index]
        task = dataset.base_dataset.task_datasets[child_index]
        allowed = getattr(child, "_allowed_station_indices", None)
        if (
            str(cache_names[child_index]) != row["cache"]
            or str(task.task_spec.resolution) != row["resolution"]
            or str(child._station_key(station_index)) != row["station_key"]
            or (allowed is not None and station_index not in allowed)
        ):
            raise ValueError(
                "Temporal pool/source station mapping or manifest filter mismatch; use a matching pool"
            )
        records = getattr(child, "station_records", None)
        if (
            records is not None
            and canonical_region_key(records[station_index]) != row["region"]
        ):
            raise ValueError(
                "Temporal pool/source region mapping mismatch; use a matching pool"
            )
        root = Path(str(getattr(child, "binary_dataset_dir", "")))
        view = root / "views" / row["resolution"] / row["station_key"]
        if root.is_dir():
            for path in [
                root / "metadata.json",
                *[view / f"{name}.npy" for name in ARRAY_NAMES],
            ]:
                key = str(path.resolve())
                if key not in files:
                    info = path.stat()
                    files[key] = [info.st_size, info.st_mtime_ns]
                    if path.name == "metadata.json":
                        files[key].append(hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            arrays = child._station_arrays(row["resolution"], row["station_index"])
            key = (row["child_index"], row["station_index"])
            memory[str(key)] = {
                name: {
                    "shape": list(np.asarray(arrays[name]).shape),
                    "dtype": str(arrays[name].dtype),
                    "sha256": hashlib.sha256(
                        np.asarray(arrays[name]).tobytes()
                    ).hexdigest(),
                }
                for name in ARRAY_NAMES
            }
    return {"files": files, "memory": memory}


def _describe(dataset):
    rows = []
    for row in dataset.pool_rows:
        arrays = dataset._source_dataset(row["child_index"])._station_arrays(
            row["resolution"], row["station_index"]
        )
        reason = _array_contract_reason(arrays)
        if reason or len(arrays["target"]) != row["length"]:
            raise ValueError(
                f"Temporal quality array contract: station={row['station_key']} reason={reason or 'length_changed'}"
            )
        start, end = split_bounds(row["length"], 1, 1, "train")
        target = np.asarray(arrays["target_mask"][start:end])
        target_good = (
            np.all(target.reshape(end - start, -1) > 0, axis=1)
            if end > start
            else np.zeros(0, dtype=bool)
        )
        segments = _intersect(
            _intersect(row["segments"], [[start, end]]),
            _cadence_runs(arrays["timestamps"][:end], row["resolution"]),
        )
        rows.append(
            {
                "train_end": end,
                "target_runs": _intersect(_runs(target_good), segments),
                "history": _bad_counts(arrays["history_covariate_mask"], end),
                "future": _bad_counts(arrays["future_covariate_mask"], end),
            }
        )
    return rows


def build_or_load_quality_index(dataset, directory=None):
    """A new sibling sidecar, never an edit to the temporal pool/cache."""
    signature = source_signature(dataset)
    pool_fp = dataset.pool_metadata["pool_fingerprint"]
    path = Path(directory).resolve() / "metadata.json" if directory else None
    if path is not None and path.is_file():
        payload = json.loads(path.read_text())
        claimed = payload.pop("fingerprint", None)
        if payload.get("protocol") != PROTOCOL or claimed != fingerprint(payload):
            raise ValueError("Temporal quality index protocol/fingerprint mismatch")
        if (
            payload.get("pool_fingerprint") != pool_fp
            or payload.get("source_signature") != signature
        ):
            raise ValueError(
                "Temporal quality index source mismatch; use a new quality directory (no overwrite)"
            )
        if len(payload["rows"]) != len(dataset.pool_rows):
            raise ValueError("Temporal quality index station count mismatch")
        return TemporalQualityIndex(dataset, payload["rows"], claimed)
    if path is not None:
        # Exclusive directory reservation also prevents competing DDP writers.
        # The launcher prebuilds this outside torchrun.
        try:
            path.parent.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing incomplete/existing quality directory: {path.parent}; prebuild once in a new directory"
            ) from exc
    rows = _describe(dataset)
    if signature != source_signature(dataset):
        raise ValueError("Source arrays changed while building temporal quality index")
    payload = {
        "protocol": PROTOCOL,
        "pool_fingerprint": pool_fp,
        "source_signature": signature,
        "rows": rows,
    }
    digest = fingerprint(payload)
    if path is not None:
        temporary = path.parent / f"metadata.json.tmp.{os.getpid()}"
        with temporary.open("x") as handle:
            json.dump({**payload, "fingerprint": digest}, handle, separators=(",", ":"))
            handle.write("\n")
        os.rename(temporary, path)
    return TemporalQualityIndex(dataset, rows, digest)


def _strict_pairs(row, history_ratio, future_ratio):
    """Each pair encodes max(hist_start+C, future_start) <= cutoff <= min(hist_end, future_end-H)."""
    hist = (
        _good_runs(row["history"], row["train_end"])
        if history_ratio > 0
        else [[0, row["train_end"]]]
    )
    future = (
        _good_runs(row["future"], row["train_end"])
        if future_ratio > 0
        else [[0, row["train_end"]]]
    )
    pairs = []
    for pv_run in row["target_runs"]:
        left, right = _intersect(hist, [pv_run]), _intersect(future, [pv_run])
        i = j = 0
        while i < len(left) and j < len(right):
            a, b = left[i]
            c, d = right[j]
            # Touching intervals can support a cutoff with different past and future masks.
            if max(a, c) <= min(b, d):
                pairs.append((a, b, c, d))
            if b < d:
                i += 1
            else:
                j += 1
    return pairs


class EligibleCutoffs:
    def __init__(self, station_ids, starts, stops, pool_rows):
        # Inclusive integer cutoff intervals, sorted by station then origin.
        self.station_ids = np.asarray(station_ids, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.stops = np.asarray(stops, dtype=np.int64)
        self.rows, offsets = np.unique(self.station_ids, return_index=True)
        self.offsets = dict(
            zip(
                self.rows.tolist(),
                zip(offsets.tolist(), np.r_[offsets[1:], len(starts)].tolist()),
            )
        )
        self.cumulative = np.cumsum(self.stops - self.starts + 1)
        self.by_region = defaultdict(list)
        for row in self.rows:
            self.by_region[pool_rows[row]["region"]].append(int(row))

    @property
    def nbytes(self):
        # Includes a conservative allowance for Python group/offset objects.
        return (
            sum(
                a.nbytes
                for a in (
                    self.station_ids,
                    self.starts,
                    self.stops,
                    self.rows,
                    self.cumulative,
                )
            )
            + len(self.rows) * 256
        )

    def draw(self, row, rng):
        start, stop = self.offsets[int(row)]
        baseline = int(self.cumulative[start - 1]) if start else 0
        position = baseline + rng.randrange(int(self.cumulative[stop - 1]) - baseline)
        index = int(np.searchsorted(self.cumulative, position, side="right"))
        previous = int(self.cumulative[index - 1]) if index else 0
        return int(self.starts[index]) + position - previous


class TemporalQualityIndex:
    def __init__(
        self,
        dataset,
        rows,
        digest,
        *,
        max_cached_shapes=64,
        max_cache_bytes=64 * 1024**2,
    ):
        self.rows = rows
        self.pool_rows = dataset.pool_rows
        self.fingerprint = digest
        self.max_cached_shapes = max_cached_shapes
        self.max_cache_bytes = max_cache_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.hits = self.misses = 0
        groups = defaultdict(list)
        self.fractional = defaultdict(list)
        self.ratios = []
        self.prefixes = {}
        for index, (pool_row, quality) in enumerate(zip(self.pool_rows, rows)):
            task = dataset.task_datasets[pool_row["child_index"]]
            rh, rf = float(task.min_history_covariate_valid_ratio), float(
                task.min_future_covariate_valid_ratio
            )
            if not (0 <= rh <= 1 and 0 <= rf <= 1):
                raise ValueError(
                    "Temporal quality index requires covariate valid ratios within [0, 1]"
                )
            self.ratios.append((rh, rf))
            key = (pool_row["resolution"], pool_row["cache"])
            if rh in (0, 1) and rf in (0, 1):
                groups[key].extend(
                    (index, *pair) for pair in _strict_pairs(quality, rh, rf)
                )
            else:
                self.fractional[key].append(index)
                self.prefixes[index] = (
                    _SparsePrefix(quality["history"]),
                    _SparsePrefix(quality["future"]),
                )
        self.groups = {
            key: np.asarray(value, dtype=np.int64).reshape(-1, 5)
            for key, value in groups.items()
        }

    def _fractional_cutoffs(self, index, c, h):
        row = self.rows[index]
        chunks = [
            np.arange(a + c, b - h + 1, dtype=np.int64)
            for a, b in row["target_runs"]
            if b - a >= c + h
        ]
        if not chunks:
            return [], []
        cutoffs = np.concatenate(chunks)
        good = np.ones(len(cutoffs), dtype=bool)
        for prefix, phase, length, ratio in zip(
            self.prefixes[index], ("history", "future"), (c, h), self.ratios[index]
        ):
            channels = row[phase]["channels"]
            if not channels or ratio == 0:
                continue
            a, b = (
                (cutoffs - c, cutoffs) if phase == "history" else (cutoffs, cutoffs + h)
            )
            total = length * channels
            good &= (total - (prefix.at(b) - prefix.at(a))) / total >= ratio
        valid = cutoffs[good]
        if not len(valid):
            return [], []
        breaks = np.flatnonzero(np.diff(valid) != 1) + 1
        return valid[np.r_[0, breaks]], valid[np.r_[breaks - 1, len(valid) - 1]]

    def query(self, resolution, cache, c, h):
        c, h = int(c), int(h)
        if c < 1 or h < 1:
            raise ValueError("C/H must be positive")
        key = (resolution, cache, c, h)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        pairs = self.groups.get((resolution, cache), np.empty((0, 5), dtype=np.int64))
        lo = np.maximum(pairs[:, 1] + c, pairs[:, 3])
        hi = np.minimum(pairs[:, 2], pairs[:, 4] - h)
        good = hi >= lo
        ids, starts, stops = pairs[good, 0], lo[good], hi[good]
        for index in self.fractional.get((resolution, cache), ()):
            left, right = self._fractional_cutoffs(index, c, h)
            ids = np.r_[ids, np.full(len(left), index, dtype=np.int64)]
            starts, stops = np.r_[starts, left], np.r_[stops, right]
        if self.fractional.get((resolution, cache)):
            order = np.lexsort((starts, ids))
            ids, starts, stops = ids[order], starts[order], stops[order]
        result = EligibleCutoffs(ids, starts, stops, self.pool_rows)
        if result.nbytes <= self.max_cache_bytes and self.max_cached_shapes > 0:
            while self.cache and (
                len(self.cache) >= self.max_cached_shapes
                or self.cache_bytes + result.nbytes > self.max_cache_bytes
            ):
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= removed.nbytes
            self.cache[key] = result
            self.cache_bytes += result.nbytes
        return result
