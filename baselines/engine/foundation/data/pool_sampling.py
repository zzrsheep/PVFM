"""Optional pool-local draw state and lightweight sampling audits.

The legacy samplers deliberately keep their replacement path in place.  This
module is used only when a caller explicitly selects ``shuffle_cycle`` or
requests an audit.  It never owns a second copy of a pool's full index list.
"""
import csv
import copy
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


def normalize_pool_draw_mode(value):
    mode = str(value or "replacement").strip().lower()
    if mode not in {"replacement", "shuffle_cycle"}:
        raise ValueError("pool_draw_mode must be 'replacement' or 'shuffle_cycle'.")
    return mode


class ShuffleCyclePoolDrawer:
    """Draw consecutive items from an in-place shuffled pool.

    ``candidates`` remains the sampler-owned list.  The state is just a
    cursor and a completed-cycle count; no duplicate full index is retained.
    """

    def __init__(self):
        self._state = {}

    def _shuffle(self, key, candidates, rng):
        rng.shuffle(candidates)
        state = self._state.setdefault(
            key, {"cursor": 0, "cycles_completed": 0, "cycles_started": 0, "needs_shuffle": False}
        )
        state["cursor"] = 0
        state["needs_shuffle"] = False
        state["cycles_started"] += 1
        return state

    def draw(self, key, candidates, batch_size, rng, shuffle=True):
        """Return a batch and metadata without duplicating normal-pool items.

        At a cycle boundary, the new permutation is adjusted in place so its
        prefix excludes the tail already emitted in this batch.  This keeps a
        normal pool's batch duplicate-free even when it spans two cycles.
        """
        if not candidates:
            return [], {"cycles_completed": 0, "forced_duplicates": 0}
        batch_size = int(batch_size)
        state = self._state.get(key)
        if state is None:
            state = self._state.setdefault(
                key, {"cursor": 0, "cycles_completed": 0, "cycles_started": 0, "needs_shuffle": False}
            )
            if shuffle:
                self._shuffle(key, candidates, rng)
        batch = []
        completed_before = int(state["cycles_completed"])
        while len(batch) < batch_size:
            cursor = int(state["cursor"])
            if cursor >= len(candidates):
                if not state.get("needs_shuffle", False):
                    state["cycles_completed"] += 1
                if shuffle:
                    self._shuffle(key, candidates, rng)
                else:
                    state["cursor"] = 0
                    state["needs_shuffle"] = False
                cursor = 0
            remaining = len(candidates) - cursor
            need = batch_size - len(batch)
            # A normal pool can cross one cycle boundary.  Place items absent
            # from the already emitted tail at the next cycle's prefix before
            # taking them, so the resulting batch stays duplicate-free.
            if need > remaining and len(candidates) >= batch_size:
                batch.extend(candidates[cursor:])
                state["cursor"] = len(candidates)
                state["cycles_completed"] += 1
                if shuffle:
                    self._shuffle(key, candidates, rng)
                else:
                    state["cursor"] = 0
                tail = set(batch)
                prefix_need = batch_size - len(batch)
                write = 0
                for read in range(len(candidates)):
                    if candidates[read] in tail:
                        continue
                    if write != read:
                        candidates[write], candidates[read] = candidates[read], candidates[write]
                    write += 1
                    if write >= prefix_need:
                        break
                batch.extend(candidates[:prefix_need])
                state["cursor"] = prefix_need
                break
            take = min(remaining, need)
            if take:
                batch.extend(candidates[cursor:cursor + take])
                state["cursor"] = cursor + take
                if state["cursor"] >= len(candidates):
                    state["cycles_completed"] += 1
                    state["needs_shuffle"] = True
            if len(batch) >= batch_size:
                break
        forced_duplicates = 0
        if len(candidates) < batch_size:
            forced_duplicates = len(batch) - len(set(batch))
        return batch, {
            "cycles_completed": int(state["cycles_completed"]) - completed_before,
            "forced_duplicates": int(forced_duplicates),
        }

    def state_for(self, key):
        return dict(self._state.get(key, {}))

    def state_dict(self):
        """Return cursor/cycle metadata; pool permutations stay sampler-owned."""
        return {
            "version": 1,
            "pool_states": copy.deepcopy(self._state),
        }

    def load_state_dict(self, state):
        if int(state.get("version", 0)) != 1:
            raise ValueError(f"Unsupported shuffle-cycle drawer state version: {state.get('version')}")
        pool_states = state.get("pool_states")
        if not isinstance(pool_states, dict):
            raise ValueError("Shuffle-cycle drawer state is missing pool_states.")
        restored = {}
        required = {"cursor", "cycles_completed", "cycles_started", "needs_shuffle"}
        for key, value in pool_states.items():
            if not isinstance(value, dict) or not required.issubset(value):
                raise ValueError(f"Invalid shuffle-cycle state for pool {key!r}.")
            restored[key] = {
                "cursor": int(value["cursor"]),
                "cycles_completed": int(value["cycles_completed"]),
                "cycles_started": int(value["cycles_started"]),
                "needs_shuffle": bool(value["needs_shuffle"]),
            }
        self._state = restored


class SamplingAudit:
    """Exact draw/unique accounting plus bounded condition reservoirs."""

    def __init__(self, dataset, mode, trace_batches=16, condition_samples_per_pool=256):
        self.dataset = dataset
        self.mode = normalize_pool_draw_mode(mode)
        self.trace_batches = max(0, int(trace_batches or 0))
        self.condition_samples_per_pool = max(0, int(condition_samples_per_pool or 0))
        self._seen = bytearray(len(dataset))
        self._pools = defaultdict(lambda: {
            "batch_draws": 0,
            "window_draws": 0,
            "unique_windows": 0,
            "duplicate_draws": 0,
            "forced_small_pool_duplicates": 0,
            "completed_cycles": 0,
            "station_indices": set(),
            "condition_samples": [],
        })
        self._batch_hashes = []
        self._global_batches = 0

    @staticmethod
    def _key_text(key):
        return "|".join(str(item) for item in key)

    def record(self, key, batch, forced_duplicates=0, completed_cycles=0):
        state = self._pools[key]
        state["batch_draws"] += 1
        state["window_draws"] += len(batch)
        state["forced_small_pool_duplicates"] += int(forced_duplicates)
        state["completed_cycles"] += int(completed_cycles)
        for sample_index in batch:
            sample_index = int(sample_index)
            if not self._seen[sample_index]:
                self._seen[sample_index] = 1
                state["unique_windows"] += 1
                if len(state["condition_samples"]) < self.condition_samples_per_pool:
                    state["condition_samples"].append(sample_index)
            else:
                state["duplicate_draws"] += 1
            try:
                state["station_indices"].add(int(self.dataset.index[sample_index][2]))
            except (IndexError, TypeError, ValueError):
                pass
        if self._global_batches < self.trace_batches:
            digest = hashlib.sha256(",".join(str(int(item)) for item in batch).encode("ascii")).hexdigest()
            self._batch_hashes.append({"global_batch_index": self._global_batches, "sha256": digest})
        self._global_batches += 1

    def _condition_values(self, sample_index):
        """Read only reservoir windows; unsupported datasets remain explicit."""
        try:
            source = self.dataset
            source_index = int(sample_index)
            while hasattr(source, "children"):
                child_id, local_index, _station_index, _sample_start = source.index[source_index]
                source = source.children[int(child_id)]
                source_index = int(local_index)
            dataset_idx, _local_idx, station_index, sample_start = source.index[source_index]
            task_dataset = source.task_datasets[int(dataset_idx)]
            resolution = str(task_dataset.task_spec.resolution)
            arrays = source._station_arrays(resolution, int(station_index))
            future_start = int(sample_start) + int(task_dataset.seq_len)
            future_end = future_start + int(task_dataset.pred_len)
            timestamps = arrays["timestamps"][future_start:future_end]
            timestamp = timestamps[0] if len(timestamps) else None
            if isinstance(timestamp, np.datetime64):
                month = int(np.datetime_as_string(timestamp, unit="D")[5:7])
            else:
                month = None
            target = np.asarray(arrays["target"][future_start:future_end], dtype=np.float64).reshape(-1)
            future_cov = np.asarray(arrays["future_covariates"][future_start:future_end], dtype=np.float64)
            meta = source._station_meta(resolution, int(station_index))
            names = [str(item).lower() for item in meta.get("future_covariate_cols", [])]
            def _mean_named(token):
                for idx, name in enumerate(names):
                    if token in name and idx < future_cov.shape[-1]:
                        return float(np.nanmean(future_cov[:, idx]))
                return None
            return {
                "month": month,
                "shortwave_mean": _mean_named("shortwave"),
                "cloud_mean": _mean_named("cloud"),
                "future_pv_ramp": float(np.nanmax(np.abs(np.diff(target)))) if target.size > 1 else 0.0,
                "future_pv_peak": float(np.nanmax(target)) if target.size else None,
            }
        except Exception:
            return None

    @staticmethod
    def _tertiles(values):
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        if not values:
            return {"valid_samples": 0, "low": 0, "mid": 0, "high": 0}
        low_cut, high_cut = np.quantile(np.asarray(values, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
        result = {"valid_samples": len(values), "low": 0, "mid": 0, "high": 0}
        for value in values:
            result["low" if value <= low_cut else "mid" if value <= high_cut else "high"] += 1
        return result

    def _condition_coverage(self, sample_indices):
        rows = [self._condition_values(index) for index in sample_indices]
        rows = [row for row in rows if row is not None]
        months = defaultdict(int)
        for row in rows:
            if row.get("month") is not None:
                months[str(row["month"])] += 1
        return {
            "reservoir_samples": len(sample_indices),
            "readable_samples": len(rows),
            "month": dict(sorted(months.items(), key=lambda item: int(item[0]))),
            "future_nwp_shortwave": self._tertiles([row.get("shortwave_mean") for row in rows]),
            "future_nwp_cloud": self._tertiles([row.get("cloud_mean") for row in rows]),
            "future_pv_ramp": self._tertiles([row.get("future_pv_ramp") for row in rows]),
            "future_pv_peak": self._tertiles([row.get("future_pv_peak") for row in rows]),
        }

    def write(self, output_dir, rank=0, world_size=1):
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        for key in sorted(self._pools, key=self._key_text):
            state = self._pools[key]
            draws = int(state["window_draws"])
            unique = int(state["unique_windows"])
            rows.append({
                "pool_key": self._key_text(key),
                "resolution": key[0] if len(key) == 5 else "",
                "task_name": key[1] if len(key) == 5 else key[0] if len(key) == 1 else "",
                "seq_len": key[2] if len(key) == 5 else "",
                "pred_len": key[3] if len(key) == 5 else "",
                "canonical_region": key[4] if len(key) == 5 else key[-1],
                "batch_draws": int(state["batch_draws"]),
                "window_draws": draws,
                "unique_windows": unique,
                "duplicate_draws": int(state["duplicate_draws"]),
                "duplicate_rate": (float(state["duplicate_draws"]) / draws) if draws else 0.0,
                "completed_cycles": int(state["completed_cycles"]),
                "unique_stations": len(state["station_indices"]),
                "forced_small_pool_duplicates": int(state["forced_small_pool_duplicates"]),
            })
        csv_path = os.path.join(output_dir, "sampling_audit.csv")
        fields = list(rows[0]) if rows else ["pool_key"]
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        payload = {
            "pool_draw_mode": self.mode,
            "rank": int(rank),
            "world_size": int(world_size),
            "global_batches_observed": int(self._global_batches),
            "global_batch_hashes": self._batch_hashes,
            "pools": [
                dict(row, condition_coverage=self._condition_coverage(self._pools[key]["condition_samples"]))
                for key, row in zip(sorted(self._pools, key=self._key_text), rows)
            ],
        }
        json_path = os.path.join(output_dir, "sampling_audit.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {"json": json_path, "csv": csv_path}
