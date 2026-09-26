from __future__ import annotations
import numpy as np
from pvfm.data.window_shapes import DynamicPoolUnavailable
from collections import Counter
import json
from pathlib import Path
from pvfm.data.temporal_average_supply import AverageSupplyPolicy, shape_config
from pvfm.data.temporal_pool_v1 import TemporalPoolSampler
from pvfm.data.temporal_quality_index import fingerprint
from pvfm.data.regions import region_cutoff_counts

POLICY = "average_supply_v1"


def read_average_supply_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    AverageSupplyPolicy(profile)
    if fingerprint(profile["config"]) != profile["binding"]["config_fingerprint"]:
        raise ValueError("Average-supply profile config fingerprint mismatch")
    return profile


class AverageSupplyTemporalSampler(TemporalPoolSampler):

    def __init__(
        self, *args, region_profile_path, average_max_oversampling=20.0, **kwargs
    ):
        TemporalPoolSampler.__init__(self, *args, **kwargs)
        profile = read_average_supply_profile(region_profile_path)
        binding = profile["binding"]
        if (
            binding["pool_fingerprint"] != self.pool_fingerprint
            or binding["quality_fingerprint"] != self.quality_index.fingerprint
        ):
            raise ValueError("Average-supply profile pool/QC fingerprint mismatch")
        shape = shape_config(self)
        if fingerprint(shape) != fingerprint(profile["config"]["shape_config"]):
            raise ValueError(
                "Average-supply profile C/H configuration differs from training"
            )
        expected_policy = {
            "alpha": self.region_balance_alpha,
            "max_region_probability": self.region_balance_max_prob,
            "max_oversampling": float(average_max_oversampling),
        }
        if expected_policy != profile["config"]["policy"]:
            raise ValueError(
                "Average-supply profile policy differs from training configuration"
            )
        expected = {key: set(value[0]) for key, value in self._regions.items()}
        sources = profile["sources"]
        actual = {
            (s["resolution"], s["cache"]): set(s["mean_legal_cutoffs"]) for s in sources
        }
        if actual != expected or len(actual) != len(sources):
            raise ValueError("Average-supply profile source/region mismatch")
        self.average_supply_policy = AverageSupplyPolicy(profile)
        self.region_profile = profile
        self.region_profile_fingerprint = profile["fingerprint"]
        self.max_window_oversampling = expected_policy["max_oversampling"]
        for source in sources:
            probs = source["base_probabilities"]
            if not probs or not set(probs).issubset(source["mean_legal_cutoffs"]):
                raise ValueError(
                    "Average-supply profile has invalid base-probability regions"
                )
            regions = sorted(probs)
            self._regions[source["resolution"], source["cache"]] = (
                regions,
                [probs[r] for r in regions],
            )
        self.region_sampling_policy = POLICY
        self.repeat_guard = Counter()
        self.sampler_config_fingerprint = fingerprint(
            {
                "base_config": self.sampler_config_fingerprint,
                "policy": POLICY,
                "region_profile_fingerprint": self.region_profile_fingerprint,
                "station_draw": "uniform_legal_cutoffs_without_replacement_within_batch",
            }
        )

    def region_probabilities_for_shape(self, resolution, cache, c, h, eligible=None):
        eligible = (
            eligible
            if eligible is not None
            else self.quality_index.query(resolution, cache, c, h)
        )
        counts = region_cutoff_counts(eligible)
        probs, _caps, info = self.average_supply_policy.probabilities(
            resolution, cache, counts
        )
        return (probs, counts, info["runtime_guard_active"])

    def _sample_schedule(self, rng):
        resolutions = list(self.source_probabilities)
        resolution = rng.choices(
            resolutions,
            weights=[self.source_probabilities[x] for x in resolutions],
            k=1,
        )[0]
        caches = sorted(
            {cache for res, cache, _region in self._groups if res == resolution}
        )
        weights = [self.cache_weights.get(cache, 1.0) for cache in caches]
        if not any(weights):
            weights = [1.0] * len(caches)
        cache = rng.choices(caches, weights=weights, k=1)[0]
        region_uniform = rng.random()
        c, h = self._sample_shape(rng, resolution)
        probs, counts, constrained = self.region_probabilities_for_shape(
            resolution, cache, c, h
        )
        self.repeat_guard["scheduled_batches"] += 1
        self.repeat_guard["capped_batches"] += int(constrained)
        for candidate in self._regions[resolution, cache][0]:
            if not counts.get(candidate, 0):
                self.excluded_region_counts[resolution, cache, candidate] += 1
        position = region_uniform * sum(probs.values())
        region = next(reversed(probs))
        for candidate, prob in probs.items():
            position -= prob
            if position < 0:
                region = candidate
                break
        return (resolution, cache, region, int(c), int(h))

    def _draw_batch(self, resolution, cache, region, c, h, rng):
        eligible = self.quality_index.query(resolution, cache, c, h)
        rows = eligible.by_region.get(region, [])
        if not rows:
            raise DynamicPoolUnavailable(
                f"{POLICY} no legal windows in {resolution}/{cache}/{region} C={c} H={h}"
            )
        intervals = np.concatenate(
            [np.arange(*eligible.offsets[row], dtype=np.int64) for row in rows]
        )
        cumulative = np.cumsum(
            eligible.stops[intervals] - eligible.starts[intervals] + 1
        )
        total = int(cumulative[-1])
        positions = (
            rng.sample(range(total), self.batch_size)
            if total >= self.batch_size
            else [rng.randrange(total) for _ in range(self.batch_size)]
        )
        self.repeat_guard["within_batch_duplicates"] += len(positions) - len(
            set(positions)
        )
        batch = []
        for position in positions:
            slot = int(np.searchsorted(cumulative, position, side="right"))
            interval = int(intervals[slot])
            row = int(eligible.station_ids[interval])
            previous = int(cumulative[slot - 1]) if slot else 0
            cutoff = int(eligible.starts[interval]) + position - previous
            reason = self.dataset.validity_reason(row, c, h, cutoff)
            if reason:
                self.rejections[reason] += 1
                raise DynamicPoolUnavailable(
                    f"{POLICY} final QC failed: {resolution}/{cache}/{region} station={row} cutoff={cutoff} C={c} H={h} reason={reason}"
                )
            batch.append((row, c, h, cutoff))
        return batch

    def state_dict(self):
        state = super().state_dict()
        state["repeat_guard"] = dict(self.repeat_guard)
        state["region_profile_fingerprint"] = self.region_profile_fingerprint
        return state

    def load_state_dict(self, state):
        if state.get("region_profile_fingerprint") != self.region_profile_fingerprint:
            raise ValueError("Temporal aligned resume region profile mismatch")
        super().load_state_dict(state)
        self.repeat_guard = Counter(state.get("repeat_guard", {}))

    def write_sampling_audit(self, directory):
        paths = super().write_sampling_audit(directory)
        path = Path(paths["json"])
        report = json.loads(path.read_text())
        report.update(
            {
                "region_profile": self.region_profile,
                "region_profile_fingerprint": self.region_profile_fingerprint,
                "repeat_guard": dict(self.repeat_guard),
                "station_draw": "uniform_legal_cutoffs_without_replacement_within_batch",
            }
        )
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return paths
