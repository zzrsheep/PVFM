from __future__ import annotations

from collections import Counter

import hashlib

import json

import math

from numbers import Integral

from pathlib import Path

import random

from pvfm.data.window_shapes import DynamicPoolUnavailable

from pvfm.data.regions import _stable_probability_caps

from pvfm.data.temporal_quality_index import fingerprint

from pvfm.data.regions import legal_count_probabilities

from pvfm.data.regions import region_cutoff_counts

PROTOCOL = "temporal_average_supply_audit_v1"


SHAPE_KEYS = (
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


def shape_config(sampler):
    """Bind calibration to the formal temporal shape distribution."""
    return {key: getattr(sampler, key) for key in SHAPE_KEYS}


def _tv(a, b):
    return sum(abs(a.get(r, 0) - b.get(r, 0)) for r in set(a) | set(b)) / 2


def build_or_load_average_supply(
    dataset,
    shape_sampler,
    directory,
    *,
    samples_per_source=5000,
    calibration_seed=20260913,
    alpha=0.25,
    max_oversampling=20.0,
    max_region_probability=0.15,
    log=lambda _: None,
):
    if (
        not isinstance(samples_per_source, int)
        or isinstance(samples_per_source, bool)
        or samples_per_source < 2
    ):
        raise ValueError("samples_per_source must be an integer >= 2")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not math.isfinite(max_oversampling) or max_oversampling < 1:
        raise ValueError("max_oversampling must be finite and >= 1")
    if not math.isfinite(max_region_probability) or not 0 < max_region_probability <= 1:
        raise ValueError("max_region_probability must be in (0, 1]")
    quality = dataset.ensure_quality_index()
    groups = {}
    for row in dataset.pool_rows:
        groups.setdefault((row["resolution"], row["cache"]), set()).add(row["region"])
    if not groups:
        raise ValueError("Average-supply calibration requires at least one source")
    config = {
        "samples_per_source": samples_per_source,
        "calibration_seed": int(calibration_seed),
        "shape_config": shape_config(shape_sampler),
        "policy": {
            "alpha": alpha,
            "max_oversampling": max_oversampling,
            "max_region_probability": max_region_probability,
        },
        "groups": [
            {"resolution": res, "cache": cache, "regions": sorted(regions)}
            for (res, cache), regions in sorted(groups.items())
        ],
        "reference_budget_definition": "sum of mean legal station/cutoff counts per source; no nominal-task or train-step budget",
        "infeasible_action": "fail; never resample source/CH, drop guards or relax caps",
    }
    binding = {
        "pool_fingerprint": dataset.pool_metadata["pool_fingerprint"],
        "quality_fingerprint": quality.fingerprint,
        "config_fingerprint": fingerprint(config),
    }
    path = Path(directory) / "profile.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        claimed = payload.pop("fingerprint", None)
        if payload.get("protocol") != PROTOCOL or fingerprint(payload) != claimed:
            raise ValueError("Average-supply profile protocol/fingerprint mismatch")
        if payload.get("binding") != binding:
            raise ValueError(
                "Average-supply source/QC/C/H/policy config changed; use a NEW profile directory"
            )
        log(f"AVERAGE_SUPPLY_REUSED path={path} fingerprint={claimed}")
        return {**payload, "fingerprint": claimed}, True
    # Reserve exclusively; partial or existing directories are never replaced.
    path.parent.mkdir(parents=True, exist_ok=False)
    sources = []
    for source_index, ((res, cache), regions) in enumerate(sorted(groups.items())):
        rng = random.Random(int(calibration_seed) + source_index * 1000003)
        sums, squares, hits, first_half = Counter(), Counter(), Counter(), Counter()
        trace = hashlib.sha256()
        for index in range(samples_per_source):
            c, h = shape_sampler._sample_shape(rng, res)
            counts = region_cutoff_counts(quality.query(res, cache, c, h))
            if not sum(counts.values()):
                raise DynamicPoolUnavailable(
                    f"calibration no legal cutoff: {res}/{cache} C={c} H={h}; no resampling"
                )
            trace.update(f"{c}:{h}\n".encode())
            sums.update(counts)
            squares.update({r: n * n for r, n in counts.items()})
            hits.update({r: int(n > 0) for r, n in counts.items()})
            if index < samples_per_source // 2:
                first_half.update(counts)
            if index == 0 or (index + 1) % 1000 == 0:
                log(
                    f"AVERAGE_SUPPLY source={res}/{cache} shapes={index+1}/{samples_per_source}"
                )
        means = {r: sums[r] / samples_per_source for r in sorted(regions)}
        errors = {
            r: math.sqrt(
                max(0.0, squares[r] - sums[r] * sums[r] / samples_per_source)
                / (samples_per_source - 1)
                / samples_per_source
            )
            for r in sorted(regions)
        }
        # Scaling every count by the same calibration sample count cancels in
        # both inverse-count normalized weights and the density cap.
        probs, caps = legal_count_probabilities(sums, **config["policy"])
        second_half = {r: sums[r] - first_half[r] for r in regions}
        split_half = None
        try:
            a, _ = legal_count_probabilities(first_half, **config["policy"])
            b, _ = legal_count_probabilities(second_half, **config["policy"])
            split_half = _tv(a, b)
        except DynamicPoolUnavailable:
            pass  # Diagnostic only; the full calibration rule already passed.
        sources.append(
            {
                "resolution": res,
                "cache": cache,
                "mean_legal_cutoffs": means,
                "mean_standard_error": errors,
                "shape_available_fraction": {
                    r: hits[r] / samples_per_source for r in sorted(regions)
                },
                "reference_draw_budget": sum(means.values()),
                "base_probabilities": probs,
                "base_caps": caps,
                "calibration_shape_trace_sha256": trace.hexdigest(),
                "split_half_base_probability_TV": split_half,
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "binding": binding,
        "config": config,
        "sources": sources,
    }
    result = {**payload, "fingerprint": fingerprint(payload)}
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    log(f"AVERAGE_SUPPLY_FROZEN path={path} fingerprint={result['fingerprint']}")
    return result, False


class AverageSupplyPolicy:
    def __init__(self, profile):
        payload = {k: v for k, v in profile.items() if k != "fingerprint"}
        if profile.get("protocol") != PROTOCOL or fingerprint(payload) != profile.get(
            "fingerprint"
        ):
            raise ValueError("Invalid average-supply profile")
        self.fingerprint = profile["fingerprint"]
        self.profile = profile
        self.sources = {(s["resolution"], s["cache"]): s for s in profile["sources"]}
        self.policy = profile["config"]["policy"]

    def probabilities(self, resolution, cache, counts):
        source = self.sources[(resolution, cache)]
        if any(
            not isinstance(n, Integral) or isinstance(n, bool) or n < 0
            for n in counts.values()
        ):
            raise ValueError("Invalid current legal-window count")
        available = {r: n for r, n in counts.items() if n > 0}
        if not available:
            raise DynamicPoolUnavailable(f"no legal cutoff: {resolution}/{cache}")
        if set(available) - set(source["base_probabilities"]):
            raise DynamicPoolUnavailable(
                "calibration missed a now-feasible region; do not silently exclude it"
            )
        raw = {r: source["base_probabilities"][r] for r in sorted(available)}
        total = sum(raw.values())
        raw = {r: p / total for r, p in raw.items()}
        budget = source["reference_draw_budget"]
        caps = {
            r: min(
                self.policy["max_region_probability"],
                self.policy["max_oversampling"] * n / budget,
            )
            for r, n in available.items()
        }
        if sum(caps.values()) < 1 - 1e-12:
            raise DynamicPoolUnavailable(
                f"average-supply caps infeasible: source={resolution}/{cache} cap_sum={sum(caps.values())} "
                f"fixed_data_budget={budget} counts={available}; caps/source/CH are not relaxed"
            )
        result = _stable_probability_caps(raw, caps)
        return (
            result,
            caps,
            {
                "base_to_runtime_TV": _tv(source["base_probabilities"], result),
                "runtime_guard_active": any(raw[r] > caps[r] + 1e-12 for r in raw),
                "excluded_regions": len(source["base_probabilities"]) - len(available),
            },
        )
