"""Region identities and probability rules used by average-supply sampling."""

from __future__ import annotations

import math
from collections import Counter
from numbers import Integral
from pvfm.data.cache_contract import DynamicPoolUnavailable

_CHINA_REGION_MAP = {
    "安徽": "Anhui",
    "广西": "Guangxi",
    "河北": "Hebei",
    "新疆": "Xinjiang",
    "广东": "Guangdong",
    "湖北": "Hubei",
    "云南": "Yunnan",
}


def _clean_text(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_region_name(region):
    region = _clean_text(region)
    return _CHINA_REGION_MAP.get(region, region) or "UNKNOWN"


def canonical_region_key(station_record):
    """Return the official region key used for multi-region balancing.

    Manifest rows for Chinese provinces often have region=China_Mainland, so
    station_dir/source_region are preferred to avoid collapsing Anhui/Guangxi
    etc. into one bucket.
    """

    row = station_record or {}
    station_dir = _clean_text(row.get("station_dir")).replace("\\", "/")
    parts = [part for part in station_dir.split("/") if part]

    split_region = _clean_text(row.get("split_region"))
    if split_region:
        return _normalize_region_name(split_region)

    official_region = _clean_text(row.get("official_region"))
    if official_region:
        return _normalize_region_name(official_region)

    source_region = _clean_text(row.get("source_region"))
    if source_region and source_region not in {"China_Mainland", "China"}:
        return _normalize_region_name(source_region)

    if len(parts) >= 3 and parts[1] == "China_Mainland":
        return _normalize_region_name(parts[2])
    if len(parts) >= 2:
        return _normalize_region_name(parts[1])

    region = _clean_text(row.get("region"))
    if region and region not in {"China_Mainland", "China"}:
        return _normalize_region_name(region)
    if source_region:
        return _normalize_region_name(source_region)
    if region:
        return _normalize_region_name(region)
    return "UNKNOWN"


def sample_weights_from_region_keys(region_keys, alpha):
    alpha = max(0.0, float(alpha or 0.0))
    if not region_keys:
        return []
    if alpha <= 0.0:
        return [1.0 for _ in region_keys]
    counts = Counter(region_keys)
    raw_by_region = {
        region: float(count) ** (-alpha)
        for region, count in counts.items()
        if count > 0
    }
    weighted_total = sum(
        float(counts[region]) * raw for region, raw in raw_by_region.items()
    )
    scale = float(len(region_keys)) / max(weighted_total, 1e-12)
    return [raw_by_region.get(region, 1.0) * scale for region in region_keys]


def normalize_region_weights(weights):
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Region weights are all zero")
    return {region: weight / total for region, weight in weights.items()}


def _stable_probability_caps(raw_probs, upper_bounds):
    """Temporal-only cap normalization with canonical floating-point order.

    Use a sorted iteration order throughout, not just in the returned mapping,
    to keep floating-point results and checkpoint fingerprints deterministic.
    """
    eps = 1e-12
    regions = sorted(raw_probs)
    if not regions:
        return {}
    finite_regions = [
        r for r in regions if math.isfinite(upper_bounds.get(r, math.inf))
    ]
    if not finite_regions:
        return {r: raw_probs[r] for r in regions}
    if (
        len(finite_regions) == len(regions)
        and sum(upper_bounds[r] for r in finite_regions) < 1.0 - eps
    ):
        raise ValueError("Temporal region probability caps are infeasible")

    remaining = list(regions)
    capped = {}
    remaining_mass = 1.0
    while remaining:
        raw_total = sum(raw_probs[r] for r in remaining)
        if raw_total <= 0.0:
            tentative = {r: remaining_mass / len(remaining) for r in remaining}
        else:
            tentative = {
                r: remaining_mass * raw_probs[r] / raw_total for r in remaining
            }
        violators = [
            r for r in remaining if tentative[r] > upper_bounds.get(r, math.inf) + eps
        ]
        if not violators:
            capped.update(tentative)
            break
        for region in violators:
            upper = upper_bounds.get(region, math.inf)
            if not math.isfinite(upper):
                continue
            capped[region] = upper
            remaining_mass -= upper
            remaining.remove(region)
        if remaining_mass < -eps:
            raise ValueError(
                "Temporal region probability caps produced negative remaining mass"
            )
        remaining_mass = max(0.0, remaining_mass)
    total = sum(capped[r] for r in regions)
    if total <= 0.0:
        raise ValueError("Temporal capped region probabilities are all zero")
    return {r: capped[r] / total for r in regions}


def legal_count_probabilities(
    counts, *, alpha=0.25, max_oversampling=20.0, max_region_probability=0.15
):
    """Inverse-count balancing with explicit, step-independent density caps.

    The density cap bounds sampling probability relative to legal-window supply.
    It is independent of epoch length. Infeasible caps fail closed; source/C/H
    are never resampled and probability limits are never silently relaxed.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not math.isfinite(max_oversampling) or max_oversampling < 1:
        raise ValueError("max_oversampling must be finite and >= 1")
    if not math.isfinite(max_region_probability) or not 0 < max_region_probability <= 1:
        raise ValueError("max_region_probability must be in (0, 1]")
    if any(
        not isinstance(n, Integral) or isinstance(n, bool) or n < 0
        for n in counts.values()
    ):
        raise ValueError("legal cutoff counts must be nonnegative integers")
    available = {r: int(counts[r]) for r in sorted(counts) if counts[r] > 0}
    if not available:
        raise DynamicPoolUnavailable("no legal cutoff for fixed source/C/H")
    total = sum(available.values())
    # Rescale log weights to avoid underflow for otherwise valid inputs.
    logs = {r: -alpha * math.log(n) for r, n in available.items()}
    largest = max(logs.values())
    weights = {r: math.exp(v - largest) for r, v in logs.items()}
    weight_total = sum(weights.values())
    raw = {r: w / weight_total for r, w in weights.items()}
    bounds = {
        r: min(max_region_probability, max_oversampling * n / total)
        for r, n in available.items()
    }
    if sum(bounds.values()) < 1 - 1e-12:
        raise DynamicPoolUnavailable(
            f"infeasible region caps: legal_cutoffs={available} sum_caps={sum(bounds.values())}; "
            "source/C/H and caps are not relaxed"
        )
    probs = _stable_probability_caps(raw, bounds)
    return probs, bounds


def region_cutoff_counts(eligible):
    counts = {}
    for region, rows in eligible.by_region.items():
        count = 0
        for row in rows:
            start, stop = eligible.offsets[row]
            before = int(eligible.cumulative[start - 1]) if start else 0
            count += int(eligible.cumulative[stop - 1]) - before
        counts[region] = count
    return counts
