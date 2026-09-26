"""Only the formal full-data average-supply policy is exposed."""

import inspect
from pathlib import Path
import random

import pytest

from pvfm.data.temporal_pool_v1 import TemporalPoolSampler
from pvfm.data.window_shapes import sample_ratio_context_horizon
from pvfm.data.regions import legal_count_probabilities, normalize_region_weights
from pvfm.data.cache_contract import DynamicPoolUnavailable
from pvfm.data.temporal_pool_v1 import build_temporal_pool, TemporalPoolDataset
from tests.loader_fixture import AuditParent

ROOT = Path(__file__).resolve().parents[1]


def test_removed_modules_and_experiment_switches():
    for name in [
        "pool_sampling",
        "region_balanced_sampler",
        "datasets",
        "batch_types",
        "supply_counts",
        "temporal_region_rule",
    ]:
        assert not (ROOT / "pvfm/data" / (name + ".py")).exists()
    signature = inspect.signature(TemporalPoolSampler)
    for keyword in [
        "pretrain_data_fraction",
        "pretrain_data_fraction_seed",
        "short_horizon_probability",
        "short_horizon_max_hours",
        "use_h6_special_contexts",
    ]:
        assert keyword not in signature.parameters
        with pytest.raises(TypeError, match=keyword):
            TemporalPoolSampler(None, 1, **{keyword: 0.5})
    with pytest.raises(TypeError, match="canonical_context_probability"):
        sample_ratio_context_horizon(random.Random(0), canonical_context_probability=1)


def test_region_rules_remain_fail_closed():
    assert normalize_region_weights({"b": 3, "a": 1}) == {"b": 0.75, "a": 0.25}
    with pytest.raises(DynamicPoolUnavailable, match="infeasible region caps"):
        legal_count_probabilities({"only_region": 100}, max_region_probability=0.15)
    probs, caps = legal_count_probabilities({f"R{i}": 100 for i in range(8)})
    assert sum(probs.values()) == 1
    assert all(probs[r] <= caps[r] for r in probs)


def test_quality_checks_survive_cleanup(tmp_path):
    base = AuditParent(tmp_path)
    dataset = TemporalPoolDataset(
        base, build_temporal_pool(base, tmp_path / "pool"), 672, 168
    )
    dataset.ensure_quality_index(tmp_path / "quality")
    row = next(
        i
        for i, r in enumerate(dataset.pool_rows)
        if r["resolution"] == "1h" and r["station_index"] == 0
    )
    assert dataset.validity_reason(row, 6, 6, 100) is None
    assert dataset.validity_reason(row, 6, 6, 1011) == "target_mask"
    assert dataset.validity_reason(row, 6, 6, 1040) is not None
    # Cadence is enforced by the reusable eligibility index; the runtime
    # validity check covers masks and bounds without rescanning timestamps.
    eligible = dataset.quality_index.query("1h", "cache", 6, 6)
    start, stop = eligible.offsets[row]
    assert not any(
        int(left) <= 1100 <= int(right)
        for left, right in zip(eligible.starts[start:stop], eligible.stops[start:stop])
    )
    assert dataset.validity_reason(row, 6, 6, 1432) == "split_boundary"
    assert dataset.validity_reason(row, 6, 6, 0) is not None
    assert len(dataset.pool_rows) == 32  # 16 physical stations, both resolutions.
