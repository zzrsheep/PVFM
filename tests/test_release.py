import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pvfm.runtime import load_model, model_inputs, PVFM
from pvfm.metrics import common_q9, station_metrics, station_equal
from pvfm.training_data import sampler_kwargs, policy
from pvfm.data.temporal_pool_v1 import (
    build_temporal_pool,
    TemporalPoolDataset,
    TemporalPoolSampler,
)
from pvfm.data.temporal_average_supply import build_or_load_average_supply
from pvfm.data.sampler import AverageSupplyTemporalSampler
from tests.fixtures import _Parent

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_standalone_imports():
    for p in (ROOT / "pvfm").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("foundation", "scripts.", "models.")
                )


@pytest.mark.parametrize(
    "task",
    ["1h_C24_H6", "1h_C72_H24", "1h_C336_H168", "15min_C64_H16", "15min_C96_H24"],
)
def test_synthetic_checkpoint_forward_and_no_label_input(task):
    # Functional check only: no numerical parity claim against real observations.
    resolution, context, horizon = task.split("_")
    c, h = int(context[1:]), int(horizon[1:])
    rng = torch.Generator().manual_seed(20260926)
    batch = {
        "past_target": torch.rand(2, c, 1, generator=rng),
        "past_observed_mask": torch.ones(2, c, 1),
        "historical_covariates": torch.randn(2, c, 6, generator=rng),
        "historical_covariates_mask": torch.ones(2, c, 6),
        "future_covariates": torch.randn(2, h, 6, generator=rng),
        "future_covariates_mask": torch.ones(2, h, 6),
        "past_time_features": torch.zeros(2, c, 5),
        "future_time_features": torch.zeros(2, h, 5),
        "static_features": torch.tensor(
            [c, h, h, h * (1 if resolution == "1h" else 0.25)]
        ).repeat(2, 1),
        "site_features": torch.tensor([-30.0, 150.0, 0.0, 1.0, 10.0]).repeat(2, 1),
        "future_target": torch.rand(2, h, 1, generator=rng),
    }
    model, cfg = load_model(ROOT / "checkpoints/pvfm_3_9m_80k_q11.pt")
    with torch.inference_mode():
        q = model(**model_inputs(batch))[1]["quantiles"].numpy()
    assert q.shape == (2, h, 11)
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=-1) >= 0).all()
    # Future PV labels are not a model input.
    batch["future_target"] = np.full_like(batch["future_target"], 1e9)
    with torch.inference_mode():
        changed = model(**model_inputs(batch))[1]["quantiles"].numpy()
    np.testing.assert_array_equal(q, changed)


def test_model_configs():
    for name, d, h, l, ffn in [
        ("pvfm_3_9m", 128, 8, 6, 512),
        ("pvfm_15_3m", 256, 4, 6, 1024),
        ("pvfm_45_0m", 384, 6, 8, 1536),
    ]:
        cfg = json.loads((ROOT / f"configs/{name}.json").read_text())
        # PVFM's FFN is implemented as 4*hidden_dim, not a constructor argument.
        assert (
            cfg["hidden_dim"],
            cfg["n_heads"],
            cfg["n_layers"],
            4 * cfg["hidden_dim"],
        ) == (d, h, l, ffn)
        assert cfg["future_joint_decoder_layers"] == l
        # The public model has one fixed Solar/Q11 graph, not ablation flags.
        assert not any(key.startswith("use_") for key in cfg)


def test_probabilistic_grid():
    y = np.array([0.0, 1.0])
    mask = np.ones(2)
    levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    q = np.tile(np.arange(11) / 10, (2, 1))
    nine = common_q9(q, levels)
    a = station_metrics(q, y, mask, levels)
    b = station_metrics(nine, y, mask, np.arange(1, 10) / 10)
    assert a == b
    q[:, 0] = -10000
    q[:, -1] = 10000
    assert station_metrics(q, y, mask, levels) == a
    with pytest.raises(ValueError):
        common_q9(q, [0.01] * 11)


def test_mask_and_constant():
    q = np.zeros((3, 9))
    y = np.array([0.0, 0.0, np.nan])
    mask = np.array([1, 1, 0])
    m = station_metrics(q, y, mask, np.arange(1, 10) / 10)
    assert np.isnan(m["R2"]) and m["MAE"] == m["CRPS"] == m["AQL"] == 0
    assert m["valid_targets"] == 2
    m2 = {**m, "MAE": 2.0, "valid_targets": 100}
    assert station_equal([m, m2])["MAE"] == 1


def dataset(tmp_path):
    base = _Parent(tmp_path, n=4096)
    pool = build_temporal_pool(base, tmp_path / "pool")
    ds = TemporalPoolDataset(
        base, pool, 672, 168, dynamic_native_task_names=("pv_15min_24h_ahead",)
    )
    ds.ensure_quality_index(tmp_path / "quality")
    return ds


def test_pool_sampling_profile_and_resume(tmp_path):
    ds = dataset(tmp_path)
    kwargs = sampler_kwargs(policy(), 2, 0, 1)
    # Single-region synthetic fixture: only this fixture needs a cap of 1.
    kwargs.update(region_balance_max_prob=1.0, batches_per_epoch=10)
    shape = TemporalPoolSampler(ds, **kwargs)
    profile, reused = build_or_load_average_supply(
        ds,
        shape,
        tmp_path / "profile",
        samples_per_source=20,
        max_region_probability=1.0,
        calibration_seed=20260913,
    )
    assert not reused

    def make():
        return AverageSupplyTemporalSampler(
            ds,
            **kwargs,
            region_profile_path=tmp_path / "profile/profile.json",
            average_max_oversampling=20.0,
        )

    sampler = make()
    iterator = iter(sampler)
    descriptors = next(iterator)
    batch = ds.collate_fn([ds[d] for d in descriptors])
    sampler.mark_consumed(batch)
    state = sampler.state_dict()
    expected = next(iterator)
    restored = make()
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected
    for descriptor in expected:
        assert ds.validity_reason(*descriptor) is None
    changed = copy.deepcopy(state)
    changed["pool_fingerprint"] = "wrong"
    with pytest.raises(ValueError):
        make().load_state_dict(changed)


def test_shared_shapes_without_ddp(tmp_path):
    ds = dataset(tmp_path)
    common = sampler_kwargs(policy(), 2, 0, 4)
    common.update(region_balance_max_prob=1.0, batches_per_epoch=20)
    build_or_load_average_supply(
        ds,
        TemporalPoolSampler(ds, **common),
        tmp_path / "profile",
        samples_per_source=20,
        max_region_probability=1.0,
        calibration_seed=20260913,
    )
    traces = []
    for rank in range(4):
        kw = {**common, "rank": rank}
        sampler = AverageSupplyTemporalSampler(
            ds, **kw, region_profile_path=tmp_path / "profile/profile.json"
        )
        traces.append(
            [(descriptors[0][1], descriptors[0][2]) for descriptors in sampler]
        )
    assert all(trace == traces[0] for trace in traces)


def test_australian_cohort_is_the_only_supplied_forecast_data():
    stations = json.loads((ROOT / "baselines/australia_stations.json").read_text())
    assert len(stations) == len(set(stations)) == 11
    assert all("/Australia/" in station for station in stations)
    assert not (ROOT / "example_data").exists()
