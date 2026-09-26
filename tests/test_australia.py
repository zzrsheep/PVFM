import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from australia import records, load, item, model_batch
from pvfm.runtime import model_inputs

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["1h_C24_H6", "1h_C72_H24", "1h_C336_H168", "15min_C64_H16", "15min_C96_H24"]


def test_baseline_catalog_complete_models_portable_paths():
    data = json.loads((ROOT / "baselines/catalog.json").read_text())
    jobs = data["jobs"]
    assert len(jobs) == 2500
    assert len({j["station_dir"] for j in jobs}) == 11
    assert len({(j["model"], j["mode"]) for j in jobs}) == 16
    assert len({j["model"] for j in jobs}) == 11
    assert len({j["id"] for j in jobs}) == len(jobs)
    for job in jobs:
        d = json.loads((ROOT / "baselines" / job["config"]).read_text())
        assert (
            d["protocol"]
            in json.loads((ROOT / "baselines/protocols.json").read_text())["protocols"]
        )
        assert (ROOT / "baselines/engine/run.py").is_file()
        for values in d["options"].values():
            for v in values:
                assert not v.startswith("/"), v
                if v.startswith("{PACKAGE}/"):
                    p = (ROOT / v.removeprefix("{PACKAGE}/")).resolve()
                    assert p.is_relative_to(ROOT) and p.exists(), p
        if d.get("reference_eval_origins"):
            assert (ROOT / d["reference_eval_origins"]).is_file()


@pytest.mark.parametrize("task", TASKS)
def test_australia_windows_targets_and_no_model_label_input(
    task, prepared_evaluation_assets
):
    rows = records("pvfm", task)
    assert len(rows) == 11
    for row in rows:
        arrays = load(row)
        cuts = arrays["cutoffs"]
        assert len(cuts) == row["windows"] and len(cuts) > 0
        assert (np.diff(cuts) > 0).all()
        assert cuts.min() >= row["C"] and cuts.max() + row["H"] <= len(arrays["target"])
        selected = [0, len(cuts) // 2, len(cuts) - 1]
        batch = model_batch(row, arrays, selected)
        assert batch["past_target"].shape == (3, row["C"], 1)
        assert batch["future_covariates"].shape == (3, row["H"], 6)
        inputs = model_inputs(batch)
        assert "future_target" not in inputs and "future_observed_mask" not in inputs
        for k in ["past_target", "historical_covariates", "future_covariates"]:
            assert np.isfinite(batch[k]).all()
        replay = next(
            r
            for r in records("fullshot", task)
            if r["station_dir"] == row["station_dir"]
        )
        other = load(replay)
        assert len(other["cutoffs"]) == len(cuts)
        # Compare timestamps rather than array indices: the hourly arrays may
        # have different zero origins, but evaluation targets must align.
        np.testing.assert_array_equal(
            arrays["timestamps_ns"][cuts], other["timestamps_ns"][other["cutoffs"]]
        )
        target = arrays["target"][cuts[:, None] + np.arange(row["H"])]
        expected = other["target"][other["cutoffs"][:, None] + np.arange(row["H"])]
        np.testing.assert_allclose(target, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("task", TASKS)
def test_external_direct_adapters_ignore_future_pv(task, prepared_evaluation_assets):
    sys.path.insert(0, str(ROOT / "baselines/external/runtime"))
    from foundation.tsfm_zero_shot.chronos2_covariate import build_chronos2_input
    from foundation.tsfm_zero_shot.citras_fm_covariate import build_citras_input
    from foundation.tsfm_zero_shot.timesfm3_covariate import build_timesfm3_input

    row = records("fullshot", task, "Bannerton")[0]
    arrays = load(row)
    sample = item(row, arrays, 0)
    changed = copy.deepcopy(sample)
    changed["future_target"] = np.full_like(sample["future_target"], 99999)

    def equal(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b)

    for build in [build_chronos2_input, build_citras_input, build_timesfm3_input]:
        equal(build(sample), build(changed))


def test_external_six_model_manifest_and_syntax():
    manifest = json.loads((ROOT / "baselines/external/weights.json").read_text())
    assert set(manifest) == {
        "chronos2",
        "citras",
        "moirai2",
        "timesfm3",
        "tirex2",
        "tabpfn3",
    }
    for row in manifest.values():
        assert row["bundled_weights"] is False
    for p in (ROOT / "baselines/external").rglob("*.py"):
        ast.parse(p.read_text())
