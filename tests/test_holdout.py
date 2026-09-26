import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from australia import records, load, model_batch
from evaluate_australia import evaluation_profile
from pvfm.runtime import load_model, model_inputs

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["1h_C24_H6", "1h_C72_H24", "1h_C336_H168", "15min_C64_H16", "15min_C96_H24"]


def test_holdout_checkpoint_identity_and_scope():
    hold = evaluation_profile("australia_holdout")
    allreg = evaluation_profile("all_region")
    assert hold["checkpoint"] != allreg["checkpoint"]
    assert hold["checkpoint_sha256"] != allreg["checkpoint_sha256"]
    path = ROOT / hold["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hold["checkpoint_sha256"]
    model, cfg = load_model(path)
    assert sum(p.numel() for p in model.parameters()) == 3879252
    assert cfg == json.loads((ROOT / "configs/pvfm_3_9m.json").read_text())
    audit = json.loads(
        (ROOT / "provenance/checkpoint_australia_holdout.json").read_text()
    )
    assert all(r["australia_training_stations"] == 0 for r in audit["split_audit"])
    assert audit["training_step"] == 80000


@pytest.mark.parametrize("task", TASKS)
def test_holdout_frozen_manifest_and_model_inputs(task, prepared_evaluation_assets):
    rows = records(
        "pvfm", task, manifest=evaluation_profile("australia_holdout")["manifest"]
    )
    assert len(rows) == 11
    assert {r["station_dir"] for r in rows} == set(
        json.loads((ROOT / "baselines/australia_stations.json").read_text())
    )
    for row in rows:
        path = ROOT / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        a = load(row)
        cuts = a["cutoffs"]
        assert len(cuts) == row["windows"]
        assert cuts.min() >= row["C"] and cuts.max() + row["H"] <= len(a["target"])
        inputs = model_inputs(model_batch(row, a, [0, len(cuts) - 1]))
        assert "future_target" not in inputs
        assert inputs["past_target"].shape == (2, row["C"], 1)
        assert inputs["future_covariates"].shape == (2, row["H"], 6)
        assert np.isfinite(inputs["past_target"].numpy()).all()
