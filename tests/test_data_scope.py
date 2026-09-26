"""Australia-only distribution, explicit missing assets and training isolation."""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

import australia
import evaluate_australia
from pvfm.training_data import read_data_config

ROOT = Path(__file__).resolve().parents[1]


def test_prepared_array_error_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(australia, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="not distributed"):
        australia.load({"file": "missing.npz"})


def test_missing_arrays_fail_before_weights_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(australia, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate_australia, "records", lambda *args, **kwargs: [{"file": "missing.npz"}]
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Missing assets must be rejected before model loading or output creation"
        )

    monkeypatch.setattr(evaluate_australia, "load_model", forbidden)
    monkeypatch.setattr(evaluate_australia, "safe_output", forbidden)
    output = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        evaluate_australia.main()
    assert error.value.code == 2
    assert not output.exists()


def test_assets_manifest_only_lists_distributed_files():
    assets = json.loads((ROOT / "ASSETS.json").read_text())["assets"]
    for row in assets:
        assert not row["file"].startswith("example_data/")
        assert (ROOT / row["file"]).is_file(), row["file"]


def test_pretraining_excludes_australian_targets_without_demo_files(tmp_path):
    station = json.loads((ROOT / "baselines/australia_stations.json").read_text())[0]
    train_manifest = tmp_path / "stations.csv"
    pd.DataFrame({"station_dir": [station]}).to_csv(train_manifest, index=False)
    config = {"role": "pretrain_train", "manifest_path": str(train_manifest)}
    for key in ["binary_cache_registry", "pool_dir", "quality_dir", "region_profile"]:
        config[key] = str(tmp_path / key)
    config_path = tmp_path / "data.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Australian unseen cohort"):
        read_data_config(config_path)
    pd.DataFrame({"station_dir": ["unrelated_training_station"]}).to_csv(
        train_manifest, index=False
    )
    assert read_data_config(config_path) == config


def test_retained_manifests_only_cover_australia():
    expected = set(json.loads((ROOT / "baselines/australia_stations.json").read_text()))
    for file in [
        "datasets/australia_evaluation/manifest.json",
        "datasets/australia_holdout_evaluation/manifest.json",
    ]:
        rows = json.loads((ROOT / file).read_text())["records"]
        assert {row["station_dir"] for row in rows} == expected
        assert len({row["task"] for row in rows}) == 5
