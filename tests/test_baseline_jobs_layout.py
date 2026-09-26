"""Readable paths preserve every archived configuration and metric cell."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

from baselines import run as launcher

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "baselines"


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def catalog():
    return json.loads((BASE / "catalog.json").read_text())


def report():
    return json.loads((ROOT / "verification/baseline_jobs_layout.json").read_text())


def test_all_configuration_bytes_unchanged():
    jobs = catalog()["jobs"]
    assert len(jobs) == 2500
    assert not (BASE / "jobs").exists()
    assert len({job["config"] for job in jobs}) == len(jobs)
    actual = []
    for job in jobs:
        path = launcher.config_path(job)
        parts = path.relative_to(BASE).parts
        assert len(parts) == 5 and parts[0] == "configs"
        assert parts[2] == job["task"]
        assert parts[3] == job["station_dir"].split("__", 1)[1]
        assert parts[4] == f"seed{job['seed']}.json"
        config = launcher.load_config(job)
        assert config["source_command_sha256"] == job["source_command_sha256"]
        actual.append((job["id"], hashlib.sha256(path.read_bytes()).hexdigest()))
    assert digest(sorted(actual)) == report()["config_sha256_by_id_digest"]


def test_merged_reference_values_unchanged():
    with (BASE / catalog()["reference_metrics"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    jobs = {job["id"]: job for job in catalog()["jobs"]}
    assert len(rows) == len(jobs) == 2500
    assert {r["id"] for r in rows} == jobs.keys()
    fields = report()["original_metric_columns"]
    actual = []
    for row in rows:
        job = jobs[row["id"]]
        for key in ("model", "mode", "task", "seed", "protocol", "station_dir"):
            assert row[key] == str(job[key])
        actual.append((row["id"], {key: row[key] for key in fields}))
    assert digest(sorted(actual)) == report()["reference_rows_by_id_digest"]


def test_bad_configuration_paths_and_metadata_fail_closed():
    job = catalog()["jobs"][0]
    for path in (
        "",
        "../bad.json",
        "/bad.json",
        "configs/../bad.json",
        "jobs/hash/config.json",
    ):
        with pytest.raises(ValueError, match="configuration path"):
            launcher.config_path({**job, "config": path})
    with pytest.raises(ValueError, match="Catalog/config mismatch"):
        launcher.load_config({**job, "seed": 9999})


def test_dry_run_does_not_launch(monkeypatch, capsys):
    job = catalog()["jobs"][0]
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--id", job["id"], "--seed", str(job["seed"])]
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *a, **k: pytest.fail("dry-run launched a process"),
    )
    launcher.main()
    output = capsys.readouterr().out
    assert job["config"] in output
    assert "1 jobs; no process launched." in output


@pytest.mark.parametrize("mode", ["--audit", "--execute"])
def test_new_run_directories_are_readable(monkeypatch, tmp_path, mode):
    job = catalog()["jobs"][0]
    output = tmp_path / "results"
    calls = []
    monkeypatch.setattr(launcher, "PACKAGE", tmp_path)
    monkeypatch.setattr(launcher, "ipc_directory", lambda: tmp_path)
    monkeypatch.setattr(
        launcher, "materialize_engine", lambda protocol, out: out / "engine"
    )
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *a, **k: calls.append((a, k))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--id",
            job["id"],
            "--seed",
            str(job["seed"]),
            mode,
            "--output",
            str(output),
        ],
    )
    launcher.main()
    run = launcher.run_directory(job, output)
    launch = json.loads((run / "launch.json").read_text())
    assert launch["job"] == job and (run / "train.log").is_file()
    assert not (output / job["id"]).exists()
    assert len(calls) == (1 if mode == "--audit" else 2)
    assert calls[0][1]["env"]["PVFM_RESULTS_ROOT"] == str(run / "results")
