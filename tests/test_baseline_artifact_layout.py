"""Readable artifact paths must preserve the frozen reproduction protocol."""

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "baselines"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_named_csv_inventory():
    index = rows(BASE / "artifact_index.csv")
    assert len(index) == 44
    assert len({r["file"] for r in index}) == 44
    assert len({r["station"] for r in index}) == 11
    assert not (BASE / "artifacts").exists()
    for entry in index:
        path = BASE / entry["file"]
        assert path.is_file()
        assert not re.fullmatch(r"[a-f0-9]{64}", path.stem)
        assert len(rows(path)) == int(entry["rows"])


def test_station_manifests_have_no_machine_roots():
    files = list((BASE / "manifests").glob("*/*.csv"))
    assert len(files) == 22
    assert {p.parent.name for p in files} == {"1h", "15min"}
    for path in files:
        data = rows(path)
        assert len(data) == 1 and "source_root" not in data[0]
        assert data[0]["granularity"] == path.parent.name
        assert data[0]["region"] == "Australia"
        for value in data[0].values():
            assert not value.startswith("/")
            assert not re.match(r"^[A-Za-z]:[\\/]", value)


def test_frozen_windows_match_pre_rename_bytes():
    report = json.loads(
        (ROOT / "verification/baseline_artifact_layout.json").read_text()
    )
    assert len(report["window_sha256"]) == 22
    total = 0
    for name, digest in report["window_sha256"].items():
        path = ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        data = rows(path)
        total += len(data)
        for row in data:
            assert row["resolution"] == "15min"
            assert path.parent.name == f"15min_C{row['seq_len']}_H{row['pred_len']}"
            assert row["split"] == "test"
    assert total == 52964


def test_all_job_csv_references_exist_and_are_shared():
    references = set()
    configs = list((BASE / "configs").rglob("seed*.json"))
    assert len(configs) == 2500
    for path in configs:
        text = path.read_text()
        assert "baselines/artifacts/" not in text
        config = json.loads(text)
        for option in ("--manifest_path", "--eval_origin_manifest"):
            for name in config["options"].get(option, []):
                name = name.removeprefix("{PACKAGE}/")
                target = ROOT / name
                assert target.is_file()
                references.add(name)
        if config.get("reference_eval_origins"):
            assert (ROOT / config["reference_eval_origins"]).is_file()
    expected = {f"baselines/{r['file']}" for r in rows(BASE / "artifact_index.csv")}
    assert references == expected
