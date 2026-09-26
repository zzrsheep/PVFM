"""Distinguish optional real-data replay from always-runnable unit checks."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def prepared_evaluation_assets():
    manifest = json.loads(
        (ROOT / "datasets/australia_evaluation/manifest.json").read_text()
    )
    paths = [ROOT / row["file"] for row in manifest["records"]]
    present = [p.is_file() for p in paths]
    assert len(paths) == 110
    if not any(present):
        pytest.skip(
            "Real-data replay not run: the 110 optional prepared Australia arrays are not distributed"
        )
    assert all(
        present
    ), "Partially installed Australia evaluation arrays; restore the complete matching set"
