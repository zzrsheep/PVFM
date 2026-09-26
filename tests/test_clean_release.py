"""The public release has one graph and no private machine-path dependencies."""
import ast
import json
from pathlib import Path
import re

import torch

from pvfm import PVFMForecaster

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_graph_and_active_parameter_counts():
    expected = {"pvfm_3_9m": 3879252, "pvfm_15_3m": 15324756, "pvfm_45_0m": 44991828}
    for name, count in expected.items():
        config = json.loads((ROOT / "configs" / (name + ".json")).read_text())
        model = PVFMForecaster(**config)
        assert sum(p.numel() for p in model.parameters()) == count
        assert model.future_patch_refine is not None
        assert model.history_patch_tokenizer.pos_embedding is not None
        assert not hasattr(model, "direct_heads")
        assert not hasattr(model, "resolution_embedding")
        assert not hasattr(model, "patch_head")
        assert not hasattr(model.history_patch_tokenizer, "time_proj")


def test_no_private_machine_paths_or_historical_shell_archives():
    # Validate content, not the location where a reader extracted the repository.
    private_machine_path = re.compile(r"/home/|/mnt/|/tmp/pvfm_")
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or relative.parts[0] in {"outputs", ".git"}:
            continue
        if path.suffix in {".pt", ".npz", ".pyc"}:
            continue
        if path == Path(__file__):
            continue
        assert not private_machine_path.search(path.read_text()), str(relative)
    assert not (ROOT / "baselines/jobs").exists()
    assert not list((ROOT / "baselines/configs").rglob("source_command.txt"))
    assert not (ROOT / "baselines/engines").exists()


def test_code_is_syntactically_valid():
    for folder in [ROOT / "pvfm", ROOT / "baselines/engine", ROOT / "baselines/compat"]:
        for path in folder.rglob("*.py"):
            ast.parse(path.read_text())


def test_checkpoint_contains_only_active_tensors_and_portable_config():
    for path in (ROOT / "checkpoints").glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = PVFMForecaster(**payload["model_config"])
        assert set(model.state_dict()) == set(payload["model_state_dict"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        assert payload["model_config"] == json.loads((ROOT / "configs/pvfm_3_9m.json").read_text())
