import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from satellite.alignment import validate_historical_satellite_times
from satellite.checkpoint import load_model, DEFAULT_CHECKPOINT
from satellite.data import (
    WindowDataset,
    synthetic_batch,
    model_inputs,
    TENSOR_KEYS,
    file_hash,
)
from satellite.metrics import masked_mse_components, PointMetrics


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return load_model()[0]


def test_export_strict_and_parameter_counts(model):
    assert sum(p.numel() for p in model.backbone.parameters()) == 3879252
    assert sum(p.numel() for p in model.satellite_adapter.parameters()) == 621121
    assert all(not p.requires_grad for p in model.backbone.parameters())
    payload = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    assert payload["adaptation_step"] == 2500
    assert not any(
        ".time_proj." in k or k.startswith("backbone.patch_head.")
        for k in payload["model_state_dict"]
    )
    metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
    text = json.dumps(metadata)
    assert '"/' not in text  # No absolute paths anywhere in serialized metadata.


def test_missing_satellite_equals_backbone(model):
    model.eval()
    batch = synthetic_batch(2)
    batch["satellite_frame_mask"].zero_()
    inputs = model_inputs(batch, "cpu")
    from satellite.model import BACKBONE_INPUTS

    with torch.no_grad():
        prediction = model(**inputs)
        expected = model.backbone(**{k: inputs[k] for k in BACKBONE_INPUTS})
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


def test_initialized_adapter_is_identity():
    model, _ = load_model(reset_adapter=True)
    batch = synthetic_batch(1)
    with torch.no_grad():
        on = model(**model_inputs(batch, "cpu"))
        batch["satellite_frame_mask"].zero_()
        off = model(**model_inputs(batch, "cpu"))
    torch.testing.assert_close(on, off, atol=0, rtol=0)


def test_gradients_and_frozen_dropout(model):
    model.train()
    model.zero_grad(set_to_none=True)
    batch = synthetic_batch(1, seed=51)
    prediction = model(**model_inputs(batch, "cpu"))
    loss, count = masked_mse_components(
        prediction, batch["future_target"], batch["future_target_mask"]
    )
    (loss / count).backward()
    assert not any(m.training for m in model.backbone.modules())
    assert all(p.grad is None for p in model.backbone.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.satellite_adapter.parameters()
    )
    model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("offset", [0, 3600000000000, 7200000000000])
def test_leaking_or_misaligned_frames_rejected(offset):
    batch = synthetic_batch(1)
    batch["satellite_frame_time_ns"][0, 0] = (
        batch["forecast_origin_time_ns"][0] + offset
    )
    with pytest.raises(ValueError):
        validate_historical_satellite_times(
            batch["satellite_frame_time_ns"],
            batch["satellite_frame_mask"],
            batch["forecast_origin_time_ns"],
        )


def test_missing_frame_nat_allowed():
    batch = synthetic_batch(1)
    batch["satellite_frame_mask"][0, 0] = False
    batch["satellite_frame_time_ns"][0, 0] = torch.iinfo(torch.int64).min
    validate_historical_satellite_times(
        batch["satellite_frame_time_ns"],
        batch["satellite_frame_mask"],
        batch["forecast_origin_time_ns"],
    )


def test_masked_nan_has_finite_loss_and_gradient():
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    total, count = masked_mse_components(
        prediction, torch.tensor([float("nan"), 1.0]), torch.tensor([False, True])
    )
    (total / count).backward()
    assert torch.equal(prediction.grad, torch.tensor([0.0, 2.0]))


def make_dataset(tmp_path, split="test", cohort="unseen"):
    batch = synthetic_batch(2)
    arrays = {k: batch[k].numpy() for k in TENSOR_KEYS}
    arrays["station_id"] = np.array(batch["station_id"])
    np.savez(tmp_path / "shard.npz", **arrays)
    manifest = {
        "format": "satellite_windows_v1",
        "split": split,
        "cohort": cohort,
        "context_steps": 16,
        "horizon_steps": 4,
        "resolution_minutes": 60,
        "shards": [
            {
                "file": "shard.npz",
                "samples": 2,
                "sha256": file_hash(tmp_path / "shard.npz"),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "manifest.json", batch


def test_portable_data_preserves_tensors(tmp_path):
    path, expected = make_dataset(tmp_path)
    data = WindowDataset(path, "test")
    batch = next(iter(DataLoader(data, batch_size=2)))
    for key in TENSOR_KEYS:
        torch.testing.assert_close(batch[key], expected[key], atol=0, rtol=0)


def test_holdout_never_train_or_validate(tmp_path):
    path, _ = make_dataset(tmp_path, split="train", cohort="unseen")
    with pytest.raises(ValueError, match="test-only"):
        WindowDataset(path, "train")


def test_corrupted_data_rejected(tmp_path):
    path, _ = make_dataset(tmp_path)
    with (tmp_path / "shard.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        WindowDataset(path, "test")


def test_point_metrics():
    metrics = PointMetrics()
    metrics.update([0.0, 2.0], [0.0, 1.0], [True, True])
    result = metrics.result()
    assert result["mae"] == 0.5
    assert result["rmse"] == pytest.approx(np.sqrt(0.5))
    assert result["r2"] == -1.0
