"""Path-free, tensor-only checkpoint exports; strict loading by default."""

from pathlib import Path
import torch
from .model import SatelliteForecaster

DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints/satellite_forecaster.pt"


def load_model(path=DEFAULT_CHECKPOINT, device="cpu", reset_adapter=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "pvfm_satellite_v1":
        raise ValueError("Expected pvfm_satellite_v1 checkpoint")
    model = SatelliteForecaster(payload["backbone_config"], payload["satellite_config"])
    if reset_adapter:
        state = {
            k.removeprefix("backbone."): v
            for k, v in payload["model_state_dict"].items()
            if k.startswith("backbone.")
        }
        model.backbone.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
