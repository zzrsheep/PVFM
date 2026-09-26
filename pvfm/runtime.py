"""Small explicit model API; labels never enter model kwargs."""

import json
from pathlib import Path

import numpy as np
import torch

from .model import PVFMForecaster as PVFM

INPUT_FIELDS = (
    "past_target",
    "past_observed_mask",
    "historical_covariates",
    "historical_covariates_mask",
    "future_covariates",
    "future_covariates_mask",
    "past_time_features",
    "future_time_features",
    "static_features",
    "site_features",
)


def model_inputs(batch, device="cpu"):
    result = {}
    for name in INPUT_FIELDS:
        value = batch[name]
        result[name] = (
            value.to(device=device, dtype=torch.float32)
            if torch.is_tensor(value)
            else torch.as_tensor(value, device=device, dtype=torch.float32)
        )
    return result


def load_model(checkpoint, device="cpu"):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected portable PVFM checkpoint format_version=1")
    model = PVFM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["model_config"]


def load_window(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def safe_output(path):
    """All tool outputs stay under this independent directory; never overwrite."""
    root = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Output must be inside this package: {root}")
    path.mkdir(parents=True, exist_ok=False)
    return path
