# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

import argparse

import torch

def _load_model(args: argparse.Namespace):
    try:
        from citras_fm import CitrasFM
    except ImportError as exc:
        raise RuntimeError(
            "CITRAS-FM is not installed. Install the upstream package with "
            "pip install git+https://github.com/hitachi-ais/citras-fm.git@a7f5648cbe54253c6008608f4c2026f41f178e67"
        ) from exc

    kwargs = {
        "revision": args.revision,
        "local_files_only": bool(args.local_files_only),
        "map_location": args.device,
    }
    if args.hf_cache_dir is not None:
        kwargs["cache_dir"] = str(args.hf_cache_dir)
    model = CitrasFM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    return model


def _forecast_batch(model, target: torch.Tensor, known_cov: torch.Tensor | None, horizon: int):
    """Call public CITRAS per window and stack outputs at the PVFM boundary."""
    forecast = getattr(model, "forecast", None)
    if forecast is None:
        forecast_batch = getattr(model, "forecast_batch", None)
        if forecast_batch is None:
            raise AttributeError("CITRAS model exposes neither forecast nor forecast_batch")
        return forecast_batch(target, horizon=horizon, observed_cov=None, known_cov=known_cov)

    medians = []
    quantiles = []
    for index in range(target.shape[0]):
        target_i = target[index]
        known_i = None if known_cov is None else known_cov[index]
        median_i, quantiles_i = forecast(
            target_i, horizon=horizon, observed_cov=None, known_cov=known_i
        )
        median_i = torch.as_tensor(median_i).detach().float()
        quantiles_i = torch.as_tensor(quantiles_i).detach().float()
        if median_i.ndim == 1:
            median_i = median_i.unsqueeze(-1)
        if quantiles_i.ndim == 2:
            quantiles_i = quantiles_i.unsqueeze(1)
        if median_i.shape != (horizon, 1):
            raise ValueError(f"CITRAS median shape for one window is {tuple(median_i.shape)}")
        if quantiles_i.shape[:2] != (horizon, 1):
            raise ValueError(f"CITRAS quantile shape for one window is {tuple(quantiles_i.shape)}")
        medians.append(median_i)
        quantiles.append(quantiles_i)
    return torch.stack(medians, dim=0), torch.stack(quantiles, dim=0)

