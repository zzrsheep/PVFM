"""Strict history-only hourly satellite alignment."""

import torch
from torch import Tensor


def validate_historical_satellite_times(
    satellite_frame_time_ns: Tensor,
    satellite_frame_mask: Tensor,
    forecast_origin_time_ns: Tensor,
) -> None:
    """Check exact hourly history slots; missing frames may carry NaT.

    The origin is the first *predicted* hour, so a frame at the origin is
    already future information. All timestamps use the same local clock,
    represented as int64 nanoseconds by the sidecar.
    """

    if satellite_frame_time_ns.dtype != torch.int64:
        raise TypeError("satellite_frame_time_ns must use torch.int64.")
    if forecast_origin_time_ns.dtype != torch.int64:
        raise TypeError("forecast_origin_time_ns must use torch.int64.")
    if satellite_frame_time_ns.ndim != 2:
        raise ValueError("satellite_frame_time_ns must have shape [B,T].")
    batch, frames = satellite_frame_time_ns.shape
    if satellite_frame_mask.shape != satellite_frame_time_ns.shape:
        raise ValueError("Satellite frame mask and timestamp shapes must match.")
    mask = satellite_frame_mask.to(
        device=satellite_frame_time_ns.device, dtype=torch.bool
    )
    origin = forecast_origin_time_ns.to(device=satellite_frame_time_ns.device)
    if origin.ndim == 2 and origin.shape[1] == 1:
        origin = origin[:, 0]
    if tuple(origin.shape) != (batch,):
        raise ValueError("forecast_origin_time_ns must have shape [B] or [B,1].")
    nat = torch.iinfo(torch.int64).min
    if bool((origin == nat).any()) or bool(
        (satellite_frame_time_ns[mask] == nat).any()
    ):
        raise ValueError("Origin and valid satellite frames cannot contain NaT.")
    if bool((mask & (satellite_frame_time_ns >= origin[:, None])).any()):
        raise ValueError(
            "Future satellite frame leakage: timestamp reaches forecast origin."
        )
    offsets = torch.arange(-frames, 0, device=origin.device, dtype=torch.int64)
    expected = origin[:, None] + offsets[None, :] * 3_600_000_000_000
    if bool((mask & (satellite_frame_time_ns != expected)).any()):
        raise ValueError(
            "Satellite timestamps must exactly match the historical hourly slots."
        )
