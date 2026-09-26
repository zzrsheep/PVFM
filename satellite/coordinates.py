"""Coordinate utilities for station-centred satellite images."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

EARTH_RADIUS_KM = 6371.0088


def pool_pixel_coordinates(
    pixel_coordinates: Tensor,
    *,
    patch_size: int | Tuple[int, int],
) -> Tensor:
    """Average pixel coordinates with exactly the image patchification layout.

    Args:
        pixel_coordinates: ``[B,H,W,2]`` in any linear coordinate system.
        patch_size: integer or ``(height, width)`` patch size.

    Returns:
        ``[B,N,2]`` coordinates in H-major/W-minor token order.
    """

    if pixel_coordinates.ndim != 4 or pixel_coordinates.shape[-1] != 2:
        raise ValueError(
            "pixel_coordinates must have shape [B,H,W,2], "
            f"got {tuple(pixel_coordinates.shape)}."
        )
    if isinstance(patch_size, int):
        patch_height = patch_width = int(patch_size)
    else:
        patch_height, patch_width = map(int, patch_size)
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch_size entries must be positive.")
    height, width = pixel_coordinates.shape[1:3]
    if height % patch_height or width % patch_width:
        raise ValueError(
            f"Image {(height, width)} is not divisible by patch size "
            f"{(patch_height, patch_width)}."
        )
    if not torch.isfinite(pixel_coordinates).all():
        raise ValueError("pixel_coordinates must contain only finite values.")

    channels_first = pixel_coordinates.permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(
        channels_first,
        kernel_size=(patch_height, patch_width),
        stride=(patch_height, patch_width),
    )
    return pooled.flatten(2).transpose(1, 2).contiguous()


def geographic_grid_to_local_aeqd_km(
    geographic_grid: Tensor,
    station_lat_lon: Tensor,
) -> Tensor:
    """Convert latitude/longitude grids to spherical AEQD east/north km.

    ``geographic_grid[...,0]`` is latitude and ``[...,1]`` is longitude.
    The returned last dimension is ``[east_km, north_km]`` relative to the
    corresponding station.  This helper is intended for sidecar generation;
    the static result should normally be cached instead of recomputed per
    training step.
    """

    if geographic_grid.ndim != 4 or geographic_grid.shape[-1] != 2:
        raise ValueError(
            "geographic_grid must have shape [B,H,W,2], "
            f"got {tuple(geographic_grid.shape)}."
        )
    if station_lat_lon.ndim != 2 or station_lat_lon.shape != (
        geographic_grid.shape[0],
        2,
    ):
        raise ValueError(
            "station_lat_lon must have shape [B,2] matching the grid batch, "
            f"got {tuple(station_lat_lon.shape)}."
        )
    if (
        not torch.isfinite(geographic_grid).all()
        or not torch.isfinite(station_lat_lon).all()
    ):
        raise ValueError("Geographic coordinates must be finite.")

    dtype = torch.float64
    grid = geographic_grid.to(dtype=dtype)
    station = station_lat_lon.to(device=grid.device, dtype=dtype)
    latitude = torch.deg2rad(grid[..., 0])
    longitude = torch.deg2rad(grid[..., 1])
    latitude_0 = torch.deg2rad(station[:, 0])[:, None, None]
    longitude_0 = torch.deg2rad(station[:, 1])[:, None, None]

    delta_lon = longitude - longitude_0
    delta_lon = torch.atan2(torch.sin(delta_lon), torch.cos(delta_lon))
    sin_latitude, cos_latitude = torch.sin(latitude), torch.cos(latitude)
    sin_latitude_0, cos_latitude_0 = (
        torch.sin(latitude_0),
        torch.cos(latitude_0),
    )
    cos_c = (
        sin_latitude_0 * sin_latitude
        + cos_latitude_0 * cos_latitude * torch.cos(delta_lon)
    ).clamp(-1.0, 1.0)
    c = torch.acos(cos_c)
    sin_c = torch.sin(c)
    scale = torch.where(
        c.abs() < 1e-10,
        torch.ones_like(c),
        c / sin_c.clamp_min(1e-12),
    )
    east = EARTH_RADIUS_KM * scale * cos_latitude * torch.sin(delta_lon)
    north = (
        EARTH_RADIUS_KM
        * scale
        * (
            cos_latitude_0 * sin_latitude
            - sin_latitude_0 * cos_latitude * torch.cos(delta_lon)
        )
    )
    return torch.stack((east, north), dim=-1).to(
        device=geographic_grid.device,
        dtype=geographic_grid.dtype,
    )


def geographic_grid_to_patch_coordinates(
    geographic_grid: Tensor,
    station_lat_lon: Tensor,
    *,
    patch_size: int | Tuple[int, int],
    coordinate_scale_km: float,
) -> Tensor:
    """Convert a corrected geographic grid directly to normalized patch coords."""

    if coordinate_scale_km <= 0:
        raise ValueError("coordinate_scale_km must be positive.")
    local_grid = geographic_grid_to_local_aeqd_km(geographic_grid, station_lat_lon)
    return pool_pixel_coordinates(local_grid, patch_size=patch_size) / float(
        coordinate_scale_km
    )
