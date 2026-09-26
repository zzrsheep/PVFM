"""Read-only bridge from a frozen PVFM backbone to satellite anchors.

This module does not modify or replace any PVFM parameter.  It extracts the
exact history patch position, solar-patch projection, and site bias already
used by the loaded PVFM checkpoint so the satellite memory is anchored in the
same representation system.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class FrozenHistoryAnchors:
    solar_geometry: Tensor
    patch_position_embedding: Tensor
    solar_patch_embedding: Tensor
    site_bias: Tensor


@torch.no_grad()
def extract_history_anchors(
    backbone: nn.Module,
    satellite_encoder: nn.Module,
    *,
    past_time_features: Tensor,
    site_features: Tensor,
    static_features: Tensor | None,
) -> FrozenHistoryAnchors:
    """Extract exact PVFM patch/solar/site anchors without copying weights."""

    required = (
        "hidden_dim",
        "history_patch_tokenizer",
        "_build_step_solar_geometry",
        "_inject_site_token_context",
    )
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise TypeError(
            "backbone does not expose the required PVFM interface: "
            + ", ".join(missing)
        )
    if past_time_features.ndim != 3:
        raise ValueError(
            "past_time_features must have shape [B,T,F], "
            f"got {tuple(past_time_features.shape)}."
        )
    if site_features.ndim != 2 or site_features.shape[0] != past_time_features.shape[0]:
        raise ValueError(
            "site_features must have shape [B,F] with the same batch as time features."
        )

    batch, time_steps = past_time_features.shape[:2]
    hidden_dim = int(backbone.hidden_dim)
    tokenizer = backbone.history_patch_tokenizer
    encoder_required = (
        "temporal_patch_len",
        "temporal_patch_stride",
        "output_dim",
    )
    encoder_missing = [
        name for name in encoder_required if not hasattr(satellite_encoder, name)
    ]
    if encoder_missing:
        raise TypeError(
            "satellite_encoder does not expose its alignment contract: "
            + ", ".join(encoder_missing)
        )
    if int(satellite_encoder.temporal_patch_len) != int(tokenizer.patch_len):
        raise ValueError(
            "Satellite/PVFM temporal patch lengths differ: "
            f"{satellite_encoder.temporal_patch_len} != {tokenizer.patch_len}."
        )
    if int(satellite_encoder.temporal_patch_stride) != int(tokenizer.patch_stride):
        raise ValueError(
            "Satellite/PVFM temporal patch strides differ: "
            f"{satellite_encoder.temporal_patch_stride} != {tokenizer.patch_stride}."
        )
    if int(satellite_encoder.output_dim) != hidden_dim:
        raise ValueError(
            "Satellite output dimension must match PVFM hidden dimension: "
            f"{satellite_encoder.output_dim} != {hidden_dim}."
        )
    patch_count = int(tokenizer._num_segments(time_steps))
    if tokenizer.pos_embedding.shape[2] < patch_count:
        raise ValueError(
            "PVFM position table is shorter than the requested patch count: "
            f"{tokenizer.pos_embedding.shape[2]} < {patch_count}."
        )

    solar_geometry = backbone._build_step_solar_geometry(
        past_time_features,
        site_features,
        time_steps,
    )
    if solar_geometry is None:
        solar_geometry = past_time_features.new_zeros(batch, time_steps, 4)
    solar_geometry = solar_geometry.to(device=past_time_features.device)

    anchor_device = tokenizer.pos_embedding.device
    anchor_dtype = tokenizer.pos_embedding.dtype

    patch_position = tokenizer.pos_embedding[:, 0, :patch_count, :]
    patch_position = patch_position.expand(batch, -1, -1).detach()

    if tokenizer.solar_patch_time_proj is None:
        solar_patch = torch.zeros(
            batch,
            patch_count,
            hidden_dim,
            device=anchor_device,
            dtype=anchor_dtype,
        )
    else:
        solar = solar_geometry.to(device=anchor_device, dtype=anchor_dtype)
        if solar.shape[-1] < 4:
            solar = F.pad(solar, (0, 4 - solar.shape[-1]))
        solar = solar[..., :4]
        solar_patches = tokenizer._patchify_channel_first(solar.permute(0, 2, 1))
        if solar_patches.shape[2] != patch_count:
            raise ValueError(
                "PVFM solar patch count does not match satellite windows: "
                f"{solar_patches.shape[2]} != {patch_count}."
            )
        # Exact tokenizer order: [B, P, patch_len, 4] -> time-major flatten.
        solar_flat = solar_patches.permute(0, 2, 3, 1).reshape(
            batch, patch_count, tokenizer.patch_len * 4
        )
        solar_patch = (1.0 * tokenizer.solar_patch_time_proj(solar_flat)).detach()

    zero_tokens = torch.zeros(
        batch,
        1,
        patch_count,
        hidden_dim,
        device=anchor_device,
        dtype=anchor_dtype,
    )
    site_tokens = backbone._inject_site_token_context(
        zero_tokens,
        None if static_features is None else static_features.to(anchor_device),
        site_features.to(anchor_device),
    )
    site_bias = site_tokens[:, 0, 0, :].detach()
    if not torch.allclose(
        site_tokens,
        site_bias[:, None, None, :].expand_as(site_tokens),
    ):
        raise RuntimeError("PVFM site adapter did not produce one shared station bias.")

    return FrozenHistoryAnchors(
        solar_geometry=solar_geometry.detach(),
        patch_position_embedding=patch_position,
        solar_patch_embedding=solar_patch,
        site_bias=site_bias,
    )
