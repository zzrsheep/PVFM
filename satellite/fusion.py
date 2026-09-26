"""Patch-aligned fusion between frozen PV states and satellite memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from satellite.attention import (
    SpatialRotaryMultiheadAttention,
)
from satellite.encoder import (
    SatelliteEncoderOutput,
    SatellitePatchEncoder,
)


@dataclass(frozen=True)
class SatelliteFusionOutput:
    fused_pv_states: Tensor
    raw_delta: Tensor
    delta: Tensor
    gate: Tensor
    spatial_attention: Optional[Tensor]


@dataclass(frozen=True)
class SatelliteAdapterOutput:
    encoded_satellite: SatelliteEncoderOutput
    fusion: SatelliteFusionOutput


class PatchAlignedSatelliteFusion(nn.Module):
    """Use each encoded historical PV state to query its satellite window."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
        gate_hidden_dim: int = 128,
        spatial_rope_max_frequency: float = 10.0,
        initial_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = SpatialRotaryMultiheadAttention(
            hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_frequency=spatial_rope_max_frequency,
        )
        self.zero_output = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.zeros_(self.zero_output.weight)
        nn.init.zeros_(self.zero_output.bias)
        nn.init.constant_(self.gate[-1].bias, float(initial_gate_bias))

    def forward(
        self,
        pv_states: Tensor,
        satellite: SatelliteEncoderOutput,
        *,
        pv_patch_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> SatelliteFusionOutput:
        if pv_states.ndim != 3 or pv_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                "pv_states must have shape "
                f"[B,P,{self.hidden_dim}], got {tuple(pv_states.shape)}."
            )
        memory = satellite.memory
        if memory.ndim != 4 or memory.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Satellite memory must have shape "
                f"[B,P,N,{self.hidden_dim}], got {tuple(memory.shape)}."
            )
        batch, patches, spatial_tokens, hidden = memory.shape
        if pv_states.shape[:2] != (batch, patches):
            raise ValueError(
                "PV and satellite batch/patch axes must match: "
                f"pv={tuple(pv_states.shape)}, satellite={tuple(memory.shape)}."
            )
        coordinates = satellite.patch_coordinates
        if coordinates.shape != (batch, spatial_tokens, 2):
            raise ValueError(
                "Satellite patch coordinates must have shape "
                f"{(batch, spatial_tokens, 2)}, got {tuple(coordinates.shape)}."
            )
        window_valid = satellite.window_has_satellite.to(
            device=memory.device, dtype=torch.bool
        )
        if window_valid.shape != (batch, patches):
            raise ValueError(
                f"window_has_satellite must have shape {(batch, patches)}."
            )
        if pv_patch_mask is None:
            pv_valid = torch.ones_like(window_valid)
        else:
            if tuple(pv_patch_mask.shape) != (batch, patches):
                raise ValueError(
                    "pv_patch_mask must have shape "
                    f"{(batch, patches)}, got {tuple(pv_patch_mask.shape)}."
                )
            pv_valid = pv_patch_mask.to(device=memory.device, dtype=torch.bool)
        # A satellite observation must never resurrect a history PV patch that
        # the frozen backbone marked invalid.
        fusion_valid = window_valid & pv_valid

        query = self.query_norm(pv_states).reshape(batch * patches, 1, hidden)
        key_value = self.memory_norm(memory).reshape(
            batch * patches, spatial_tokens, hidden
        )
        query_coordinates = torch.zeros(
            batch * patches,
            1,
            2,
            device=memory.device,
            dtype=torch.float32,
        )
        key_coordinates = (
            coordinates[:, None, :, :]
            .expand(batch, patches, spatial_tokens, 2)
            .reshape(batch * patches, spatial_tokens, 2)
        )
        flat_fusion_valid = fusion_valid.reshape(batch * patches)
        query_mask = flat_fusion_valid[:, None]
        key_mask = flat_fusion_valid[:, None].expand(batch * patches, spatial_tokens)

        attention_result = self.cross_attention(
            query,
            query_coordinates=query_coordinates,
            key_value=key_value,
            key_coordinates=key_coordinates,
            query_mask=query_mask,
            key_mask=key_mask,
            return_attention=return_attention,
        )
        if return_attention:
            raw_delta_flat, attention = attention_result
            attention = attention.reshape(
                batch, patches, attention.shape[1], 1, spatial_tokens
            )
        else:
            raw_delta_flat = attention_result
            attention = None
        raw_delta = raw_delta_flat.reshape(batch, patches, hidden)
        delta = self.zero_output(raw_delta)

        satellite_summary = memory.mean(dim=2)
        gate_features = torch.cat(
            (
                pv_states,
                satellite_summary,
                satellite.valid_fraction.to(memory.dtype).unsqueeze(-1),
                satellite.daylight_fraction.to(memory.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_features))
        gate = gate * fusion_valid.unsqueeze(-1).to(gate.dtype)
        fused = pv_states + gate * delta
        return SatelliteFusionOutput(
            fused_pv_states=fused,
            raw_delta=raw_delta,
            delta=delta,
            gate=gate,
            spatial_attention=attention,
        )


class SatelliteAdapter(nn.Module):
    """Compose the satellite encoder and the post-PV fusion operation."""

    def __init__(
        self,
        encoder: SatellitePatchEncoder,
        fusion: PatchAlignedSatelliteFusion,
    ) -> None:
        super().__init__()
        if encoder.output_dim != fusion.hidden_dim:
            raise ValueError(
                "Satellite encoder output_dim must equal fusion hidden_dim, "
                f"got {encoder.output_dim} and {fusion.hidden_dim}."
            )
        self.encoder = encoder
        self.fusion = fusion

    def forward(
        self,
        pv_states: Tensor,
        satellite_history: Tensor,
        satellite_frame_mask: Tensor,
        satellite_patch_coords: Tensor,
        *,
        solar_geometry: Optional[Tensor] = None,
        patch_position_embedding: Optional[Tensor] = None,
        solar_patch_embedding: Optional[Tensor] = None,
        site_bias: Optional[Tensor] = None,
        pv_patch_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> SatelliteAdapterOutput:
        encoded = self.encoder(
            satellite_history,
            satellite_frame_mask,
            satellite_patch_coords,
            solar_geometry=solar_geometry,
            patch_position_embedding=patch_position_embedding,
            solar_patch_embedding=solar_patch_embedding,
            site_bias=site_bias,
        )
        fusion = self.fusion(
            pv_states,
            encoded,
            pv_patch_mask=pv_patch_mask,
            return_attention=return_attention,
        )
        return SatelliteAdapterOutput(
            encoded_satellite=encoded,
            fusion=fusion,
        )
