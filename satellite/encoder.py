"""Factorised spatial/temporal encoder for station-centred satellite images."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from satellite.attention import (
    MaskedAttentionPool,
    SpatialRotaryMultiheadAttention,
    TemporalRotarySelfAttention,
)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(hidden_dim) * int(expansion)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class SpatialEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        n_heads: int,
        dropout: float,
        max_frequency: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = SpatialRotaryMultiheadAttention(
            hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_frequency=max_frequency,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(
        self,
        value: Tensor,
        *,
        coordinates: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        normalized = self.attention_norm(value)
        value = value + self.attention(
            normalized,
            query_coordinates=coordinates,
            query_mask=token_mask,
            key_mask=token_mask,
        )
        value = value + self.ffn(self.ffn_norm(value))
        return value * token_mask.unsqueeze(-1).to(value.dtype)


class TemporalEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = TemporalRotarySelfAttention(
            hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(
        self,
        value: Tensor,
        *,
        positions: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        value = value + self.attention(
            self.attention_norm(value),
            positions=positions,
            mask=token_mask,
        )
        value = value + self.ffn(self.ffn_norm(value))
        return value * token_mask.unsqueeze(-1).to(value.dtype)


@dataclass(frozen=True)
class SatelliteEncoderOutput:
    """Outputs retained for fusion, diagnostics, and mask verification."""

    memory: Tensor
    patch_coordinates: Tensor
    window_frame_mask: Tensor
    window_has_satellite: Tensor
    valid_fraction: Tensor
    daylight_fraction: Tensor
    temporal_pool_attention: Tensor


class SatellitePatchEncoder(nn.Module):
    """Encode historical satellite frames into dynamically aligned PVFM windows.

    The image content, validity mask, and coordinates have intentionally
    separate paths:

    * image values produce content tokens through ``Conv2d``;
    * the frame mask controls zeroing, temporal attention, and pooling;
    * local east/north coordinates affect spatial Query/Key through 2D RoPE.
    """

    def __init__(
        self,
        *,
        image_channels: int = 4,
        image_patch_size: int | Tuple[int, int] = 8,
        satellite_dim: int = 128,
        output_dim: int = 256,
        spatial_layers: int = 1,
        temporal_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
        temporal_patch_len: int = 12,
        temporal_patch_stride: int = 6,
        spatial_rope_max_frequency: float = 10.0,
        channel_mean: Optional[Sequence[float]] = None,
        channel_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if isinstance(image_patch_size, int):
            image_patch_size = (int(image_patch_size), int(image_patch_size))
        else:
            image_patch_size = tuple(map(int, image_patch_size))
        if len(image_patch_size) != 2 or min(image_patch_size) <= 0:
            raise ValueError("image_patch_size must contain two positive values.")
        if temporal_patch_len <= 0 or temporal_patch_stride <= 0:
            raise ValueError("Temporal patch length and stride must be positive.")
        if spatial_layers < 0 or temporal_layers < 0:
            raise ValueError("Encoder layer counts must be non-negative.")

        self.image_channels = int(image_channels)
        self.image_patch_size = image_patch_size
        self.satellite_dim = int(satellite_dim)
        self.output_dim = int(output_dim)
        self.temporal_patch_len = int(temporal_patch_len)
        self.temporal_patch_stride = int(temporal_patch_stride)

        if channel_mean is None:
            channel_mean = [0.0] * self.image_channels
        if channel_std is None:
            channel_std = [1.0] * self.image_channels
        if (
            len(channel_mean) != self.image_channels
            or len(channel_std) != self.image_channels
        ):
            raise ValueError("channel_mean and channel_std must match image_channels.")
        mean = torch.tensor(channel_mean, dtype=torch.float32)
        std = torch.tensor(channel_std, dtype=torch.float32)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Channel normalization values must be finite.")
        if (std <= 0).any():
            raise ValueError("All channel standard deviations must be positive.")
        self.register_buffer(
            "channel_mean",
            mean.reshape(1, 1, self.image_channels, 1, 1),
        )
        self.register_buffer(
            "channel_std",
            std.reshape(1, 1, self.image_channels, 1, 1),
        )

        self.patch_embedding = nn.Conv2d(
            self.image_channels,
            self.satellite_dim,
            kernel_size=self.image_patch_size,
            stride=self.image_patch_size,
        )
        self.patch_norm = nn.LayerNorm(self.satellite_dim)
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialEncoderBlock(
                    self.satellite_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_frequency=spatial_rope_max_frequency,
                )
                for _ in range(int(spatial_layers))
            ]
        )
        solar_hidden = max(16, self.satellite_dim // 2)
        self.hour_solar_projection = nn.Sequential(
            nn.Linear(4, solar_hidden),
            nn.GELU(),
            nn.Linear(solar_hidden, self.satellite_dim),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalEncoderBlock(
                    self.satellite_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(int(temporal_layers))
            ]
        )
        self.temporal_pool = MaskedAttentionPool(self.satellite_dim, dropout=dropout)
        self.output_projection = nn.Linear(self.satellite_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

    @staticmethod
    def _expand_patch_anchor(
        value: Optional[Tensor],
        *,
        batch: int,
        patches: int,
        hidden: int,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if value is None:
            return None
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or value.shape[1:] != (patches, hidden):
            raise ValueError(
                f"{name} must have shape [P,D], [1,P,D], or [B,P,D]; "
                f"expected P={patches}, D={hidden}, got {tuple(value.shape)}."
            )
        if value.shape[0] == 1 and batch != 1:
            value = value.expand(batch, -1, -1)
        if value.shape[0] != batch:
            raise ValueError(
                f"{name} batch must be one or {batch}, got {value.shape[0]}."
            )
        return value.to(device=device, dtype=dtype)

    @staticmethod
    def _expand_coordinates(
        coordinates: Tensor,
        *,
        batch: int,
        tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (tokens, 2):
            raise ValueError(
                "satellite_patch_coords must have shape [N,2], [1,N,2], "
                f"or [B,N,2]; expected N={tokens}, got {tuple(coordinates.shape)}."
            )
        if coordinates.shape[0] == 1 and batch != 1:
            coordinates = coordinates.expand(batch, -1, -1)
        if coordinates.shape[0] != batch:
            raise ValueError(
                f"Coordinate batch must be one or {batch}, got {coordinates.shape[0]}."
            )
        coordinates = coordinates.to(device=device, dtype=dtype)
        if not torch.isfinite(coordinates).all():
            raise ValueError("satellite_patch_coords must be finite.")
        return coordinates

    def _make_windows(
        self,
        tokens: Tensor,
        frame_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        time_steps = tokens.shape[1]
        if time_steps <= 0:
            raise ValueError("Satellite history must contain at least one step.")
        # Match HistoryPatchTokenizer exactly: use ceil window count and
        # right-pad incomplete final windows.  Padded frames are always masked.
        patches = max(
            1,
            math.ceil(
                max(0, time_steps - self.temporal_patch_len)
                / self.temporal_patch_stride
            )
            + 1,
        )
        padded_steps = max(
            self.temporal_patch_len,
            (patches - 1) * self.temporal_patch_stride + self.temporal_patch_len,
        )
        padding = padded_steps - time_steps
        if padding:
            tokens = torch.cat(
                (
                    tokens,
                    tokens.new_zeros(
                        tokens.shape[0],
                        padding,
                        tokens.shape[2],
                        tokens.shape[3],
                    ),
                ),
                dim=1,
            )
            frame_mask = torch.cat(
                (
                    frame_mask,
                    torch.zeros(
                        frame_mask.shape[0],
                        padding,
                        dtype=torch.bool,
                        device=frame_mask.device,
                    ),
                ),
                dim=1,
            )
        token_windows = tokens.unfold(
            1, self.temporal_patch_len, self.temporal_patch_stride
        )
        # torch.unfold appends the window dimension at the end:
        # [B,P,N,D,L] -> [B,P,N,L,D].
        token_windows = token_windows.permute(0, 1, 2, 4, 3).contiguous()
        mask_windows = frame_mask.unfold(
            1, self.temporal_patch_len, self.temporal_patch_stride
        ).contiguous()
        hour_positions = torch.arange(
            padded_steps,
            device=tokens.device,
            dtype=torch.float32,
        ) / float(self.temporal_patch_stride)
        position_windows = hour_positions.unfold(
            0, self.temporal_patch_len, self.temporal_patch_stride
        ).contiguous()
        return token_windows, mask_windows, position_windows

    def forward(
        self,
        satellite_history: Tensor,
        satellite_frame_mask: Tensor,
        satellite_patch_coords: Tensor,
        *,
        solar_geometry: Optional[Tensor] = None,
        patch_position_embedding: Optional[Tensor] = None,
        solar_patch_embedding: Optional[Tensor] = None,
        site_bias: Optional[Tensor] = None,
    ) -> SatelliteEncoderOutput:
        if satellite_history.ndim != 5:
            raise ValueError(
                "satellite_history must have shape [B,T,C,H,W], "
                f"got {tuple(satellite_history.shape)}."
            )
        batch, time_steps, channels, height, width = satellite_history.shape
        if channels != self.image_channels:
            raise ValueError(
                f"Expected {self.image_channels} satellite channels, got {channels}."
            )
        if tuple(satellite_frame_mask.shape) != (batch, time_steps):
            raise ValueError(
                "satellite_frame_mask must have shape [B,T], "
                f"got {tuple(satellite_frame_mask.shape)}."
            )
        patch_height, patch_width = self.image_patch_size
        if height % patch_height or width % patch_width:
            raise ValueError(
                f"Satellite image {(height, width)} is not divisible by "
                f"patch size {self.image_patch_size}."
            )

        frame_mask = satellite_frame_mask.to(
            device=satellite_history.device, dtype=torch.bool
        )
        image = torch.nan_to_num(
            satellite_history,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        image = (
            image - self.channel_mean.to(device=image.device, dtype=image.dtype)
        ) / self.channel_std.to(device=image.device, dtype=image.dtype)
        image = torch.where(
            frame_mask[:, :, None, None, None],
            image,
            torch.zeros((), device=image.device, dtype=image.dtype),
        )

        embedded = self.patch_embedding(
            image.reshape(batch * time_steps, channels, height, width)
        )
        spatial_height, spatial_width = embedded.shape[-2:]
        spatial_tokens = spatial_height * spatial_width
        embedded = embedded.flatten(2).transpose(1, 2)
        embedded = self.patch_norm(embedded)

        spatial_mask = (
            frame_mask[:, :, None]
            .expand(batch, time_steps, spatial_tokens)
            .reshape(batch * time_steps, spatial_tokens)
        )
        embedded = embedded * spatial_mask.unsqueeze(-1).to(embedded.dtype)
        coordinates = self._expand_coordinates(
            satellite_patch_coords,
            batch=batch,
            tokens=spatial_tokens,
            device=embedded.device,
            dtype=torch.float32,
        )
        spatial_coordinates = (
            coordinates[:, None, :, :]
            .expand(batch, time_steps, spatial_tokens, 2)
            .reshape(batch * time_steps, spatial_tokens, 2)
        )
        for block in self.spatial_blocks:
            embedded = block(
                embedded,
                coordinates=spatial_coordinates,
                token_mask=spatial_mask,
            )
        encoded = embedded.reshape(
            batch, time_steps, spatial_tokens, self.satellite_dim
        )

        if solar_geometry is None:
            solar = encoded.new_zeros(batch, time_steps, 4)
        else:
            if solar_geometry.ndim != 3 or solar_geometry.shape[:2] != (
                batch,
                time_steps,
            ):
                raise ValueError(
                    "solar_geometry must have shape [B,T,4], "
                    f"got {tuple(solar_geometry.shape)}."
                )
            if solar_geometry.shape[-1] < 4:
                raise ValueError("solar_geometry must have at least four channels.")
            solar = torch.nan_to_num(
                solar_geometry[..., :4].to(device=encoded.device, dtype=encoded.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        encoded = encoded + self.hour_solar_projection(solar).unsqueeze(2)
        encoded = encoded * frame_mask[:, :, None, None].to(encoded.dtype)

        token_windows, window_mask, position_windows = self._make_windows(
            encoded, frame_mask
        )
        patches = token_windows.shape[1]
        flat_windows = token_windows.reshape(
            batch * patches * spatial_tokens,
            self.temporal_patch_len,
            self.satellite_dim,
        )
        flat_mask = (
            window_mask[:, :, None, :]
            .expand(batch, patches, spatial_tokens, self.temporal_patch_len)
            .reshape(batch * patches * spatial_tokens, self.temporal_patch_len)
        )
        flat_positions = (
            position_windows[None, :, None, :]
            .expand(batch, patches, spatial_tokens, self.temporal_patch_len)
            .reshape(batch * patches * spatial_tokens, self.temporal_patch_len)
        )
        for block in self.temporal_blocks:
            flat_windows = block(
                flat_windows,
                positions=flat_positions,
                token_mask=flat_mask,
            )
        pooled, pool_attention = self.temporal_pool(
            flat_windows,
            mask=flat_mask,
            return_attention=True,
        )

        pooled = pooled.reshape(batch, patches, spatial_tokens, self.satellite_dim)
        pool_attention = pool_attention.reshape(
            batch, patches, spatial_tokens, self.temporal_patch_len
        )

        memory = self.output_projection(pooled)
        patch_position = self._expand_patch_anchor(
            patch_position_embedding,
            batch=batch,
            patches=patches,
            hidden=self.output_dim,
            name="patch_position_embedding",
            device=memory.device,
            dtype=memory.dtype,
        )
        solar_patch = self._expand_patch_anchor(
            solar_patch_embedding,
            batch=batch,
            patches=patches,
            hidden=self.output_dim,
            name="solar_patch_embedding",
            device=memory.device,
            dtype=memory.dtype,
        )
        if patch_position is not None:
            memory = memory + patch_position.unsqueeze(2)
        if solar_patch is not None:
            memory = memory + solar_patch.unsqueeze(2)
        memory = self.output_norm(memory)

        if site_bias is not None:
            if site_bias.ndim != 2 or site_bias.shape != (batch, self.output_dim):
                raise ValueError(
                    f"site_bias must have shape {(batch, self.output_dim)}, "
                    f"got {tuple(site_bias.shape)}."
                )
            memory = (
                memory
                + site_bias.to(device=memory.device, dtype=memory.dtype)[
                    :, None, None, :
                ]
            )

        window_has_satellite = window_mask.any(dim=-1)
        valid_fraction = window_mask.to(memory.dtype).mean(dim=-1)
        memory = memory * window_has_satellite[:, :, None, None].to(memory.dtype)
        daylight = (solar[..., 0] > 0).to(memory.dtype)
        padded_steps = (
            patches - 1
        ) * self.temporal_patch_stride + self.temporal_patch_len
        if daylight.shape[1] < padded_steps:
            daylight = torch.cat(
                (
                    daylight,
                    daylight.new_zeros(batch, padded_steps - daylight.shape[1]),
                ),
                dim=1,
            )
        daylight_fraction = daylight.unfold(
            1, self.temporal_patch_len, self.temporal_patch_stride
        ).mean(dim=-1)
        return SatelliteEncoderOutput(
            memory=memory,
            patch_coordinates=coordinates,
            window_frame_mask=window_mask,
            window_has_satellite=window_has_satellite,
            valid_fraction=valid_fraction,
            daylight_fraction=daylight_fraction,
            temporal_pool_attention=pool_attention,
        )
