"""Mask-safe rotary attention used by the isolated satellite branch.

The original PVFM implementation provides a one-dimensional temporal RoPE.
Satellite images also need a genuine two-dimensional RoPE: one rotary
subspace represents east/west displacement and another represents
north/south displacement.  The implementations below deliberately keep
masking explicit so an entirely missing satellite window never reaches a
softmax containing only ``-inf`` values.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _rotate_pairs(value: Tensor) -> Tensor:
    """Rotate adjacent pairs ``(x0, x1) -> (-x1, x0)``."""

    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _validate_token_mask(
    mask: Optional[Tensor],
    *,
    batch: int,
    length: int,
    name: str,
    device: torch.device,
) -> Optional[Tensor]:
    if mask is None:
        return None
    if tuple(mask.shape) != (batch, length):
        raise ValueError(
            f"{name} must have shape {(batch, length)}, got {tuple(mask.shape)}."
        )
    return mask.to(device=device, dtype=torch.bool)


def _safe_key_mask(
    key_mask: Optional[Tensor],
    *,
    batch: int,
    key_length: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Return a softmax-safe mask and whether each row has a real key.

    A temporary first key is enabled for all-masked rows.  The corresponding
    attention output is multiplied by ``has_key`` after attention, so the
    temporary key can never leak into the model state.
    """

    if key_mask is None:
        safe = torch.ones(batch, key_length, dtype=torch.bool, device=device)
        has_key = torch.ones(batch, dtype=torch.bool, device=device)
        return safe, has_key

    safe = key_mask.to(device=device, dtype=torch.bool).clone()
    has_key = safe.any(dim=-1)
    if (~has_key).any():
        safe[~has_key, 0] = True
    return safe, has_key


class RotaryEmbedding1D(nn.Module):
    """One-dimensional RoPE accepting integer or floating-point positions."""

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError(f"1D RoPE dimension must be positive and even, got {dim}.")
        frequencies = 1.0 / (
            float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.dim = int(dim)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _phase(self, positions: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.ndim != 2:
            raise ValueError(
                "1D RoPE positions must have shape [L] or [B,L], "
                f"got {tuple(positions.shape)}."
            )
        angles = (
            positions.to(
                device=self.frequencies.device, dtype=self.frequencies.dtype
            ).unsqueeze(-1)
            * self.frequencies
        )
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    def apply(self, value: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE to ``value`` with shape ``[B,H,L,D]``."""

        if value.ndim != 4 or value.shape[-1] != self.dim:
            raise ValueError(
                f"value must have shape [B,H,L,{self.dim}], got {tuple(value.shape)}."
            )
        cos, sin = self._phase(positions.to(value.device), value.dtype)
        if cos.shape[0] == 1 and value.shape[0] != 1:
            cos = cos.expand(value.shape[0], -1, -1)
            sin = sin.expand(value.shape[0], -1, -1)
        if cos.shape[0] != value.shape[0] or cos.shape[1] != value.shape[-2]:
            raise ValueError(
                "RoPE positions do not match value batch/length: "
                f"positions phase={tuple(cos.shape)}, value={tuple(value.shape)}."
            )
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return value * cos + _rotate_pairs(value) * sin


class RotaryEmbedding2D(nn.Module):
    """Two-dimensional RoPE for normalized local east/north coordinates.

    Half of each attention head encodes east/west displacement and the other
    half encodes north/south displacement.  Coordinates are expected to be
    normalized to a documented physical scale, e.g. ``1600 km -> 1.0``.
    """

    def __init__(self, dim: int, max_frequency: float = 10.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 4:
            raise ValueError(
                "2D RoPE dimension must be a positive multiple of four, " f"got {dim}."
            )
        if max_frequency <= 0:
            raise ValueError("max_frequency must be positive.")
        self.dim = int(dim)
        self.axis_dim = self.dim // 2
        pair_count = self.axis_dim // 2
        scales = torch.linspace(
            1.0,
            float(max_frequency) / 2.0,
            pair_count,
            dtype=torch.float32,
        )
        self.register_buffer("scales", scales * math.pi, persistent=False)

    def _axis_phase(
        self, positions: Tensor, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        angles = (
            positions.to(device=self.scales.device, dtype=self.scales.dtype).unsqueeze(
                -1
            )
            * self.scales
        )
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)

    def apply(self, value: Tensor, coordinates: Tensor) -> Tensor:
        """Apply 2D RoPE to ``value`` with shape ``[B,H,N,D]``."""

        if value.ndim != 4 or value.shape[-1] != self.dim:
            raise ValueError(
                f"value must have shape [B,H,N,{self.dim}], got {tuple(value.shape)}."
            )
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError(
                "coordinates must have shape [N,2] or [B,N,2], "
                f"got {tuple(coordinates.shape)}."
            )
        if coordinates.shape[0] == 1 and value.shape[0] != 1:
            coordinates = coordinates.expand(value.shape[0], -1, -1)
        if (
            coordinates.shape[0] != value.shape[0]
            or coordinates.shape[1] != value.shape[-2]
        ):
            raise ValueError(
                "coordinates do not match value batch/length: "
                f"coordinates={tuple(coordinates.shape)}, value={tuple(value.shape)}."
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must contain only finite values.")

        east, north = coordinates[..., 0], coordinates[..., 1]
        east_cos, east_sin = self._axis_phase(east, value.dtype)
        north_cos, north_sin = self._axis_phase(north, value.dtype)

        east_value = value[..., : self.axis_dim]
        north_value = value[..., self.axis_dim :]
        east_cos, east_sin = east_cos.unsqueeze(1), east_sin.unsqueeze(1)
        north_cos, north_sin = north_cos.unsqueeze(1), north_sin.unsqueeze(1)
        east_value = east_value * east_cos + _rotate_pairs(east_value) * east_sin
        north_value = north_value * north_cos + _rotate_pairs(north_value) * north_sin
        return torch.cat((east_value, north_value), dim=-1)


class SpatialRotaryMultiheadAttention(nn.Module):
    """Multi-head attention whose Query and Key use local-coordinate 2D RoPE."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        max_frequency: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}."
            )
        head_dim = hidden_dim // n_heads
        if head_dim % 4:
            raise ValueError(
                "Each spatial-attention head must be divisible by four; "
                f"received head_dim={head_dim}."
            )
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rotary = RotaryEmbedding2D(head_dim, max_frequency=max_frequency)

    def _heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return (
            value.reshape(batch, length, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: Tensor,
        *,
        query_coordinates: Tensor,
        key_value: Optional[Tensor] = None,
        key_coordinates: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
        key_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if query.ndim != 3 or query.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"query must have shape [B,Q,{self.hidden_dim}], got {tuple(query.shape)}."
            )
        if key_value is None:
            key_value = query
        if key_coordinates is None:
            key_coordinates = query_coordinates
        if key_value.ndim != 3 or key_value.shape[-1] != self.hidden_dim:
            raise ValueError(
                "key_value must have shape "
                f"[B,K,{self.hidden_dim}], got {tuple(key_value.shape)}."
            )
        if key_value.shape[0] != query.shape[0]:
            raise ValueError("query and key_value batch sizes must match.")

        batch, query_length = query.shape[:2]
        key_length = key_value.shape[1]
        query_mask = _validate_token_mask(
            query_mask,
            batch=batch,
            length=query_length,
            name="query_mask",
            device=query.device,
        )
        key_mask = _validate_token_mask(
            key_mask,
            batch=batch,
            length=key_length,
            name="key_mask",
            device=query.device,
        )
        safe_key_mask, has_key = _safe_key_mask(
            key_mask,
            batch=batch,
            key_length=key_length,
            device=query.device,
        )

        q = self.rotary.apply(self._heads(self.q_proj(query)), query_coordinates)
        k = self.rotary.apply(self._heads(self.k_proj(key_value)), key_coordinates)
        v = self._heads(self.v_proj(key_value))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            (~safe_key_mask).unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout)
        output = torch.matmul(attention, v)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .reshape(batch, query_length, self.hidden_dim)
        )
        output = self.out_proj(output)

        output_valid = has_key.unsqueeze(-1)
        if query_mask is not None:
            output_valid = output_valid & query_mask
        output = output * output_valid.unsqueeze(-1).to(output.dtype)
        if return_attention:
            attention = attention * output_valid.unsqueeze(1).unsqueeze(-1).to(
                attention.dtype
            )
            return output, attention
        return output


class TemporalRotarySelfAttention(nn.Module):
    """Mask-safe temporal self-attention with floating-point 1D RoPE positions."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}."
            )
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        if self.head_dim % 2:
            raise ValueError(
                f"Temporal attention head_dim must be even, got {self.head_dim}."
            )
        self.dropout = float(dropout)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rotary = RotaryEmbedding1D(self.head_dim, base=rope_base)

    def _heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return (
            value.reshape(batch, length, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        value: Tensor,
        *,
        positions: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        if value.ndim != 3 or value.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"value must have shape [B,L,{self.hidden_dim}], got {tuple(value.shape)}."
            )
        batch, length = value.shape[:2]
        mask = _validate_token_mask(
            mask,
            batch=batch,
            length=length,
            name="mask",
            device=value.device,
        )
        safe_mask, has_key = _safe_key_mask(
            mask,
            batch=batch,
            key_length=length,
            device=value.device,
        )

        q_raw, k_raw, v_raw = self.qkv_proj(value).chunk(3, dim=-1)
        q = self.rotary.apply(self._heads(q_raw), positions)
        k = self.rotary.apply(self._heads(k_raw), positions)
        v = self._heads(v_raw)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            (~safe_mask).unsqueeze(1).unsqueeze(2), float("-inf")
        )
        attention = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout)
        output = torch.matmul(attention, v)
        output = (
            output.transpose(1, 2).contiguous().reshape(batch, length, self.hidden_dim)
        )
        output = self.out_proj(output)

        query_valid = safe_mask if mask is None else mask
        output_valid = query_valid & has_key.unsqueeze(-1)
        return output * output_valid.unsqueeze(-1).to(output.dtype)


class MaskedAttentionPool(nn.Module):
    """Pool a masked temporal sequence with a learned content query."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.empty(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = float(dropout)
        nn.init.normal_(self.query, std=hidden_dim**-0.5)

    def forward(
        self,
        value: Tensor,
        *,
        mask: Tensor,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if value.ndim != 3 or value.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"value must have shape [B,L,{self.hidden_dim}], got {tuple(value.shape)}."
            )
        batch, length = value.shape[:2]
        mask = _validate_token_mask(
            mask,
            batch=batch,
            length=length,
            name="mask",
            device=value.device,
        )
        assert mask is not None
        safe_mask, has_key = _safe_key_mask(
            mask,
            batch=batch,
            key_length=length,
            device=value.device,
        )
        keys = torch.tanh(self.key_proj(value))
        scores = torch.einsum("bld,d->bl", keys, self.query.to(keys.dtype))
        scores = scores / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~safe_mask, float("-inf"))
        attention = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        if self.training and self.dropout:
            attention = F.dropout(attention, p=self.dropout)
        pooled = torch.einsum("bl,bld->bd", attention, self.value_proj(value))
        pooled = self.out_proj(pooled)
        pooled = pooled * has_key.unsqueeze(-1).to(pooled.dtype)
        if return_attention:
            attention = attention * has_key.unsqueeze(-1).to(attention.dtype)
            return pooled, attention
        return pooled
