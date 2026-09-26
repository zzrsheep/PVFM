"""Shared non-learned utilities for native q9 model variants."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def parse_quantile_levels(value: str | Iterable[float] | None) -> tuple[float, ...]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = [] if value is None else list(value)
    levels = tuple(float(item) for item in values)
    if len(levels) < 2:
        raise ValueError("At least two quantile levels are required.")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError(f"Quantile levels must lie strictly between 0 and 1: {levels}")
    if tuple(sorted(levels)) != levels or len(set(levels)) != len(levels):
        raise ValueError(f"Quantile levels must be strictly increasing: {levels}")
    if not any(math.isclose(level, 0.5, abs_tol=1e-6) for level in levels):
        raise ValueError("Quantile levels must include q=0.5.")
    return levels


class NativeQuantileMixin:
    """Shape and metadata helpers for model-internal quantile heads.

    The mixin deliberately contains no trainable layer.  Concrete variants
    create their own ``native_quantile_head`` in place of the original point
    head and call :meth:`_native_quantile_output` after decoding.
    """

    supports_native_quantiles = True

    def _init_native_quantiles(self, configs) -> None:
        levels = getattr(configs, "quantiles", ())
        self.quantile_levels = parse_quantile_levels(levels) if levels else ()
        self.num_quantiles = len(self.quantile_levels)
        self.head_mode = "native_quantile" if self.num_quantiles else "deterministic_original"
        self.quantile_median_index = (
            min(range(self.num_quantiles), key=lambda i: abs(self.quantile_levels[i] - 0.5))
            if self.num_quantiles else -1
        )
        self.quantile_parameterization = str(
            getattr(configs, "quantile_parameterization", "independent")
        ).strip().lower()
        if self.quantile_parameterization not in {"independent", "noncrossing"}:
            raise ValueError("quantile_parameterization must be independent|noncrossing")
        self.quantile_increment_scale = float(getattr(configs, "quantile_increment_scale", 0.1))
        if self.quantile_increment_scale <= 0:
            raise ValueError("quantile_increment_scale must be positive")
        # Audit metadata: concrete adapters may refine the original head
        # description, but every native-q9 model has this common contract.
        self.original_head_shape = "model-specific point head (preserved)"
        self.native_q9_head_shape = "[B,H,Q]"

    def _parameterize_native_quantiles(self, values: torch.Tensor) -> torch.Tensor:
        if not self.num_quantiles or self.quantile_parameterization == "independent":
            return values
        m = self.quantile_median_index
        median = values[..., m:m + 1]
        scale = self.quantile_increment_scale
        left_raw = values[..., :m]
        right_raw = values[..., m + 1:]
        if left_raw.shape[-1]:
            left_steps = F.softplus(left_raw) * scale
            left = median - torch.flip(
                torch.cumsum(torch.flip(left_steps, dims=(-1,)), dim=-1), dims=(-1,)
            )
        else:
            left = values[..., :0]
        if right_raw.shape[-1]:
            right = median + torch.cumsum(F.softplus(right_raw) * scale, dim=-1)
        else:
            right = values[..., :0]
        return torch.cat((left, median, right), dim=-1)

    def _native_quantile_output(self, values: torch.Tensor):
        """Return the PVFM-compatible ``(q50, auxiliary)`` tuple."""
        if values.ndim != 3 or values.shape[-1] != self.num_quantiles:
            raise ValueError(
                "Native quantile head must return [B,H,Q], "
                f"got {tuple(values.shape)} for Q={self.num_quantiles}."
            )
        values = self._parameterize_native_quantiles(values)
        median = values[..., self.quantile_median_index:self.quantile_median_index + 1]
        return median, {"quantiles": values, "quantile_levels": self.quantile_levels}
