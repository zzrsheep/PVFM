"""Legacy external quantile output head for unsupported baselines.

The wrapped forecasting model keeps its original input contract and produces a
point forecast tensor.  This adapter only replaces the final output contract
Native-q9 variants are preferred for trainable PVFM baselines.  This adapter is
kept only as a compatibility fallback for models that do not expose a native
quantile head; it maps an already-produced point forecast to quantiles.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def parse_quantile_levels(value):
    """Parse and validate a comma-separated or iterable quantile list."""
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = [] if value is None else list(value)
    levels = tuple(float(item) for item in values)
    if len(levels) < 2:
        raise ValueError("At least two quantile levels are required.")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError(f"Quantile levels must lie strictly between 0 and 1: {levels}")
    if tuple(sorted(levels)) != levels or len(set(levels)) != len(levels):
        raise ValueError(f"Quantile levels must be strictly increasing: {levels}")
    if not any(math.isclose(level, 0.5, rel_tol=0.0, abs_tol=1e-6) for level in levels):
        raise ValueError("Quantile levels must include q=0.5 for point-metric compatibility.")
    return levels


class QuantileOutputWrapper(nn.Module):
    """Attach an FM-compatible multi-quantile head to an existing model.

    ``base_model`` is untouched.  Its forecast output channels are treated as
    features for the new head, which lets a fused baseline use the channels
    produced from future NWP without introducing a model-family-specific
    adapter.  The wrapper returns ``(q50, auxiliary)`` where
    ``auxiliary["quantiles"]`` has shape ``[B, H, Q]``.
    """

    def __init__(
        self,
        base_model,
        output_dim,
        pred_len,
        quantile_levels,
        quantile_parameterization="independent",
        quantile_increment_scale=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.output_dim = int(output_dim)
        self.pred_len = int(pred_len)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.pred_len <= 0:
            raise ValueError(f"pred_len must be positive, got {self.pred_len}")

        self.quantile_levels = parse_quantile_levels(quantile_levels)
        self.quantile_parameterization = str(quantile_parameterization).strip().lower()
        if self.quantile_parameterization not in {"independent", "noncrossing"}:
            raise ValueError("quantile_parameterization must be independent|noncrossing.")
        self.quantile_increment_scale = float(quantile_increment_scale)
        if self.quantile_increment_scale <= 0.0:
            raise ValueError("quantile_increment_scale must be positive.")

        self.median_index = min(
            range(len(self.quantile_levels)),
            key=lambda index: abs(self.quantile_levels[index] - 0.5),
        )
        self.quantile_head = nn.Linear(self.output_dim, len(self.quantile_levels))
        # Start from the base model's target forecast.  This keeps the q50
        # point path numerically close to the old deterministic baseline at
        # the beginning of an opt-in probabilistic run, while the head can
        # learn non-zero predictive spread through pinball loss.  In the
        # noncrossing parameterization, only q50 is initialized from the base
        # forecast; the other rows are initialized to a tiny positive gap.
        # Reusing the identity initialization for every raw row there would
        # turn the base forecast itself into a large, sign-dependent interval
        # after softplus.
        with torch.no_grad():
            self.quantile_head.weight.zero_()
            self.quantile_head.bias.zero_()
            if self.quantile_parameterization == "independent":
                self.quantile_head.weight[:, -1] = 1.0
            else:
                self.quantile_head.weight[self.median_index, -1] = 1.0
                initial_gap = min(1.0e-3, self.quantile_increment_scale * 1.0e-2)
                raw_gap = math.log(math.expm1(initial_gap / self.quantile_increment_scale))
                for index in range(len(self.quantile_levels)):
                    if index != self.median_index:
                        self.quantile_head.bias[index] = raw_gap

    @property
    def last_aux_loss(self):
        """Expose auxiliary losses from a wrapped model, if it has one."""
        return getattr(self.base_model, "last_aux_loss", None)

    def _parameterize_quantiles(self, values):
        if self.quantile_parameterization == "independent":
            return values

        median = values[..., self.median_index:self.median_index + 1]
        scale = self.quantile_increment_scale

        left_raw = values[..., :self.median_index]
        if left_raw.shape[-1]:
            left_steps = F.softplus(left_raw) * scale
            left_distances = torch.flip(
                torch.cumsum(torch.flip(left_steps, dims=(-1,)), dim=-1),
                dims=(-1,),
            )
            left = median - left_distances
        else:
            left = values[..., :0]

        right_raw = values[..., self.median_index + 1:]
        if right_raw.shape[-1]:
            right_steps = F.softplus(right_raw) * scale
            right = median + torch.cumsum(right_steps, dim=-1)
        else:
            right = values[..., :0]
        return torch.cat((left, median, right), dim=-1)

    @staticmethod
    def _split_base_output(output):
        if isinstance(output, (tuple, list)) and len(output) == 2 and torch.is_tensor(output[0]):
            auxiliary = output[1] if isinstance(output[1], dict) else {}
            return output[0], auxiliary
        if not torch.is_tensor(output):
            raise TypeError(
                "Probabilistic output wrapper requires the base model to return "
                f"a tensor, got {type(output).__name__}."
            )
        return output, {}

    def forward(self, *args, **kwargs):
        base_output, base_auxiliary = self._split_base_output(self.base_model(*args, **kwargs))
        if base_output.ndim != 3:
            raise ValueError(
                "Probabilistic output wrapper requires base output [B,H,C], "
                f"got {tuple(base_output.shape)}."
            )
        if base_output.shape[-1] != self.output_dim:
            raise ValueError(
                "Base output width changed after wrapper construction: "
                f"expected {self.output_dim}, got {base_output.shape[-1]}."
            )
        if base_output.shape[1] < self.pred_len:
            raise ValueError(
                "Base output is shorter than pred_len: "
                f"output_length={base_output.shape[1]}, pred_len={self.pred_len}."
            )

        states = base_output[:, -self.pred_len:, :]
        quantiles = self._parameterize_quantiles(self.quantile_head(states))
        point = quantiles[..., self.median_index:self.median_index + 1]
        auxiliary = dict(base_auxiliary)
        auxiliary.update(
            {
                "quantiles": quantiles,
                "quantile_levels": self.quantile_levels,
            }
        )
        return point, auxiliary
