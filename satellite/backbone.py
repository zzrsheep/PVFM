"""The satellite experiment's PVFM path: masked history weather, future NWP.

Reuse the published PVFM layers and parameters. Only the encoder/decoder
boundary is exposed for the satellite residual; no research switches exist.
"""

from dataclasses import dataclass
import torch
from pvfm.model import PVFMForecaster


@dataclass(frozen=True)
class DecodeContext:
    pv_mask: torch.Tensor
    future_tokens: torch.Tensor
    future_patch_mask: torch.Tensor
    future_point_mask: torch.Tensor
    target_len: int


class SatelliteBackbone(PVFMForecaster):
    """Fixed no-history-weather path, retaining future NWP conditioning."""

    def encode_history(
        self,
        *,
        past_target,
        past_observed_mask,
        future_covariates,
        future_covariates_mask,
        past_time_features,
        future_time_features,
        static_features,
        site_features,
    ):
        weather = past_target.new_zeros(*past_target.shape[:2], 6)
        history_values = torch.cat([past_target, weather], dim=-1)
        history_mask = torch.cat(
            [past_observed_mask, torch.zeros_like(weather)], dim=-1
        )
        history_solar = self._build_step_solar_geometry(
            past_time_features, site_features, past_target.shape[1]
        )
        future_solar = self._build_step_solar_geometry(
            future_time_features, site_features, future_covariates.shape[1]
        )
        history_tokens, history_mask = self.history_patch_tokenizer(
            history_values, value_mask=history_mask, solar_time_features=history_solar
        )
        history_tokens = self._inject_site_token_context(
            history_tokens, static_features, site_features
        )
        future_tokens, future_mask = self.future_patch_tokenizer(
            future_covariates,
            value_mask=future_covariates_mask,
            solar_time_features=future_solar,
        )
        future_tokens = self._inject_site_token_context(
            future_tokens, static_features, site_features
        )
        for layer in self.history_encoder_layers:
            history_tokens = layer(
                history_tokens,
                mask=history_mask,
                time_attn_bias=None,
                time_rope_positions=None,
            )
        return history_tokens[:, 0], DecodeContext(
            pv_mask=history_mask[:, 0],
            future_tokens=future_tokens,
            future_patch_mask=future_mask,
            future_point_mask=future_covariates_mask,
            target_len=int(future_covariates.shape[1]),
        )

    def decode_from_pv_states(self, pv_states, context):
        future_tokens = context.future_tokens
        future_mask = context.future_patch_mask
        states = self._project_history_patches_to_future(
            pv_states, future_tokens.shape[2]
        )
        valid = future_mask.permute(0, 2, 1).any(dim=-1).to(future_tokens.dtype)
        for layer in self.future_joint_decoder_layers:
            states, future_tokens = layer(
                states,
                future_tokens,
                pv_mask=valid,
                nwp_mask=future_mask,
                time_attn_bias=None,
                time_rope_positions=None,
            )
        states = self.future_patch_refine(
            states, mask=valid, attn_bias=None, rope_positions=None
        )
        quantiles = self._parameterize_quantile_sequence(
            self._decode_quantiles_to_sequence(states, context.target_len)
        )
        quantiles = quantiles * (context.future_point_mask.sum(dim=-1) > 0).to(
            quantiles.dtype
        ).unsqueeze(-1)
        # This experiment fine-tunes q50 MSE, not quantile calibration.
        return quantiles[..., 5:6]

    def forward(self, **inputs):
        states, context = self.encode_history(**inputs)
        return self.decode_from_pv_states(states, context)
