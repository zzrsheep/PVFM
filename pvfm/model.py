"""Published PVFM graph. No experimental architecture switches.

Historical PV/weather -> patch + variable + absolute-position + solar tokens
-> window/site conditioning -> factorized encoder -> future projection
-> PV/NWP decoder -> future self-attention refinement -> ordered Q11.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from .layers import HistoryPatchTokenizer, ResidualAttentionBlock
from .blocks import EncoderBlock, DecoderBlock

QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


class PVFMForecaster(nn.Module):
    """PVFM: historical factorized encoder and future NWP-conditioned decoder."""

    def __init__(
        self,
        hidden_dim=128,
        n_heads=8,
        n_layers=6,
        future_joint_decoder_layers=6,
        dropout=0.1,
        fixed_seq_len=672,
        max_pred_len=168,
        patch_len=12,
        patch_stride=6,
        future_patch_len=12,
        future_patch_stride=6,
    ):
        super().__init__()
        if (
            min(
                fixed_seq_len,
                max_pred_len,
                patch_len,
                patch_stride,
                future_patch_len,
                future_patch_stride,
                n_layers,
                future_joint_decoder_layers,
            )
            <= 0
        ):
            raise ValueError("Lengths and layer counts must be positive")
        self.hidden_dim = int(hidden_dim)
        self.fixed_seq_len = int(fixed_seq_len)
        self.max_pred_len = int(max_pred_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.future_patch_len = int(future_patch_len)
        self.future_patch_stride = int(future_patch_stride)
        self.probabilistic_quantiles = QUANTILES
        with torch.random.fork_rng(devices=[]):
            self.site_token_adapter = nn.Sequential(
                nn.Linear(8, max(self.hidden_dim, 16)),
                nn.GELU(),
                nn.Linear(max(self.hidden_dim, 16), self.hidden_dim),
            )
        nn.init.zeros_(self.site_token_adapter[-1].weight)
        nn.init.zeros_(self.site_token_adapter[-1].bias)
        self.history_patch_tokenizer = HistoryPatchTokenizer(
            self.patch_len,
            self.patch_stride,
            self.hidden_dim,
            7,
            fixed_seq_len=self.fixed_seq_len,
            dropout=dropout,
        )
        self.future_patch_tokenizer = HistoryPatchTokenizer(
            self.future_patch_len,
            self.future_patch_stride,
            self.hidden_dim,
            6,
            fixed_seq_len=self.max_pred_len,
            dropout=dropout,
        )
        self.history_encoder_layers = nn.ModuleList(
            [
                EncoderBlock(self.hidden_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.max_history_patches = self._num_patches(self.fixed_seq_len)
        self.max_future_patches = self._num_patches_with(
            self.max_pred_len, self.future_patch_len, self.future_patch_stride
        )
        self.history_to_future_patch = nn.Linear(
            self.max_history_patches, self.max_future_patches
        )
        self.future_joint_decoder_layers = nn.ModuleList(
            [
                DecoderBlock(self.hidden_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(future_joint_decoder_layers)
            ]
        )
        self.future_patch_refine = ResidualAttentionBlock(
            self.hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=True
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.quantile_patch_head = nn.Linear(
            self.hidden_dim, self.future_patch_len * 11
        )

    def _num_patches(self, seq_len):
        return self._num_patches_with(seq_len, self.patch_len, self.patch_stride)

    @staticmethod
    def _num_patches_with(seq_len, patch_len, patch_stride):
        seq_len, patch_len, patch_stride = (
            int(seq_len),
            int(patch_len),
            int(patch_stride),
        )
        if seq_len <= 0:
            return 0
        return max(1, math.ceil(max(0, seq_len - patch_len) / patch_stride) + 1)

    def _solar_geometry_from_time_features(self, time_features, lat, lon_and_tz, steps):
        if time_features is None or time_features.shape[-1] < 4:
            return lat.new_zeros(lat.shape[0], int(steps), 4)
        if time_features.shape[-1] != 5:
            raise ValueError("Expected minute/hour/weekday/day/day-of-year features")
        hour_index = 1
        hour = (time_features[..., hour_index] + 0.5) * 23.0
        minute = (time_features[..., 0] + 0.5) * 59.0
        hour = hour + minute / 60.0
        day = (time_features[..., -1] + 0.5) * 365.0 + 1.0
        tz = (
            lon_and_tz[..., 1:2]
            if lon_and_tz.shape[-1] > 1
            else lat.new_zeros(lat.shape[0], 1)
        )
        lon = lon_and_tz[..., 0:1]
        solar_hour = (
            hour - tz.squeeze(-1).unsqueeze(1) + lon.squeeze(-1).unsqueeze(1) / 15.0
        )
        declination = 23.45 * torch.sin(2 * math.pi * (284.0 + day) / 365.0)
        hour_angle = 15.0 * (solar_hour - 12.0)
        elevation = torch.clamp(
            torch.sin(torch.deg2rad(lat)) * torch.sin(torch.deg2rad(declination))
            + torch.cos(torch.deg2rad(lat))
            * torch.cos(torch.deg2rad(declination))
            * torch.cos(torch.deg2rad(hour_angle)),
            min=0.0,
        )
        angle = torch.deg2rad(hour_angle)
        return torch.stack(
            [elevation, 1.0 - elevation, torch.sin(angle), torch.cos(angle)], dim=-1
        ).float()

    def _build_step_solar_geometry(self, time_features, site_features, steps):
        if time_features is None or steps <= 0:
            return None
        if site_features is None:
            site_features = time_features.new_zeros(time_features.shape[0], 2)
        lat, lon = (site_features[..., :1], site_features[..., 1:2])
        tz = (
            site_features[..., 4:5]
            if site_features.shape[-1] > 4
            else site_features.new_zeros(site_features.shape[0], 1)
        )
        return self._solar_geometry_from_time_features(
            time_features, lat, torch.cat([lon, tz], dim=-1), steps
        )

    def _project_history_patches_to_future(self, history_states, future_patch_count):
        batch, history_patch_count, hidden = history_states.shape
        states = history_states.transpose(1, 2)
        if (
            history_patch_count != self.max_history_patches
            and self.max_history_patches > 0
        ):
            states = F.interpolate(
                states,
                size=self.max_history_patches,
                mode="linear",
                align_corners=False,
            )
        states = self.history_to_future_patch(states)
        if states.shape[-1] != future_patch_count:
            states = F.interpolate(
                states, size=future_patch_count, mode="linear", align_corners=False
            )
        return states.transpose(1, 2).reshape(batch, future_patch_count, hidden)

    def _inject_site_token_context(self, tokens, static_features, site_features):
        batch, dtype, device = (tokens.shape[0], tokens.dtype, tokens.device)
        static = (
            tokens.new_zeros(batch, 4)
            if static_features is None
            else torch.nan_to_num(
                static_features.to(device=device, dtype=dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        if static.shape[-1] < 4:
            static = F.pad(static, (0, 4 - static.shape[-1]))
        static = static[:, :4]
        site = (
            torch.zeros(batch, 5, device=device, dtype=torch.float32)
            if site_features is None
            else torch.nan_to_num(
                site_features.to(device=device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        if site.shape[-1] < 5:
            site = F.pad(site, (0, 5 - site.shape[-1]))
        lat = (site[:, :1] / 90.0).clamp(-1.0, 1.0)
        longitude = torch.deg2rad(site[:, 1:2])
        timezone = (site[:, 4:5] / 14.0).clamp(-1.0, 1.0)
        physical = torch.cat(
            [lat, torch.sin(longitude), torch.cos(longitude), timezone], dim=-1
        ).to(dtype)
        bias = self.site_token_adapter(torch.cat([static, physical], dim=-1)).to(dtype)
        return tokens + 1.0 * bias[:, None, None, :]

    def _decode_quantiles_to_sequence(self, states, target_len):
        quantile_patch_head = self.quantile_patch_head
        future_patch_len = self.future_patch_len
        future_patch_stride = self.future_patch_stride
        batch, patches, _ = states.shape
        values = quantile_patch_head(self.output_norm(states)).view(
            batch, patches, future_patch_len, 11
        )
        total = max(
            future_patch_len, (patches - 1) * future_patch_stride + future_patch_len
        )
        sequence = values.new_zeros(batch, total, 11)
        counts = values.new_zeros(batch, total, 1)
        for index in range(patches):
            start = index * future_patch_stride
            end = start + future_patch_len
            sequence[:, start:end] += values[:, index]
            counts[:, start:end] += 1.0
        return (sequence / counts.clamp_min(1.0))[:, :target_len]

    def _parameterize_quantile_sequence(self, values):
        """A free median plus positive softplus increments yields ordered Q11."""
        median_index = 5
        median = values[..., median_index : median_index + 1]
        scale = 0.1
        left_raw = values[..., :median_index]
        if left_raw.shape[-1]:
            left_steps = F.softplus(left_raw) * scale
            left_distances = torch.flip(
                torch.cumsum(torch.flip(left_steps, dims=(-1,)), dim=-1), dims=(-1,)
            )
            left = median - left_distances
        else:
            left = values[..., :0]
        right_raw = values[..., median_index + 1 :]
        if right_raw.shape[-1]:
            right_steps = F.softplus(right_raw) * scale
            right = median + torch.cumsum(right_steps, dim=-1)
        else:
            right = values[..., :0]
        return torch.cat((left, median, right), dim=-1)

    def forward(
        self,
        *,
        past_target,
        past_observed_mask,
        historical_covariates,
        historical_covariates_mask,
        future_covariates,
        future_covariates_mask,
        past_time_features,
        future_time_features,
        static_features,
        site_features,
    ):
        """Accept observed history and known future weather, never future PV labels."""
        if past_target is None or past_observed_mask is None:
            raise ValueError("PVFM requires past_target and past_observed_mask.")
        if future_covariates is None or future_covariates.shape[-1] != 6:
            received = 0 if future_covariates is None else future_covariates.shape[-1]
            raise ValueError(f"PVFM expected 6 future covariates, received {received}.")
        if historical_covariates is not None and historical_covariates.shape[-1] != 6:
            raise ValueError(
                "historical_covariates dimension does not match the frozen PVFM configuration."
            )
        if historical_covariates is None:
            raise ValueError(
                "PVFM was configured with historical covariates but none were supplied."
            )
        covariate_mask = (
            historical_covariates_mask
            if historical_covariates_mask is not None
            else torch.ones_like(historical_covariates)
        )
        history_values = torch.cat([past_target, historical_covariates], dim=-1)
        history_mask = torch.cat([past_observed_mask, covariate_mask], dim=-1)
        history_solar = self._build_step_solar_geometry(
            past_time_features, site_features, past_target.shape[1]
        )
        future_solar = self._build_step_solar_geometry(
            future_time_features, site_features, future_covariates.shape[1]
        )
        history_tokens, history_patch_mask = self.history_patch_tokenizer(
            history_values, value_mask=history_mask, solar_time_features=history_solar
        )
        history_tokens = self._inject_site_token_context(
            history_tokens, static_features, site_features
        )
        future_tokens, future_patch_mask = self.future_patch_tokenizer(
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
                mask=history_patch_mask,
                time_attn_bias=None,
                time_rope_positions=None,
            )
        pv_states = history_tokens[:, 0]
        future_count = future_tokens.shape[2]
        future_states = self._project_history_patches_to_future(pv_states, future_count)
        future_valid = (
            future_patch_mask.permute(0, 2, 1).any(dim=-1).to(future_tokens.dtype)
            if future_patch_mask is not None
            else None
        )
        for layer in self.future_joint_decoder_layers:
            future_states, future_tokens = layer(
                future_states,
                future_tokens,
                pv_mask=future_valid,
                nwp_mask=future_patch_mask,
                time_attn_bias=None,
                time_rope_positions=None,
            )
        future_states = self.future_patch_refine(
            future_states, mask=future_valid, attn_bias=None, rope_positions=None
        )
        quantile_output = self._decode_quantiles_to_sequence(
            future_states, future_covariates.shape[1]
        )
        quantile_output = self._parameterize_quantile_sequence(quantile_output)
        if future_covariates_mask is not None:
            quantile_output = quantile_output * (
                future_covariates_mask.sum(dim=-1) > 0
            ).to(quantile_output.dtype).unsqueeze(-1)
        median_index = 5
        return (
            quantile_output[..., median_index : median_index + 1],
            {"quantiles": quantile_output, "quantile_levels": QUANTILES},
        )
