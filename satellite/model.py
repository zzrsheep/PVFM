"""Frozen PVFM + historical satellite residual + frozen future decoder."""

import torch
from torch import nn
from .backbone import SatelliteBackbone
from .encoder import SatellitePatchEncoder
from .fusion import PatchAlignedSatelliteFusion, SatelliteAdapter
from .anchors import extract_history_anchors
from .alignment import validate_historical_satellite_times

BACKBONE_INPUTS = (
    "past_target",
    "past_observed_mask",
    "future_covariates",
    "future_covariates_mask",
    "past_time_features",
    "future_time_features",
    "static_features",
    "site_features",
)
SATELLITE_INPUTS = (
    "satellite_history",
    "satellite_frame_mask",
    "satellite_patch_coords",
    "satellite_frame_time_ns",
    "forecast_origin_time_ns",
)


class SatelliteForecaster(nn.Module):
    """Only satellite parameters train; backbone dropout stays disabled."""

    def __init__(self, backbone_config, satellite_config):
        super().__init__()
        self.backbone = SatelliteBackbone(**backbone_config)
        config = dict(satellite_config)
        encoder = SatellitePatchEncoder(
            image_channels=config["image_channels"],
            image_patch_size=config["image_patch_size"],
            satellite_dim=config["satellite_dim"],
            output_dim=self.backbone.hidden_dim,
            spatial_layers=config["spatial_layers"],
            temporal_layers=config["temporal_layers"],
            n_heads=config["encoder_n_heads"],
            dropout=config["dropout"],
            temporal_patch_len=self.backbone.patch_len,
            temporal_patch_stride=self.backbone.patch_stride,
            spatial_rope_max_frequency=config["spatial_rope_max_frequency"],
            channel_mean=config["channel_mean"],
            channel_std=config["channel_std"],
        )
        fusion = PatchAlignedSatelliteFusion(
            hidden_dim=self.backbone.hidden_dim,
            n_heads=config["fusion_n_heads"],
            dropout=config["dropout"],
            gate_hidden_dim=config["gate_hidden_dim"],
            spatial_rope_max_frequency=config["spatial_rope_max_frequency"],
            initial_gate_bias=config["initial_gate_bias"],
        )
        self.satellite_adapter = SatelliteAdapter(encoder, fusion)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
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
        satellite_history,
        satellite_frame_mask,
        satellite_patch_coords,
        satellite_frame_time_ns,
        forecast_origin_time_ns
    ):
        if past_target.ndim != 3 or past_target.shape[1:] != (16, 1):
            raise ValueError("This released checkpoint uses 16 hourly history points.")
        if future_covariates.shape != (len(past_target), 4, 6):
            raise ValueError(
                "This released checkpoint uses four future hours of six NWP variables."
            )
        if satellite_history.shape != (len(past_target), 16, 4, 64, 64):
            raise ValueError("Expected historical satellite images [B,16,4,64,64].")
        if past_time_features.shape != (len(past_target), 16, 5):
            raise ValueError(
                "Expected five minute/hour/weekday/day/day-of-year fields."
            )
        validate_historical_satellite_times(
            satellite_frame_time_ns, satellite_frame_mask, forecast_origin_time_ns
        )
        self.backbone.eval()
        with torch.no_grad():
            states, context = self.backbone.encode_history(
                past_target=past_target,
                past_observed_mask=past_observed_mask,
                future_covariates=future_covariates,
                future_covariates_mask=future_covariates_mask,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                static_features=static_features,
                site_features=site_features,
            )
            anchors = extract_history_anchors(
                self.backbone,
                self.satellite_adapter.encoder,
                past_time_features=past_time_features,
                site_features=site_features,
                static_features=static_features,
            )
        satellite = self.satellite_adapter(
            states.detach(),
            satellite_history,
            satellite_frame_mask,
            satellite_patch_coords,
            solar_geometry=anchors.solar_geometry,
            patch_position_embedding=anchors.patch_position_embedding,
            solar_patch_embedding=anchors.solar_patch_embedding,
            site_bias=anchors.site_bias,
            pv_patch_mask=context.pv_mask,
        )
        # Frozen parameters, but differentiable with respect to adapter output.
        return self.backbone.decode_from_pv_states(
            satellite.fusion.fused_pv_states, context
        )
