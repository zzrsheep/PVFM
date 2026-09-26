"""Native-q9 TiDE adapter.

The upstream TiDE implementation is loaded without modification.  The
native core subclasses it and changes only the terminal forecast projections
needed to represent nine quantiles.  Its inherited ``forward`` continues to
build the original history-covariate + future-covariate feature sequence.
"""

from __future__ import annotations

import torch
from torch import nn

from models.native_q9.common import NativeQuantileMixin
from models.wrapper._dag_external_common import (
    base_config,
    covariate_dims,
    future_covariates,
    load_external,
    reorder_target_first,
)


def _native_tide_class(upstream_tide, num_quantiles: int):
    """Create a subclass of the untouched upstream TiDE class."""

    class NativeTiDE(upstream_tide):
        def __init__(self, configs):
            super().__init__(configs)
            self.native_num_quantiles = int(num_quantiles)

            old_fc2 = self.temporalDecoder.fc2
            old_fc3 = self.temporalDecoder.fc3
            object.__setattr__(self, "original_point_head_temporal_fc2", old_fc2)
            object.__setattr__(self, "original_point_head_temporal_fc3", old_fc3)
            self.temporalDecoder.fc2 = nn.Linear(
                old_fc2.in_features,
                self.native_num_quantiles,
                bias=old_fc2.bias is not None,
            )
            self.temporalDecoder.fc3 = nn.Linear(
                old_fc3.in_features,
                self.native_num_quantiles,
                bias=old_fc3.bias is not None,
            )

            old_residual = self.residual_proj
            object.__setattr__(self, "original_point_head_residual", old_residual)
            self.residual_proj = nn.Linear(
                old_residual.in_features,
                self.pred_len * self.native_num_quantiles,
                bias=old_residual.bias is not None,
            )

        def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
            """Original scalar TiDE forecast with only the scalar-to-q9 changes.

            The upstream future-exogenous branch calls ``forecast`` once per
            endogenous series, therefore ``x_enc`` is ``[B,L]`` here while
            ``batch_y_mark`` is the real ``[B,L+H,C_cov]`` history/future
            covariate sequence.
            """
            del x_mark_enc, x_dec

            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            centered = centered / stdev

            feature = self.feature_encoder(batch_y_mark)
            hidden = self.encoders(
                torch.cat([centered, feature.reshape(feature.shape[0], -1)], dim=-1)
            )
            decoded = self.decoders(hidden).reshape(
                hidden.shape[0], self.pred_len, self.decode_dim
            )
            decoded_q = self.temporalDecoder(
                torch.cat([feature[:, self.seq_len :], decoded], dim=-1)
            )
            residual_q = self.residual_proj(centered).reshape(
                centered.shape[0], self.pred_len, self.native_num_quantiles
            )
            values = decoded_q + residual_q

            scale = stdev[:, 0].reshape(-1, 1, 1)
            mean = means[:, 0].reshape(-1, 1, 1)
            return values * scale + mean

    NativeTiDE.__name__ = "NativeTiDE"
    return NativeTiDE


class Model(NativeQuantileMixin, nn.Module):
    """PVFM-facing TiDE adapter with an in-core native q9 head."""

    def __init__(self, configs):
        super().__init__()
        self._init_native_quantiles(configs)
        if not self.num_quantiles:
            raise ValueError("TiDE native-q9 requires --quantiles")

        hist_dim, _ = covariate_dims(configs)
        cfg = base_config(configs, covariate_dim=hist_dim)
        tide_module = load_external(
            "ts_benchmark.baselines.time_series_library.models.TiDE", configs
        )
        native_tide = _native_tide_class(tide_module.TiDE, self.num_quantiles)
        self.core = native_tide(cfg)
        # Keep the original terminal modules discoverable at the adapter
        # boundary without registering them as active trainable modules.
        object.__setattr__(
            self,
            "original_point_heads",
            (
                self.core.original_point_head_temporal_fc2,
                self.core.original_point_head_temporal_fc3,
                self.core.original_point_head_residual,
            ),
        )
        self.pred_len = cfg.pred_len
        self.output_dim = 1
        self.supports_native_quantiles = True
        self.head_mode = "native_quantile"

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_mark_dec, mask
        history = reorder_target_first(x_enc)
        future_exog = future_covariates(x_dec, self.pred_len)

        feature_dim = self.core.feature_dim
        dummy_hist = history.new_zeros(
            (history.shape[0], history.shape[1], feature_dim)
        )
        dummy_future = history.new_zeros(
            (history.shape[0], self.pred_len, feature_dim)
        )
        # The inherited upstream forward replaces the placeholder marks in
        # future-exog mode with history[:,:,series_dim:] and future_exog.
        output = self.core(history, dummy_hist, future_exog, dummy_future)

        if output.ndim != 4 or output.shape[-2] != self.num_quantiles:
            raise ValueError(
                "Unexpected native TiDE output layout; expected [B,H,Q,C], "
                f"got {tuple(output.shape)}"
            )
        values = output[..., 0]
        if values.shape[1] != self.pred_len or values.shape[-1] != self.num_quantiles:
            raise ValueError(
                "Unexpected native TiDE target layout; expected "
                f"[B,{self.pred_len},{self.num_quantiles}], got {tuple(values.shape)}"
            )
        return self._native_quantile_output(values)
