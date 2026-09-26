"""Native-q9 DAG adapter; upstream ``ts_benchmark`` sources are untouched."""

import torch
import torch.nn as nn

from models.native_q9.common import NativeQuantileMixin
from models.wrapper.DAGWrapper import Model as _DAGWrapper


class Model(NativeQuantileMixin, _DAGWrapper):
    """DAG with the original target heads replaced by native q9 heads."""

    def __init__(self, configs):
        _DAGWrapper.__init__(self, configs)
        self._init_native_quantiles(configs)
        if not self.num_quantiles:
            raise ValueError("DAG native-q9 requires --quantiles")
        old_x = self.core.temporal_encoder.x_head.linear
        object.__setattr__(self, "original_point_head_temporal", old_x)
        self.core.temporal_encoder.x_head.linear = nn.Linear(
            old_x.in_features, self.pred_len * self.num_quantiles
        )
        old_c = self.core.cov_encoder.future_projection.sequence_proj
        object.__setattr__(self, "original_point_head_covariate", old_c)
        self.core.cov_encoder.future_projection.sequence_proj = nn.Linear(
            old_c.in_features, self.pred_len * self.num_quantiles
        )
        self.native_output_len = self.pred_len * self.num_quantiles
        self.supports_native_quantiles = True

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_mark_dec, mask
        output, aux = self.core(
            self.core_input_reorder(x_enc), self.core_future_covariates(x_dec)
        )
        self.last_aux_loss = aux if self.training else None
        expected = self.native_output_len
        if output.ndim != 3 or output.shape[1] != expected or output.shape[2] != 1:
            raise ValueError(
                "DAG native-q9 core output layout changed: expected "
                f"[B,{expected},1], got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise ValueError("DAG native-q9 core produced non-finite values")
        values = output.reshape(output.shape[0], self.pred_len, self.num_quantiles)
        return self._native_quantile_output(values)

    @staticmethod
    def core_input_reorder(x):
        # The external core expects the endogenous PV channel first.
        return x[..., [-1] + list(range(x.shape[-1] - 1))]

    def core_future_covariates(self, x):
        return x[:, -self.pred_len:, :-1]
