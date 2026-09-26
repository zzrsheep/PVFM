"""Native-q9 TimeXer adapter; upstream ``ts_benchmark`` sources are untouched."""

import torch
import torch.nn as nn

from models.native_q9.common import NativeQuantileMixin
from models.wrapper.TimeXerWrapper import Model as _TimeXerWrapper


class Model(NativeQuantileMixin, _TimeXerWrapper):
    def __init__(self, configs):
        _TimeXerWrapper.__init__(self, configs)
        self._init_native_quantiles(configs)
        if not self.num_quantiles:
            raise ValueError("TimeXer native-q9 requires --quantiles")
        old = self.core.head.linear
        object.__setattr__(self, "original_point_head", old)
        self.core.head.linear = nn.Linear(old.in_features, self.pred_len * self.num_quantiles)
        self.native_output_len = self.pred_len * self.num_quantiles
        self.supports_native_quantiles = True

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_mark_dec, mask
        # Call forecast directly so the upstream forward's ``[-pred_len:]``
        # point-forecast slice does not discard the widened q axis.
        output = self.core.forecast(
            x_enc[..., [-1] + list(range(x_enc.shape[-1] - 1))],
            x_dec[:, -self.pred_len:, :-1],
        )
        expected = self.native_output_len
        if output.ndim != 3 or output.shape[1] != expected or output.shape[2] != 1:
            raise ValueError(
                "TimeXer native-q9 core output layout changed: expected "
                f"[B,{expected},1], got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise ValueError("TimeXer native-q9 core produced non-finite values")
        values = output.reshape(output.shape[0], self.pred_len, self.num_quantiles)
        return self._native_quantile_output(values)
