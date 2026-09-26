"""Native-q9 GCGNet adapter; upstream ``ts_benchmark`` sources are untouched."""

import torch
import torch.nn as nn

from models.native_q9.common import NativeQuantileMixin
from models.wrapper.GCGNetWrapper import Model as _GCGNetWrapper


class Model(NativeQuantileMixin, _GCGNetWrapper):
    def __init__(self, configs):
        _GCGNetWrapper.__init__(self, configs)
        self._init_native_quantiles(configs)
        if not self.num_quantiles:
            raise ValueError("GCGNet native-q9 requires --quantiles")
        old = self.core.head.mlp[-1]
        object.__setattr__(self, "original_point_head", old)
        self.core.head.mlp[-1] = nn.Linear(old.in_features, self.pred_len * self.num_quantiles)
        self.native_output_len = self.pred_len * self.num_quantiles
        # The untouched upstream GCGNet VAE computes exp(0.5 * logvar)
        # internally.  A few PV windows produce extreme graph-VAE logvars
        # (hundreds in magnitude), which overflow before the returned
        # auxiliary loss can be inspected.  Clamp only the VAE logvar output
        # at this adapter boundary; this does not alter the upstream source,
        # architecture, parameter shapes, or forecast head semantics.
        self.native_logvar_clip = 5.0
        self._native_logvar_hooks = []
        for vae in (self.core.vae, self.core.graph_discriminator.graph_vae.vae):
            self._native_logvar_hooks.append(
                vae.fc_logvar.register_forward_hook(self._clip_native_logvar)
            )
        self.supports_native_quantiles = True

    def _clip_native_logvar(self, _module, _inputs, output):
        return output.clamp(-self.native_logvar_clip, self.native_logvar_clip)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, *, target_future=None):
        del x_mark_enc, x_mark_dec, mask
        if target_future is None:
            raise ValueError("GCGNet native-q9 requires target_future during training/evaluation")
        output, aux = self.core(
            x_enc[..., [-1] + list(range(x_enc.shape[-1] - 1))],
            x_dec[:, -self.pred_len:, :-1],
            target_future[:, -self.pred_len:, -1:],
        )
        self.last_aux_loss = aux if self.training else None
        expected = self.native_output_len
        if output.ndim != 3 or output.shape[1] != expected or output.shape[2] != 1:
            raise ValueError(
                "GCGNet native-q9 core output layout changed: expected "
                f"[B,{expected},1], got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise ValueError("GCGNet native-q9 core produced non-finite values")
        values = output.reshape(output.shape[0], self.pred_len, self.num_quantiles)
        return self._native_quantile_output(values)
