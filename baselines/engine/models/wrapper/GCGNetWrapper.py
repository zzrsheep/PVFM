"""Adapter for the unmodified GCGNet benchmark model."""

import torch
import torch.nn as nn

from ._dag_external_common import base_config, future_covariates, load_external, reorder_target_first


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        cfg = base_config(
            configs,
            rank=int(getattr(configs, "gcgnet_rank", 4)),
            use_norm=bool(int(getattr(configs, "use_norm", 1))),
        )
        gcg_model = load_external("ts_benchmark.baselines.GCGNet.models.gcgnet_model", configs)
        self.core = gcg_model.GCGNetModel(cfg)
        self.pred_len = cfg.pred_len
        # Target-only forecast; keep q9 head from treating enc_in covariates as output width.
        self.output_dim = 1
        self.last_aux_loss = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, *, target_future=None):
        del x_mark_enc, x_mark_dec, mask
        if target_future is None:
            raise ValueError("GCGNetWrapper requires target_future during train/validation/test")
        history = reorder_target_first(x_enc)
        exog = future_covariates(x_dec, self.pred_len)
        endo_future = target_future[:, -self.pred_len:, -1:].to(dtype=history.dtype, device=history.device)
        output, aux = self.core(history, exog, endo_future)
        self.last_aux_loss = aux if self.training else None
        return output[..., :1]
