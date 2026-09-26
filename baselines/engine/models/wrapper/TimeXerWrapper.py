"""Adapter for the unmodified TimeXer benchmark model."""

import torch.nn as nn

from ._dag_external_common import base_config, future_covariates, load_external, reorder_target_first


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        cfg = base_config(configs, use_future=1)
        timexer = load_external("ts_benchmark.baselines.timexer.model.timexer_model", configs)
        self.core = timexer.timexer_model(cfg)
        self.pred_len = cfg.pred_len
        # Target-only forecast; keep q9 head from treating enc_in covariates as output width.
        self.output_dim = 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_mark_dec, mask
        output = self.core(reorder_target_first(x_enc), future_covariates(x_dec, self.pred_len))
        return output[..., :1]
