"""Adapter for the unmodified DAG benchmark model."""

import torch
import torch.nn as nn

from ._dag_external_common import base_config, future_covariates, load_external, reorder_target_first


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        cfg = base_config(
            configs,
            alpha=float(getattr(configs, "dag_alpha", 0.7)),
            beta=float(getattr(configs, "dag_beta", 0.1)),
            use_c=True,
            use_t=True,
            use_c_exog=True,
            use_t_exog=True,
            infer_use_future=True,
            criterion=nn.L1Loss(),
        )
        dag_model = load_external("ts_benchmark.baselines.dag.models.dag_model", configs)
        self.core = dag_model.DAGModel(cfg)
        self.pred_len = cfg.pred_len
        # Target-only forecast; keep q9 head from treating enc_in covariates as output width.
        self.output_dim = 1
        self.last_aux_loss = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_mark_dec, mask
        output, aux = self.core(reorder_target_first(x_enc), future_covariates(x_dec, self.pred_len))
        self.last_aux_loss = aux if self.training else None
        return output[..., :1]
