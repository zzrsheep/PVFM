"""Adapter for the unmodified DAG-project time_series_library TiDE model."""

import torch.nn as nn

from ._dag_external_common import base_config, future_covariates, load_external, reorder_target_first


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # The original TiDE concatenates history and future exogenous values
        # along time, so the feature width is the (equal) covariate count.
        hist_dim = len([x for x in str(configs.history_covariate_cols).split(",") if x.strip()])
        cfg = base_config(configs, covariate_dim=hist_dim)
        tide = load_external("ts_benchmark.baselines.time_series_library.models.TiDE", configs)
        self.core = tide.TiDE(cfg)
        self.pred_len = cfg.pred_len
        # Target-only forecast; keep q9 head from treating enc_in covariates as output width.
        self.output_dim = 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del mask, x_mark_enc, x_mark_dec
        history = reorder_target_first(x_enc)
        future_exog = future_covariates(x_dec, self.pred_len)
        # The original future-exog branch infers series_dim as
        # x_enc_width - x_dec_width, so x_dec must contain exogenous columns only.
        # TiDE's forward performs an initial concatenation even though the
        # future-exogenous branch replaces it immediately afterwards.
        dummy_hist = history.new_zeros((history.shape[0], history.shape[1], self.core.feature_dim))
        dummy_future = history.new_zeros((history.shape[0], self.pred_len, self.core.feature_dim))
        output = self.core(history, dummy_hist, future_exog, dummy_future)
        return output[..., :1]
