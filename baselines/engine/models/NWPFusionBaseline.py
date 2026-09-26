import copy
import importlib

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Wrapper baseline that injects future NWP into an existing forecasting model.

    Supported strategies:
    - none: use the underlying model directly
    - feature_concat: project future NWP over the history axis, then append as feature channels
    - time_concat: project future NWP over the future axis, then append as future-time tokens
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.nwp_mode = (getattr(configs, "nwp_mode", "none") or "none").strip().lower()
        self.base_model_name = (getattr(configs, "base_model_name", "") or "").strip()
        self._native_quantiles_requested = bool(getattr(configs, "quantiles", ()))

        if not self.base_model_name:
            raise ValueError("NWPFusionBaseline requires --base_model_name")
        if self.base_model_name == "NWPFusionBaseline":
            raise ValueError("base_model_name cannot be NWPFusionBaseline")

        future_cov_cols = [item.strip() for item in getattr(configs, "future_covariate_cols", "").split(",") if item.strip()]
        self.future_cov_dim = len(future_cov_cols)
        self.base_enc_in = int(configs.enc_in)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)

        base_configs = copy.deepcopy(configs)
        if self.nwp_mode in {"feature_concat", "time_concat"} and self.future_cov_dim > 0:
            base_configs.enc_in = self.base_enc_in + self.future_cov_dim
            base_configs.dec_in = base_configs.enc_in
            base_configs.c_out = base_configs.enc_in
            if self.nwp_mode == "time_concat":
                base_configs.seq_len = self.seq_len + self.pred_len
        self.base_model = self._build_base_model(base_configs)
        self.supports_native_quantiles = bool(
            self._native_quantiles_requested
            and getattr(self.base_model, "supports_native_quantiles", False)
        )

        # The lightweight models do not all expose the same output-width
        # attribute.  Record the width produced by the actual base model so a
        # shared probabilistic head can be attached without model-specific
        # feature-layout assumptions.
        base_output_dim = getattr(self.base_model, "output_dim", None)
        if base_output_dim is None:
            if self.base_model_name == "TimeMixer":
                base_output_dim = getattr(base_configs, "c_out", None)
            elif hasattr(self.base_model, "channels"):
                base_output_dim = self.base_model.channels
            elif hasattr(self.base_model, "enc_in"):
                base_output_dim = self.base_model.enc_in
            else:
                base_output_dim = getattr(base_configs, "enc_in", getattr(base_configs, "c_out", None))
        if base_output_dim is None:
            raise ValueError(f"Unable to infer output width for base model {self.base_model_name}")
        self.output_dim = 1 if self.supports_native_quantiles else int(base_output_dim)

        if self.nwp_mode == "feature_concat":
            self.projector = nn.Linear(self.pred_len, self.seq_len)
        elif self.nwp_mode == "time_concat":
            self.projector = nn.Linear(self.pred_len, self.pred_len)
        else:
            self.projector = None

    def _build_base_model(self, configs):
        if self._native_quantiles_requested:
            from models.native_q9.local import (
                CrossUnetQ9, CrossformerQ9, DLinearQ9, ITransformerQ9,
                LightTSQ9, PatchTSTQ9, TimeMixerQ9,
            )
            native = {
                "Cross_Unet": CrossUnetQ9,
                "Crossformer": CrossformerQ9,
                "DLinear": DLinearQ9,
                "iTransformer": ITransformerQ9,
                "LightTS": LightTSQ9,
                "PatchTST": PatchTSTQ9,
                "TimeMixer": TimeMixerQ9,
            }.get(self.base_model_name)
            if native is not None:
                return native(configs)
        module = importlib.import_module(f"models.{self.base_model_name}")
        if not hasattr(module, "Model"):
            raise ValueError(f"Base model module models.{self.base_model_name} has no Model class")
        return module.Model(configs)

    def _extract_future_covariates(self, x_dec):
        if self.future_cov_dim <= 0 or x_dec is None or x_dec.shape[-1] <= 1:
            return None
        future_slice = x_dec[:, -self.pred_len:, :]
        return future_slice[:, :, :-1]

    def _feature_concat(self, x_enc, x_dec):
        future_cov = self._extract_future_covariates(x_dec)
        if future_cov is None:
            return x_enc

        projected = self.projector(future_cov.transpose(1, 2)).transpose(1, 2)
        if x_enc.shape[-1] == 1:
            target = x_enc
            fused = torch.cat([projected, target], dim=-1)
        else:
            hist_non_target = x_enc[:, :, :-1]
            target = x_enc[:, :, -1:]
            fused = torch.cat([hist_non_target, projected, target], dim=-1)
        return fused

    def _time_concat(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        future_cov = self._extract_future_covariates(x_dec)
        if future_cov is None:
            return x_enc, x_mark_enc

        projected = self.projector(future_cov.transpose(1, 2)).transpose(1, 2)
        batch_size = x_enc.shape[0]
        hist_feature_dim = x_enc.shape[-1]

        if hist_feature_dim == 1:
            hist_aug = torch.cat([torch.zeros(batch_size, self.seq_len, self.future_cov_dim, device=x_enc.device, dtype=x_enc.dtype), x_enc], dim=-1)
            future_target = torch.zeros(batch_size, self.pred_len, 1, device=x_enc.device, dtype=x_enc.dtype)
            future_aug = torch.cat([projected, future_target], dim=-1)
        else:
            hist_non_target = x_enc[:, :, :-1]
            target = x_enc[:, :, -1:]
            hist_aug = torch.cat(
                [
                    hist_non_target,
                    torch.zeros(batch_size, self.seq_len, self.future_cov_dim, device=x_enc.device, dtype=x_enc.dtype),
                    target,
                ],
                dim=-1,
            )
            future_hist = torch.zeros(batch_size, self.pred_len, hist_non_target.shape[-1], device=x_enc.device, dtype=x_enc.dtype)
            future_target = torch.zeros(batch_size, self.pred_len, 1, device=x_enc.device, dtype=x_enc.dtype)
            future_aug = torch.cat([future_hist, projected, future_target], dim=-1)

        fused_x = torch.cat([hist_aug, future_aug], dim=1)
        fused_x_mark = x_mark_enc
        if x_mark_enc is not None and x_mark_dec is not None:
            fused_x_mark = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        return fused_x, fused_x_mark

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.nwp_mode == "none" or self.future_cov_dim <= 0:
            return self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)

        if self.nwp_mode == "feature_concat":
            fused_x = self._feature_concat(x_enc, x_dec)
            return self.base_model(fused_x.clone(), x_mark_enc, x_dec, x_mark_dec, mask=mask)

        if self.nwp_mode == "time_concat":
            fused_x, fused_x_mark = self._time_concat(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return self.base_model(fused_x.clone(), fused_x_mark, x_dec, x_mark_dec, mask=mask)

        raise ValueError(f"Unsupported nwp_mode={self.nwp_mode}")
