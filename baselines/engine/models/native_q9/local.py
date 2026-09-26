"""Native-q9 variants of the local lightweight baselines.

The original modules are imported unchanged.  Each class replaces only the
last forecast projection and reshapes its output to PVFM's ``[B,H,Q]``
contract.  The old projection is retained as a non-registered
``original_point_head`` attribute for provenance, so it is not counted or
optimized in a q9 run.
"""

from __future__ import annotations

import torch
from torch import nn

from models.iTransformer import Model as _ITransformer
from models.PatchTST import Model as _PatchTST
from models.Crossformer import Model as _Crossformer
from models.Cross_Unet import Model as _CrossUnet
from models.DLinear import Model as _DLinear
from models.LightTS import Model as _LightTS
from models.TimeMixer import Model as _TimeMixer
from .common import NativeQuantileMixin


def _keep_original(obj, name, module):
    # Bypass nn.Module.__setattr__: this preserves the old head for code
    # inspection without registering its parameters in the q9 model.
    object.__setattr__(obj, name, module)


class _LocalQ9(NativeQuantileMixin):
    def _setup_q9(self, configs):
        self._init_native_quantiles(configs)
        if not self.num_quantiles:
            raise ValueError("Native q9 variants require --quantiles.")
        self.output_dim = 1

    def _finish(self, output):
        # Local models retain all channels internally; the PV target is the
        # final channel, matching the existing MS/NWP adapter convention.
        if output.ndim == 4:
            # Canonical internal layout is [B,H,C,Q].
            output = output[:, :, -1, :]
        elif output.ndim == 3 and output.shape[-1] != self.num_quantiles:
            b, length, width = output.shape
            expected = int(self._native_original_pred_len) * self.num_quantiles
            if length == expected:
                output = output.reshape(b, int(self._native_original_pred_len), self.num_quantiles, width)
                output = output[..., -1, :]
        return self._native_quantile_output(output)


class ITransformerQ9(_LocalQ9, _ITransformer):
    def __init__(self, configs):
        _ITransformer.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        old = self.projection
        _keep_original(self, "original_point_head", old)
        self.native_quantile_head = nn.Linear(configs.d_model, configs.pred_len * self.num_quantiles, bias=True)
        self.projection = self.native_quantile_head

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        del x_dec, x_mark_dec
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        centered = centered / stdev
        _, _, n_vars = centered.shape
        enc_out = self.enc_embedding(centered, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        raw = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, n_vars)
        raw = raw.permute(0, 1, 3, 2).contiguous()
        scale = stdev[:, 0, :].unsqueeze(1).unsqueeze(-1)
        mean = means[:, 0, :].unsqueeze(1).unsqueeze(-1)
        raw = raw * scale + mean
        return raw[:, :, -1, :]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del mask
        return self._finish(self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec))


class PatchTSTQ9(_LocalQ9, _PatchTST):
    def __init__(self, configs):
        _PatchTST.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        old = self.head
        _keep_original(self, "original_point_head", old)
        self.native_quantile_head = type(old)(configs.enc_in, self.head_nf, configs.pred_len * self.num_quantiles, head_dropout=configs.dropout)
        self.head = self.native_quantile_head

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        del x_dec, x_mark_dec
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # Match the original PatchTST normalization before patch embedding.
        centered = centered / stdev
        centered = centered.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(centered)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        raw = self.head(enc_out).permute(0, 2, 1)
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, n_vars)
        raw = raw.permute(0, 1, 3, 2).contiguous()
        scale = stdev[:, 0, :].unsqueeze(1).unsqueeze(-1)
        mean = means[:, 0, :].unsqueeze(1).unsqueeze(-1)
        return raw * scale + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del mask
        return self._finish(self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec))


class CrossformerQ9(_LocalQ9, _Crossformer):
    def __init__(self, configs):
        _Crossformer.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        originals = []
        for layer in self.decoder.decode_layers:
            originals.append(layer.linear_pred)
            layer.linear_pred = nn.Linear(configs.d_model, self.seg_len * self.num_quantiles)
        _keep_original(self, "original_point_heads", originals)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del mask
        raw = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # The original Crossformer keeps the LAST H positions after segment
        # padding.  Each temporal position now occupies Q consecutive rows.
        raw = raw[:, -self.native_output_len:, :]
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, raw.shape[-1])
        raw = raw.permute(0, 1, 3, 2).contiguous()
        return self._finish(raw)


class CrossUnetQ9(_LocalQ9, _CrossUnet):
    def __init__(self, configs):
        _CrossUnet.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        originals = []
        for layer in self.decoder.decode_layers:
            originals.append(layer.linear_pred)
            layer.linear_pred = nn.Linear(configs.d_model, self.seg_len * self.num_quantiles)
        _keep_original(self, "original_point_heads", originals)

    def forward(self, x_enc, x_mark_enc, w_enc, x_mark_dec, seq_w_nwp_hist, seq_x_hist, mask=None):
        del x_mark_enc, mask
        # Keep the project-specific direct-NWP interface unchanged.
        newcat = torch.cat([seq_w_nwp_hist, seq_x_hist, x_enc[..., -1:]], dim=-1)
        corr = self.compute_channel_correlation(newcat)
        x = torch.cat([w_enc, x_enc], dim=-1) if self.useweather else x_enc
        raw = self.forecast(x, x_mark_dec, corr)
        raw = raw[:, : self._native_original_pred_len * self.num_quantiles, :]
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, raw.shape[-1])
        raw = raw.permute(0, 1, 3, 2).contiguous()
        return self._finish(raw)


class DLinearQ9(_LocalQ9, _DLinear):
    def __init__(self, configs):
        _DLinear.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        # Keep the public model contract intact.  Only the private output
        # projection is widened for the native quantile representation.
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        if self.individual:
            old_s, old_t = self.Linear_Seasonal, self.Linear_Trend
            _keep_original(self, "original_point_head_seasonal", old_s)
            _keep_original(self, "original_point_head_trend", old_t)
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.native_output_len) for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList([nn.Linear(self.seq_len, self.native_output_len) for _ in range(self.channels)])
        else:
            _keep_original(self, "original_point_head_seasonal", self.Linear_Seasonal)
            _keep_original(self, "original_point_head_trend", self.Linear_Trend)
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.native_output_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.native_output_len)

    def encoder(self, x):
        """DLinear encoder with a private H*Q temporal width.

        This is the upstream encoder logic verbatim except for the output
        allocation in the individual-channel branch.  ``pred_len`` remains
        H so external runners and audit logs retain their normal meaning.
        """
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.native_output_len],
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.native_output_len],
                dtype=trend_init.dtype, device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        raw = self.encoder(x_enc)
        raw = raw[:, : self._native_original_pred_len * self.num_quantiles, :]
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, raw.shape[-1])
        raw = raw.permute(0, 1, 3, 2).contiguous()
        return self._finish(raw)


class LightTSQ9(_LocalQ9, _LightTS):
    def __init__(self, configs):
        _LightTS.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        old_output_proj = self.layer_3.output_proj
        _keep_original(self, "original_point_head_layer3", old_output_proj)
        _keep_original(self, "original_point_head_ar", self.ar)
        self.layer_3.output_proj = nn.Linear(
            old_output_proj.in_features,
            self.native_output_len,
            bias=old_output_proj.bias is not None,
        )
        self.ar = nn.Linear(self.seq_len, self.native_output_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        raw = self.encoder(x_enc)
        raw = raw[:, : self._native_original_pred_len * self.num_quantiles, :]
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, raw.shape[-1])
        raw = raw.permute(0, 1, 3, 2).contiguous()
        return self._finish(raw)


class TimeMixerQ9(_LocalQ9, _TimeMixer):
    def __init__(self, configs):
        _TimeMixer.__init__(self, configs)
        self._setup_q9(configs)
        self._native_original_pred_len = int(configs.pred_len)
        self.native_output_len = self._native_original_pred_len * self.num_quantiles
        originals = []
        for layer in self.predict_layers:
            originals.append(layer)
        _keep_original(self, "original_point_heads", originals)
        self.predict_layers = nn.ModuleList([
            nn.Linear(layer.in_features, self.native_output_len, bias=layer.bias is not None)
            for layer in originals
        ])
        if hasattr(self, "regression_layers"):
            old_reg = self.regression_layers
            _keep_original(self, "original_point_regression_heads", old_reg)
            self.regression_layers = nn.ModuleList([
                nn.Linear(layer.in_features, self.native_output_len, bias=layer.bias is not None)
                for layer in old_reg
            ])

    def future_multi_mixing(self, B, enc_out_list, x_list):
        """Upstream decoder with only its private temporal width widened."""
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.native_output_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del mask
        raw = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raw = raw[:, : self._native_original_pred_len * self.num_quantiles, :]
        raw = raw.reshape(raw.shape[0], self._native_original_pred_len, self.num_quantiles, raw.shape[-1])
        raw = raw.permute(0, 1, 3, 2).contiguous()
        return self._finish(raw)
