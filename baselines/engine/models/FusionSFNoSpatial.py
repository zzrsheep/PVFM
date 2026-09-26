"""FusionSF no-spatial adapter for PVFM.

This module freezes the non-spatial computation from the official FusionSF3M
implementation.  It deliberately removes only the satellite-context branch:
patch embedding, spatial encoder, spatial coordinates, cross-modal mixer, and
context VQ.  The remaining PV/NWP path keeps the original modules and order:

    ts_encoder(ts_embedding(PV + cyclic time))
      + guide_encoder(guide_embedding(future NWP))
      -> ts_enctodec -> temporal_transformer -> mean(mlp_heads)

The source provenance is recorded in ``fusionsf_nospatial_provenance.json``.
"""

from math import pi
from typing import Sequence

import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.native_q9.common import NativeQuantileMixin
from einops import rearrange
from einops.layers.torch import Rearrange


def _split_csv_arg(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


# Copied unchanged from FusionSF/src/models/modules/attention_modules.py.
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        del b, n
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# Copied unchanged from FusionSF/src/models/fusionSF_3modal.py.
class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        x += self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# Copied from FusionSF/src/models/modules/positional_encoding.py.  The input
# retains the official [B, T, C, H, W] contract; PVFM adds singleton H/W.
class Cyclical_embedding(nn.Module):
    def __init__(self, frequencies: Sequence[int]):
        super().__init__()
        self.frequencies = list(frequencies)
        self.dim = len(self.frequencies) * 2

    def forward(self, time_coords):
        embeddings = []
        for i, frequency in enumerate(self.frequencies):
            embeddings += [
                torch.sin(2 * torch.pi * time_coords[:, :, i] / frequency),
                torch.cos(2 * torch.pi * time_coords[:, :, i] / frequency),
            ]
        return torch.stack(embeddings, axis=2)


class Model(NativeQuantileMixin, nn.Module):
    """Official FusionSF PV/NWP path with the satellite branch removed."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        # The retained FusionSF point-head ensemble is averaged to one scalar
        # forecast before the shared PVFM probabilistic wrapper is attached.
        # Declare this explicitly so the generic q9 wrapper does not infer the
        # seven input channels (PV + six NWP channels) as the output width.
        self.output_dim = 1
        self.dim = int(getattr(configs, "fusionsf_nospatial_dim", getattr(configs, "fusionsf_dim", 64)))
        self.depth = int(getattr(configs, "fusionsf_nospatial_depth", getattr(configs, "fusionsf_depth", 4)))
        self.heads = int(getattr(configs, "fusionsf_nospatial_heads", getattr(configs, "fusionsf_heads", 4)))
        self.dim_head = int(getattr(configs, "fusionsf_nospatial_dim_head", getattr(configs, "fusionsf_dim_head", 64)))
        self.mlp_ratio = int(getattr(configs, "fusionsf_nospatial_mlp_ratio", getattr(configs, "fusionsf_ff_mult", 4)))
        self.dropout = float(getattr(configs, "fusionsf_nospatial_dropout", getattr(configs, "fusionsf_dropout", 0.3)))
        self.decoder_dim = int(getattr(configs, "fusionsf_nospatial_decoder_dim", getattr(configs, "fusionsf_decoder_dim", 128)))
        self.decoder_depth = int(getattr(configs, "fusionsf_nospatial_decoder_depth", getattr(configs, "fusionsf_decoder_depth", 4)))
        self.decoder_heads = int(getattr(configs, "fusionsf_nospatial_decoder_heads", getattr(configs, "fusionsf_decoder_heads", 6)))
        self.decoder_dim_head = int(getattr(configs, "fusionsf_nospatial_decoder_dim_head", getattr(configs, "fusionsf_decoder_dim_head", 128)))
        # The official release configs (configs/pl_module/fusionsf_{2,3}modal.yaml)
        # set num_mlp_heads: 9.  The FusionSF3M class signature defaults to 1, but
        # that value is always overridden by Hydra and is never actually trained.
        # 9 matters: the heads are parallel LayerNorm->Linear->ReLU branches whose
        # outputs are averaged, so a head whose bias goes negative dies (zero grad)
        # without killing the trunk.  With a single head that death is terminal --
        # the whole network's gradient becomes exactly zero and predictions collapse
        # to 0 permanently.  Do not lower this to 1.
        self.num_mlp_heads = int(getattr(configs, "fusionsf_nospatial_num_mlp_heads", getattr(configs, "fusionsf_num_mlp_heads", 9)))
        # Match the official release config; VQ remains an explicit ablation.
        self.vq_in_ts = bool(getattr(configs, "fusionsf_vq_in_ts", False))

        if self.pred_len <= 0:
            raise ValueError("FusionSFNoSpatial requires pred_len > 0")
        if getattr(configs, "features", "") != "MS":
            raise ValueError("FusionSFNoSpatial requires --features MS")
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("FusionSFNoSpatial dimensions and depth must be positive")
        if self.num_mlp_heads <= 0:
            raise ValueError("fusionsf_num_mlp_heads must be positive")

        self.future_cov_cols = _split_csv_arg(getattr(configs, "future_covariate_cols", ""))
        if not self.future_cov_cols:
            raise ValueError("FusionSFNoSpatial requires --future_covariate_cols")

        self.time_coords_encoder = Cyclical_embedding([12, 31, 24])
        self.ts_channels = 1 + self.time_coords_encoder.dim
        self.guide_channels = len(self.future_cov_cols)

        # These definitions and defaults match the retained FusionSF3M path.
        self.ts_embedding = nn.Linear(self.ts_channels, self.dim)
        self.guide_embedding = nn.Linear(self.guide_channels, self.dim)
        self.ts_encoder = Transformer(
            self.dim, self.pred_len, self.depth, self.heads, self.dim_head,
            self.dim * self.mlp_ratio, dropout=self.dropout,
        )
        self.guide_encoder = Transformer(
            self.dim, self.pred_len, self.depth, self.heads, self.dim_head,
            self.dim * self.mlp_ratio, dropout=self.dropout,
        )
        self.ts_enctodec = nn.Linear(self.dim, self.decoder_dim)
        self.temporal_transformer = Transformer(
            self.decoder_dim, self.pred_len, self.decoder_depth,
            self.decoder_heads, self.decoder_dim_head,
            self.decoder_dim * self.mlp_ratio, dropout=self.dropout,
        )
        self.ts_mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        self.mlp_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.decoder_dim),
                    nn.Linear(self.decoder_dim, 1, bias=True),
                    nn.ReLU(),
                )
                for _ in range(self.num_mlp_heads)
            ]
        )
        # This is present in the official model even though its forward path
        # does not consume it.  Keep it for structural fidelity.
        self.quantile_masker = nn.Sequential(
            nn.Conv1d(self.decoder_dim, self.dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Rearrange("b c t -> b t c"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_mlp_heads),
        )

        self.ts_vq = None
        if self.vq_in_ts:
            try:
                from vector_quantize_pytorch import ResidualVQ
            except ImportError as exc:
                raise ImportError(
                    "FusionSFNoSpatial with TS-VQ requires vector-quantize-pytorch. "
                    "Install the project requirements or pass --disable_fusionsf_vq_in_ts."
                ) from exc
            self.ts_vq = ResidualVQ(
                dim=self.dim,
                num_quantizers=8,
                codebook_size=1024,
                shared_codebook=True,
                kmeans_init=True,
            )
        self.last_aux_loss = None
        self._init_native_quantiles(configs)
        if self.num_quantiles:
            if self.num_quantiles != self.num_mlp_heads:
                raise ValueError(
                    "FusionSF native q9 requires num_mlp_heads equal to quantile count; "
                    f"got heads={self.num_mlp_heads}, Q={self.num_quantiles}."
                )
            self.supports_native_quantiles = True

    @staticmethod
    def _calendar_month_day_hour(x_mark_enc, pred_len):
        if x_mark_enc is None or x_mark_enc.shape[-1] < 4:
            raise ValueError(
                "FusionSFNoSpatial requires raw pv_multires calendar marks "
                "[month, day, weekday, hour]; use the default non-timeF embedding."
            )
        marks = x_mark_enc[:, -pred_len:, :]
        return torch.stack((marks[..., 0], marks[..., 1], marks[..., 3]), dim=-1)

    def _future_guide(self, x_dec):
        if x_dec is None or x_dec.shape[-1] <= 1:
            raise ValueError("FusionSFNoSpatial could not find future NWP in x_dec")
        guide = x_dec[:, -self.pred_len:, :-1]
        if guide.shape[-1] != self.guide_channels:
            raise ValueError(
                f"FusionSFNoSpatial expected {self.guide_channels} future NWP channels, "
                f"got {guide.shape[-1]}"
            )
        return guide

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_mark_dec, mask
        if x_enc.shape[1] != self.pred_len:
            raise ValueError(
                "FusionSFNoSpatial requires native equal context: "
                f"seq_len={x_enc.shape[1]}, pred_len={self.pred_len}."
            )

        calendar = self._calendar_month_day_hour(x_mark_enc, self.pred_len)
        time_coords = calendar.unsqueeze(-1).unsqueeze(-1)
        time_embedding = self.time_coords_encoder(time_coords).squeeze(-1).squeeze(-1)
        ts = torch.cat((x_enc[:, :, -1:], time_embedding), dim=-1)
        guide = self._future_guide(x_dec)

        ts = self.ts_embedding(ts)
        if self.ts_vq is not None:
            ts, _, commit_loss = self.ts_vq(ts)
            self.last_aux_loss = commit_loss.mean()
        else:
            self.last_aux_loss = ts.new_zeros(())

        latent_ts = self.ts_encoder(ts)
        latent_guide = self.guide_encoder(self.guide_embedding(guide))
        latent_ts = self.ts_enctodec(latent_ts + latent_guide)
        y = self.temporal_transformer(latent_ts)
        outputs = torch.stack([head(y) for head in self.mlp_heads], dim=2)
        if self.num_quantiles:
            return self._native_quantile_output(outputs.squeeze(-1))
        return outputs.mean(dim=2)
