"""Minimal PVFM factorized Transformer blocks."""

import torch
from torch import nn
from .layers import FeedForward, RotaryMultiheadAttention


class ResidualSelfAttentionOnlyBlock(nn.Module):
    """Pre-norm self-attention residual sublayer without an internal FFN."""

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_rope=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = RotaryMultiheadAttention(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )

    def forward(self, x, mask=None, attn_bias=None, rope_positions=None):
        x = x + self.attn(
            self.norm(x),
            query_mask=mask,
            key_mask=mask,
            attn_bias=attn_bias,
            rope_positions=rope_positions,
        )
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class ResidualCrossAttentionOnlyBlock(nn.Module):
    """Pre-norm cross-attention residual sublayer without an internal FFN."""

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_rope=True):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = RotaryMultiheadAttention(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )

    def forward(self, query, key_value, query_mask=None, key_mask=None):
        key_value = self.norm_kv(key_value)
        query = query + self.attn(
            self.norm_q(query),
            key=key_value,
            value=key_value,
            query_mask=query_mask,
            key_mask=key_mask,
        )
        if query_mask is not None:
            query = query * query_mask.unsqueeze(-1).to(query.dtype)
        return query


class EncoderBlock(nn.Module):
    """Two-stage patch block with one shared FFN after time and variable attention."""

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_rope=True):
        super().__init__()
        self.time_attn = ResidualSelfAttentionOnlyBlock(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )
        self.variable_attn = ResidualSelfAttentionOnlyBlock(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=False
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(self, x, mask=None, time_attn_bias=None, time_rope_positions=None):
        batch, num_vars, seg_num, hidden_dim = x.shape
        time_in = x.reshape(batch * num_vars, seg_num, hidden_dim)
        time_mask = None if mask is None else mask.reshape(batch * num_vars, seg_num)
        rope_positions = None
        if time_rope_positions is not None:
            rope_positions = (
                time_rope_positions.unsqueeze(1)
                .expand(batch, num_vars, seg_num)
                .reshape(batch * num_vars, seg_num)
            )
        time_bias = None
        if time_attn_bias is not None:
            if time_attn_bias.dim() == 3:
                time_bias = (
                    time_attn_bias.unsqueeze(1)
                    .expand(batch, num_vars, seg_num, seg_num)
                    .reshape(batch * num_vars, seg_num, seg_num)
                )
            elif time_attn_bias.dim() == 4:
                heads = time_attn_bias.shape[1]
                time_bias = (
                    time_attn_bias.unsqueeze(1)
                    .expand(batch, num_vars, heads, seg_num, seg_num)
                    .reshape(batch * num_vars, heads, seg_num, seg_num)
                )
            else:
                raise ValueError(
                    f"time_attn_bias must have shape [B, P, P] or [B, H, P, P], got {tuple(time_attn_bias.shape)}"
                )
        time_out = self.time_attn(
            time_in, mask=time_mask, attn_bias=time_bias, rope_positions=rope_positions
        ).reshape(batch, num_vars, seg_num, hidden_dim)
        variable_in = time_out.permute(0, 2, 1, 3).reshape(
            batch * seg_num, num_vars, hidden_dim
        )
        variable_mask = (
            None
            if mask is None
            else mask.permute(0, 2, 1).reshape(batch * seg_num, num_vars)
        )
        variable_out = self.variable_attn(variable_in, mask=variable_mask)
        out = variable_out.reshape(batch, seg_num, num_vars, hidden_dim).permute(
            0, 2, 1, 3
        )
        out = out + self.ffn(self.ffn_norm(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class DecoderBlock(nn.Module):
    """Future/history PV<-weather block with time attention, cross attention, then one FFN."""

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_rope=True):
        super().__init__()
        self.pv_time_attn = ResidualSelfAttentionOnlyBlock(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )
        self.nwp_time_attn = ResidualSelfAttentionOnlyBlock(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )
        self.pv_from_nwp = ResidualCrossAttentionOnlyBlock(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=False
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def _apply_pv_ffn(self, pv_state, pv_norm, pv_mask):
        return pv_state + self.ffn(pv_norm)

    def forward(
        self,
        pv_state,
        nwp_tokens,
        pv_mask=None,
        nwp_mask=None,
        time_attn_bias=None,
        time_rope_positions=None,
    ):
        batch, patch_count, hidden_dim = pv_state.shape
        pv_state = self.pv_time_attn(
            pv_state,
            mask=pv_mask,
            attn_bias=time_attn_bias,
            rope_positions=time_rope_positions,
        )
        if nwp_tokens is None or nwp_tokens.shape[1] <= 0:
            pv_norm = self.ffn_norm(pv_state)
            pv_state = self._apply_pv_ffn(pv_state, pv_norm, pv_mask=pv_mask)
            if pv_mask is not None:
                pv_state = pv_state * pv_mask.unsqueeze(-1).to(pv_state.dtype)
            return (pv_state, nwp_tokens)
        _, var_count, nwp_patch_count, _ = nwp_tokens.shape
        nwp_time = nwp_tokens.reshape(batch * var_count, nwp_patch_count, hidden_dim)
        nwp_time_mask = (
            None
            if nwp_mask is None
            else nwp_mask.reshape(batch * var_count, nwp_patch_count)
        )
        nwp_time_bias = None
        if time_attn_bias is not None:
            if time_attn_bias.dim() == 3:
                nwp_time_bias = (
                    time_attn_bias.unsqueeze(1)
                    .expand(batch, var_count, nwp_patch_count, nwp_patch_count)
                    .reshape(batch * var_count, nwp_patch_count, nwp_patch_count)
                )
            elif time_attn_bias.dim() == 4:
                heads = time_attn_bias.shape[1]
                nwp_time_bias = (
                    time_attn_bias.unsqueeze(1)
                    .expand(batch, var_count, heads, nwp_patch_count, nwp_patch_count)
                    .reshape(batch * var_count, heads, nwp_patch_count, nwp_patch_count)
                )
            else:
                raise ValueError(
                    f"time_attn_bias must have shape [B, P, P] or [B, H, P, P], got {tuple(time_attn_bias.shape)}"
                )
        nwp_rope_positions = None
        if time_rope_positions is not None:
            nwp_rope_positions = (
                time_rope_positions.unsqueeze(1)
                .expand(batch, var_count, nwp_patch_count)
                .reshape(batch * var_count, nwp_patch_count)
            )
        nwp_tokens = self.nwp_time_attn(
            nwp_time,
            mask=nwp_time_mask,
            attn_bias=nwp_time_bias,
            rope_positions=nwp_rope_positions,
        ).reshape(batch, var_count, nwp_patch_count, hidden_dim)
        pv_query = pv_state.reshape(batch * patch_count, 1, hidden_dim)
        memory = nwp_tokens.permute(0, 2, 1, 3).reshape(
            batch * patch_count, var_count, hidden_dim
        )
        pv_query_mask = (
            pv_mask.reshape(batch * patch_count, 1) if pv_mask is not None else None
        )
        memory_mask = (
            nwp_mask.permute(0, 2, 1).reshape(batch * patch_count, var_count)
            if nwp_mask is not None
            else None
        )
        pv_state = self.pv_from_nwp(
            pv_query, memory, query_mask=pv_query_mask, key_mask=memory_mask
        ).reshape(batch, patch_count, hidden_dim)
        pv_norm = self.ffn_norm(pv_state)
        pv_state = self._apply_pv_ffn(pv_state, pv_norm, pv_mask=pv_mask)
        nwp_tokens = nwp_tokens + self.ffn(self.ffn_norm(nwp_tokens))
        if pv_mask is not None:
            pv_state = pv_state * pv_mask.unsqueeze(-1).to(pv_state.dtype)
        if nwp_mask is not None:
            nwp_tokens = nwp_tokens * nwp_mask.unsqueeze(-1).to(nwp_tokens.dtype)
        return (pv_state, nwp_tokens)
