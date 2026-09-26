import math
import torch
from torch import nn
import torch.nn.functional as F


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Minimal RoPE helper for temporal attention."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype, positions=None):
        if positions is None:
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        else:
            positions = positions.to(device=device, dtype=self.inv_freq.dtype)
        freqs = positions.unsqueeze(-1) * self.inv_freq.to(device=device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype))


def _broadcast_rotary_phase(phase):
    if phase.dim() == 2:
        return phase.unsqueeze(0).unsqueeze(0)
    if phase.dim() == 3:
        return phase.unsqueeze(1)
    raise ValueError(
        f"RoPE positions must produce 2D or 3D phase tensors, got shape={tuple(phase.shape)}"
    )


def apply_rotary(q, k, rotary_emb, q_positions=None, k_positions=None):
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    q_cos, q_sin = rotary_emb(q_len, q.device, q.dtype, positions=q_positions)
    k_cos, k_sin = rotary_emb(k_len, k.device, k.dtype, positions=k_positions)
    q_cos = _broadcast_rotary_phase(q_cos)
    q_sin = _broadcast_rotary_phase(q_sin)
    k_cos = _broadcast_rotary_phase(k_cos)
    k_sin = _broadcast_rotary_phase(k_sin)
    q = q * q_cos + rotate_half(q) * q_sin
    k = k * k_cos + rotate_half(k) * k_sin
    return (q, k)


class FeedForward(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1, d_ff=None):
        super().__init__()
        inner_dim = int(d_ff or 4 * hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RotaryMultiheadAttention(nn.Module):

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_rope=True):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
            )
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.use_rope = bool(use_rope)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = float(dropout)
        self.rotary = RotaryEmbedding(self.head_dim) if self.use_rope else None

    def _reshape_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_mask=None,
        key_mask=None,
        attn_bias=None,
        rope_positions=None,
        key_rope_positions=None,
    ):
        self_attention = key is None
        if key is None:
            key = query
        if value is None:
            value = key
        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key))
        v = self._reshape_heads(self.v_proj(value))
        if self.rotary is not None:
            if key_rope_positions is None and self_attention:
                key_rope_positions = rope_positions
            q, k = apply_rotary(
                q,
                k,
                self.rotary,
                q_positions=rope_positions,
                k_positions=key_rope_positions,
            )
        attn_mask = None
        if key_mask is not None or attn_bias is not None:
            batch = query.shape[0]
            q_len = query.shape[1]
            k_len = key.shape[1]
            attn_mask = torch.zeros(
                batch, 1, q_len, k_len, device=query.device, dtype=q.dtype
            )
            if attn_bias is not None:
                bias = attn_bias.to(device=query.device, dtype=q.dtype)
                if bias.dim() == 3:
                    bias = bias.unsqueeze(1)
                elif bias.dim() != 4:
                    raise ValueError(
                        f"attn_bias must have shape [B, Q, K] or [B, H, Q, K], got {tuple(bias.shape)}"
                    )
                attn_mask = attn_mask + bias
            if key_mask is not None:
                key_mask = key_mask.to(dtype=torch.bool, device=query.device)
                attn_mask = attn_mask.masked_fill(
                    (~key_mask).unsqueeze(1).unsqueeze(2), float("-inf")
                )
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(query.shape[0], query.shape[1], self.hidden_dim)
        )
        out = self.out_proj(out)
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).to(out.dtype)
        return out


class ResidualAttentionBlock(nn.Module):

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1, use_rope=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RotaryMultiheadAttention(
            hidden_dim, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(self, x, mask=None, attn_bias=None, rope_positions=None):
        x = x + self.attn(
            self.norm1(x),
            query_mask=mask,
            key_mask=mask,
            attn_bias=attn_bias,
            rope_positions=rope_positions,
        )
        x = x + self.ffn(self.norm2(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class HistoryPatchTokenizer(nn.Module):
    """Variable-wise patch tokens with solar and learned absolute-position embeddings."""

    def __init__(
        self,
        patch_len,
        patch_stride,
        hidden_dim,
        total_vars,
        fixed_seq_len=672,
        dropout=0.1,
    ):
        super().__init__()
        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("Patch length and stride must be positive")
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.hidden_dim = int(hidden_dim)
        self.total_vars = int(total_vars)
        self.fixed_seq_len = int(fixed_seq_len)
        self.value_proj = nn.Linear(self.patch_len, hidden_dim)
        self.var_embedding = nn.Embedding(self.total_vars, hidden_dim)
        with torch.random.fork_rng(devices=[]):
            self.solar_patch_time_proj = nn.Sequential(
                nn.Linear(self.patch_len * 4, max(hidden_dim, self.patch_len * 4, 16)),
                nn.GELU(),
                nn.Linear(max(hidden_dim, self.patch_len * 4, 16), hidden_dim),
            )
        nn.init.zeros_(self.solar_patch_time_proj[-1].weight)
        nn.init.zeros_(self.solar_patch_time_proj[-1].bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1, self._num_segments(self.fixed_seq_len), hidden_dim)
        )

    def _num_segments(self, seq_len):
        if seq_len <= self.patch_len:
            return 1
        return int(math.ceil((seq_len - self.patch_len) / self.patch_stride) + 1)

    def _patchify_channel_first(self, x):
        seq_len = int(x.shape[-1])
        seg_num = self._num_segments(seq_len)
        padded_len = max(
            self.patch_len, (seg_num - 1) * self.patch_stride + self.patch_len
        )
        pad_len = max(0, padded_len - seq_len)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return patches

    def forward(self, values, value_mask=None, solar_time_features=None):
        values_cf = values.permute(0, 2, 1)
        value_patches = self._patchify_channel_first(values_cf)
        seg_num = value_patches.shape[2]
        tokens = self.value_proj(value_patches)
        if self.var_embedding is not None:
            var_ids = torch.arange(self.total_vars, device=values.device)
            tokens = tokens + self.var_embedding(var_ids).view(
                1, self.total_vars, 1, self.hidden_dim
            )
        tokens = tokens + self.pos_embedding[:, :, :seg_num, :]
        if self.solar_patch_time_proj is not None and solar_time_features is not None:
            solar = solar_time_features.to(device=values.device, dtype=values.dtype)
            if solar.shape[-1] < 4:
                solar = F.pad(solar, (0, 4 - solar.shape[-1]))
            solar = solar[..., :4]
            if solar.shape[1] != values.shape[1]:
                solar = F.interpolate(
                    solar.transpose(1, 2),
                    size=values.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            solar_patches = self._patchify_channel_first(solar.permute(0, 2, 1))
            solar_flat = solar_patches.permute(0, 2, 3, 1).reshape(
                values.shape[0], seg_num, self.patch_len * 4
            )
            solar_bias = self.solar_patch_time_proj(solar_flat).to(dtype=tokens.dtype)
            tokens = tokens + 1.0 * solar_bias.unsqueeze(1)
        token_mask = None
        if value_mask is not None:
            mask_cf = value_mask.permute(0, 2, 1).to(values.dtype)
            mask_patches = self._patchify_channel_first(mask_cf)
            token_mask = (mask_patches.sum(dim=-1) > 0).to(values.dtype)
            tokens = tokens * token_mask.unsqueeze(-1)
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        if token_mask is not None:
            tokens = tokens * token_mask.unsqueeze(-1)
        return (tokens, token_mask)
