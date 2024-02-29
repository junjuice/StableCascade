import math

import torch
import torch.nn.functional as F
from torch import nn

from zeta.nn.attention.base import BaseAttention
from zeta.nn.embeddings.multiway_network import MultiwayNetwork
from zeta.nn.embeddings.xpos_relative_position import XPOS
import bitnet

import torch
from einops import rearrange
from torch import nn
import bitnet

class MultiheadAttention(BaseAttention):
    def __init__(
        self,
        embed_dim: int = None,
        num_heads: int = None,
        dropout: int = 0.0,
        self_attention: bool = False,
        subln: bool = False,
        layernorm_eps=1e-05,
        xpos_scale_base: int = 512,
        xpos_rel_pos=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention

        self.k_proj = bitnet.BitLinear(embed_dim, embed_dim, bias=True)
        self.v_proj = bitnet.BitLinear(embed_dim, embed_dim, bias=True)
        self.q_proj = bitnet.BitLinear(embed_dim, embed_dim, bias=True)
        self.out_proj = bitnet.BitLinear(embed_dim, embed_dim, bias=True)
        self.inner_attn_ln = (
            MultiwayNetwork(nn.LayerNorm(self.embed_dim, eps=layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, xpos_scale_base)
            if xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert (
            embed_dim == self.embed_dim
        ), f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = (
            attn.transpose(0, 1)
            .reshape(tgt_len, bsz, embed_dim)
            .transpose(0, 1)
        )

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        return attn

class MultiModalCrossAttention(nn.Module):
    """
    Enhanced CrossAttention module with conditional layer normalization, lambda masking, and dropout.


    Args:
        dim: Dimension of the model.
        heads: Number of attention heads.
        context_dim: Dimension of the context.
        dim_head: Dimension of each attention head.
        dropout: Dropout rate.
        qk: Whether to use conditional layer normalization.
        post_attn_norm: Whether to use post-attention

    Examples:
        import torch
        import torch.nn as nn
        from zeta.nn.attention.cross_attn_images import CrossAttention
        x = torch.randn(1, 32, 1024)
        context = torch.randn(1, 32, 1024)
        attn = CrossAttention(1024, 8, 1024)
        out = attn(x, context)
        out.shape
        torch.Size([1, 32, 1024])
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        context_dim: int,
        dim_head=64,
        dropout=0.1,
        qk: bool = False,
        post_attn_norm: bool = False,
        attention_strategy: str = None,  # "average",
        mask=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.qk = qk
        self.post_attn_norm = post_attn_norm
        self.attention_strategy = attention_strategy
        self.mask = mask
        self.context_dim = context_dim

        # Linear layers for q, k, v
        self.to_q = bitnet.BitLinear(dim, dim_head * heads, bias=False)
        self.to_k = bitnet.BitLinear(dim, dim_head * heads, bias=False)
        self.to_v = bitnet.BitLinear(dim, dim_head * heads, bias=False)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            bitnet.BitLinear(dim_head * heads, dim), nn.Dropout(dropout)
        )

    def forward(self, x, context):
        # Compute query, key, value
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Optional conditional layer normalization
        if self.qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Reshape for multi-head attention
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        # Scaled dot-product attention
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Optional masking
        if self.mask is not None:
            dots.masked_fill_(~self.mask, float("-inf"))

        # Softmax and dropout on attention weights
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Compute output
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Average or concatenate heads based on strategy
        if self.attention_strategy == "average":
            out = out.mean(dim=1)

        # Post-attention normalization
        if self.post_attn_norm:
            out = self.norm_post_attn(out)

        # Output projection
        return self.to_out(out)