# -*- coding:utf-8 -*-
"""Transformer Encoder ported from Biosignal-Foundation-Model (uni2ts-derived,
Apache 2.0).

PhysioME-side modifications versus the BFM original:
  * ``d_cond=0`` is supported and selects plain ``RMSNorm`` for every layer
    norm (no AdaLN modulation). PhysioME's NeuroNet backbone has no
    auxiliary conditioning vector to inject, so this lets us reuse the same
    block stack without having to feed a dummy ``cond`` tensor each call.
  * When ``d_cond > 0`` the AdaRMSNorm path is unchanged — keeping us
    compatible with the BFM Phase-2 pipeline.

Note: the layer's ``forward`` accepts ``var_id``/``time_id``/``token_mask``/
``cond`` for compatibility, but PhysioME currently uses none of them. The
sequential time index used by RoPE is derived inside attention when
``time_id`` is None — exactly what we need for the variable-length encoder.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward, MoEFeedForward
from .norm import AdaRMSNorm, RMSNorm
from .position import AttentionBias, QueryKeyProjection


class TransformerEncoderLayer(nn.Module):
    """Single Pre/Post-norm Transformer block (Self-Attention + FFN).

    ``norm1``/``norm2`` may be either ``AdaRMSNorm`` (uses ``cond``) or any
    plain norm module. The wrapper auto-detects which to call.
    """

    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: nn.Module | None,
        norm2: nn.Module | None,
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)

    def _norm(self, n: nn.Module, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if isinstance(n, AdaRMSNorm):
            assert cond is not None, "AdaRMSNorm requires cond"
            return n(x, cond)
        return n(x)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        var_id: torch.Tensor | None = None,
        time_id: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self._sa_block(
                self._norm(self.norm1, x, cond), attn_mask, var_id=var_id, time_id=time_id
            )
            x = x + self.ffn(self._norm(self.norm2, x, cond), token_mask=token_mask)
        else:
            x = self._norm(
                self.norm1,
                x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id),
                cond,
            )
            x = self._norm(self.norm2, x + self.ffn(x, token_mask=token_mask), cond)
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        var_id: torch.Tensor | None = None,
        time_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            query_var_id=var_id, kv_var_id=var_id,
            query_time_id=time_id, kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Stacked Transformer Encoder with GQA + GLU FFN + (optional) RoPE.

    ``d_cond=0`` (default for PhysioME) → all layer norms are plain RMSNorm.
    ``d_cond>0`` → AdaRMSNorm modulation, requires ``cond`` at forward.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int | None = None,
        num_groups: int | None = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = RMSNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_moe: bool = False,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        time_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        var_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        time_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: int | None = None,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        d_cond: int = 0,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.d_cond = d_cond
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads  # default = MHA

        var_attn_bias = self.get_layer(
            d_model, num_heads, num_groups, var_attn_bias_layer, shared_var_attn_bias
        )
        time_attn_bias = self.get_layer(
            d_model, num_heads, num_groups, time_attn_bias_layer, shared_time_attn_bias
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        if not use_moe:
            get_ffn = partial(
                GatedLinearUnitFeedForward if use_glu else FeedForward,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )
        else:
            get_ffn = partial(
                MoEFeedForward,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )

        # Norm factory: AdaRMSNorm when d_cond>0, else plain RMSNorm.
        if d_cond and d_cond > 0:
            get_layer_norm = partial(AdaRMSNorm, d_model, d_cond=d_cond)
            final_norm: nn.Module = AdaRMSNorm(d_model, d_cond=d_cond)
        else:
            get_layer_norm = partial(norm_layer, d_model)
            final_norm = norm_layer(d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_layer_norm(),
                    norm2=get_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = final_norm

    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable | None,
        shared_layer: bool,
    ) -> Callable[[], nn.Module] | None:
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        var_id: torch.Tensor | None = None,
        time_id: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, attn_mask, var_id=var_id, time_id=time_id,
                token_mask=token_mask, cond=cond,
            )
        if isinstance(self.norm, AdaRMSNorm):
            assert cond is not None, "AdaRMSNorm final norm requires cond"
            return self.norm(x, cond)
        return self.norm(x)
