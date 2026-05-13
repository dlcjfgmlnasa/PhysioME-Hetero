# -*- coding:utf-8 -*-
"""Attention-map visualization helpers for BiosignalDINO.

DINOv3 / iBOT do NOT reconstruct the raw signal, so the old MAE-style
real-vs-predicted waveform plot is no longer available. The closest
useful viz is the CLS token's attention pattern on the last encoder
layer — the standard DINO interpretation plot ("which time positions
does the CLS look at to form its global summary?").

The attention path uses ``F.scaled_dot_product_attention`` (fused
kernel, no weights returned). To get weights without changing the
production forward, we briefly monkey-patch ``F.scaled_dot_product_attention``
in a ``torch.no_grad`` context, capture (query, key) from each call,
and recompute attention weights via a manual softmax(QK^T) for the
LAST captured pair (which is the last encoder layer's attention).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def last_layer_cls_attention(encoder, x: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``encoder(x)`` and return ``(out, cls_attn)``.

    ``cls_attn`` is ``[B, L]`` where ``L = 1 + n_storage_tokens + n_patches``,
    averaged over attention heads. Index 0 is the CLS self-attention slot,
    indices ``1 .. n_storage`` are the storage / register tokens, and the
    tail are per-patch attentions in temporal order.
    """
    captured = {'q': None, 'k': None, 'scale': None}
    orig_sdpa = F.scaled_dot_product_attention

    def spy_sdpa(query, key, value, *args, **kwargs):
        captured['q'] = query.detach()
        captured['k'] = key.detach()
        captured['scale'] = kwargs.get('scale', None) or (
            1.0 / math.sqrt(query.shape[-1])
        )
        return orig_sdpa(query, key, value, *args, **kwargs)

    # Monkey-patch at the module-level so attention.py's ``F.scaled_dot_product_attention``
    # call picks up the spy (it resolves the attribute at call time).
    F.scaled_dot_product_attention = spy_sdpa
    try:
        out = encoder(x)
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    q, k = captured['q'], captured['k']
    if q is None or k is None:
        raise RuntimeError(
            'Encoder forward did not invoke F.scaled_dot_product_attention; '
            'attention viz cannot be computed.'
        )

    # GQA shape: [..., group, hpg, seq_len, head_dim]. The leading dims are
    # the batch. Compute attn = softmax(QK^T / sqrt(d_k)).
    attn = (q @ k.transpose(-2, -1)) * captured['scale']
    attn = F.softmax(attn, dim=-1).float()
    # Reduce attention-head axes (group, hpg) down to one number per
    # (q_pos, k_pos) per batch element.
    while attn.dim() > 3:
        attn = attn.mean(dim=1)
    # attn now [B, q_len, kv_len]. CLS is q_pos = 0.
    cls_attn = attn[:, 0, :]  # [B, kv_len]
    return out, cls_attn
