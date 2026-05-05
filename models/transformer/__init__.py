# -*- coding:utf-8 -*-
"""Transformer building blocks ported from Biosignal-Foundation-Model.

The block stack is uni2ts (Salesforce, Apache 2.0)-derived: GQA + GLU FFN
+ RMSNorm + RoPE via QueryKeyProjection. PhysioME uses ``d_cond=0`` so the
AdaLN modulation path is bypassed and every layer norm is plain RMSNorm.
"""
from .norm import RMSNorm, AdaRMSNorm
from .attention import GroupedQueryAttention, MultiHeadAttention, MultiQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward, MoEFeedForward
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .position import (
    AttentionBias, BinaryAttentionBias,
    Projection, RotaryProjection, QueryKeyProjection,
)
from .lora import LoRALinear, apply_lora

__all__ = [
    'RMSNorm', 'AdaRMSNorm',
    'GroupedQueryAttention', 'MultiHeadAttention', 'MultiQueryAttention',
    'FeedForward', 'GatedLinearUnitFeedForward', 'MoEFeedForward',
    'TransformerEncoder', 'TransformerEncoderLayer',
    'AttentionBias', 'BinaryAttentionBias',
    'Projection', 'RotaryProjection', 'QueryKeyProjection',
    'LoRALinear', 'apply_lora',
]
