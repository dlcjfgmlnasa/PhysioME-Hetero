# -*- coding:utf-8 -*-
"""Minimal LoRA replacement for ``peft.get_peft_model``.

Why hand-rolled instead of peft:
  * peft pulls in transformers / accelerate / safetensors purely as transitive
    dependencies; PhysioME uses none of them. Removing peft drops ~four heavy
    packages from the requirements footprint.
  * peft wraps modules under a ``base_model.model.<...>.base_layer.weight``
    prefix that complicates state-dict inspection and the direct submodule
    transfers we use in ``_load_pretrained_unimodal``. Hand-rolled LoRA leaves
    the parent state-dict path untouched -- ``out_proj.base.weight`` /
    ``out_proj.lora_A`` / ``out_proj.lora_B`` slot in cleanly.
  * peft routes adapters by substring-matching ``target_modules`` against
    every Linear name, which silently mis-targets if attention layers are
    renamed -- we hit this when migrating from timm's ``attn.proj`` to GQA's
    ``out_proj``.

Implements LoRA per Hu et al., 2021 (arXiv:2106.09685) with the rsLoRA
scaling ``alpha / sqrt(r)`` (Kalajdzievski 2023). DoRA is reserved for a
future ablation.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` and adds a trainable rank-``r`` residual.

    Forward:
        ``out = base(x) + scale * (dropout(x) @ A^T @ B^T)``

    Initialization:
        ``A ~ N(0, 1/r)``, ``B = 0`` -- LoRA output starts at zero so the
        wrapped layer initially behaves identically to the base, safe to
        slot into any pretrained model.

    Scaling:
        rsLoRA -> ``alpha / sqrt(r)`` (default).
        Set ``rslora=False`` for the original ``alpha / r``.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float,
                 dropout: float = 0.0, rslora: bool = True):
        super().__init__()
        if r <= 0:
            raise ValueError(f'LoRA rank must be positive, got {r}')
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = alpha / (math.sqrt(r) if rslora else r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear(x, W) computes ``x @ W.T``; we pass W = B @ A of shape (out, in).
        update = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scale
        return self.base(x) + update

    def extra_repr(self) -> str:
        return (f'in={self.base.in_features}, out={self.base.out_features}, '
                f'r={self.r}, alpha={self.alpha}, scale={self.scale:.3f}')


def apply_lora(module: nn.Module,
               target_attrs: Iterable[str] = ('out_proj',),
               r: int = 4,
               alpha: float = 16.0,
               dropout: float = 0.05,
               rslora: bool = True) -> nn.Module:
    """Freeze every parameter in ``module`` and replace each ``nn.Linear``
    whose attribute name is in ``target_attrs`` with a ``LoRALinear``.

    Returns the same ``module`` for chaining (matches the
    ``get_peft_model(model, cfg)`` ergonomics).

    Important: call this AFTER loading any pretrained weights into the base
    Linears -- LoRA adapters wrap the existing ``nn.Linear`` and preserve
    its weights in ``self.base``.
    """
    target_set = set(target_attrs)
    for p in module.parameters():
        p.requires_grad = False
    n_replaced = 0
    for sub in list(module.modules()):
        for attr in target_set:
            child = getattr(sub, attr, None)
            if isinstance(child, nn.Linear):
                setattr(sub, attr, LoRALinear(child, r=r, alpha=alpha,
                                              dropout=dropout, rslora=rslora))
                n_replaced += 1
    if n_replaced == 0:
        raise RuntimeError(
            f'apply_lora matched zero Linear modules with names {tuple(target_set)!r}. '
            'Check the target attribute names against the actual attention module.'
        )
    return module


__all__ = ['LoRALinear', 'apply_lora']
