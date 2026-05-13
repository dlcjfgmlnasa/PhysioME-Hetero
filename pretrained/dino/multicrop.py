# -*- coding:utf-8 -*-
"""Multi-crop pipeline for DINOv3-style training.

Given a batch ``x: [B, T_full]`` (the full SSL segment, default 60 s at 100 Hz
= 6000 samples), produces:

  * ``n_global`` global crops of length ``global_samples`` (default 60 s ≈ 6000)
  * ``n_local``  local crops  of length ``local_samples``  (default 15 s ≈ 1500)

Each crop is independently augmented. Globals are typically larger / less
aggressively augmented; locals are short windows with more jitter.

Returned shapes:
  globals: tensor ``[n_global * B, global_samples]`` (crops concatenated on B)
  locals : tensor ``[n_local  * B, local_samples]``

Concatenation order is crop-major (all of crop_0, then all of crop_1, ...) so
that downstream code can ``.view(n_global, B, ...)`` to recover per-crop axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from pretrained.dino.augment import (
    AugmentConfig, apply_augment, random_temporal_crop,
)


@dataclass
class MultiCropConfig:
    n_global: int = 2
    n_local: int = 6
    global_samples: int = 6000   # 60 s @ 100 Hz
    local_samples: int = 1500    # 15 s @ 100 Hz
    global_aug: AugmentConfig = None
    local_aug: AugmentConfig = None

    def __post_init__(self):
        if self.global_aug is None:
            self.global_aug = AugmentConfig(
                amp_scale_min=0.9, amp_scale_max=1.1,
                noise_std_max=0.02, time_mask_max_ratio=0.05,
            )
        if self.local_aug is None:
            self.local_aug = AugmentConfig(
                amp_scale_min=0.8, amp_scale_max=1.25,
                noise_std_max=0.05, time_mask_max_ratio=0.10,
            )


def make_crops(x: torch.Tensor, cfg: MultiCropConfig
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the (globals, locals) tensors from one batch ``x: [B, T]``."""
    if x.dim() != 2:
        raise ValueError(f'expected x.shape=[B, T], got {tuple(x.shape)}')
    b, t = x.shape
    if t < cfg.global_samples:
        raise ValueError(
            f'global_samples={cfg.global_samples} but input T={t}. '
            'Set global_samples <= ssl segment length.'
        )

    g_list: List[torch.Tensor] = []
    for _ in range(cfg.n_global):
        crop = random_temporal_crop(x, cfg.global_samples)
        crop = apply_augment(crop, cfg.global_aug)
        g_list.append(crop)
    globals_ = torch.cat(g_list, dim=0) if g_list else x.new_zeros((0, cfg.global_samples))

    l_list: List[torch.Tensor] = []
    for _ in range(cfg.n_local):
        crop = random_temporal_crop(x, cfg.local_samples)
        crop = apply_augment(crop, cfg.local_aug)
        l_list.append(crop)
    locals_ = torch.cat(l_list, dim=0) if l_list else x.new_zeros((0, cfg.local_samples))

    return globals_, locals_
