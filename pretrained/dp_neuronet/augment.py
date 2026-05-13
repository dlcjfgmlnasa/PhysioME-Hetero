# -*- coding:utf-8 -*-
"""Biosignal augmentation pool for DINOv3-style multi-crop SSL.

All ops work on float tensors shaped ``[B, T]`` on whatever device the input
sits on (CPU or CUDA — augmentation is meant to be batched on GPU at training
time, so we avoid numpy fallbacks). Each op samples its randomness per-sample
in the batch to maximize view diversity within a forward pass.

Augmentations are intentionally conservative for biosignals: amplitude
distortions and temporal jitter that preserve the diagnostic morphology
(QRS complex, pulse contour, capnograph waveform) rather than e.g. SpecAugment
freq-masking that can destroy frequency-domain pathology cues.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    amp_scale_min: float = 0.8
    amp_scale_max: float = 1.25
    amp_shift_std: float = 0.05    # baseline shift in std-normalised units
    noise_std_min: float = 0.0
    noise_std_max: float = 0.05    # Gaussian noise std on z-scored signal
    time_mask_max_ratio: float = 0.1  # mask up to 10% of timesteps per view
    time_mask_n_blocks: int = 2     # number of contiguous masked blocks
    polarity_flip_prob: float = 0.0  # 0 by default — flipping kills ABP/PPG morphology
    time_reverse_prob: float = 0.0   # 0 — same reason for cardiac signals


def random_temporal_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    """Random contiguous crop along the time axis. ``x: [B, T] -> [B, length]``."""
    b, t = x.shape
    if length >= t:
        return x
    max_start = t - length
    starts = torch.randint(0, max_start + 1, (b,), device=x.device)
    idx = starts[:, None] + torch.arange(length, device=x.device)[None, :]
    return torch.gather(x, 1, idx)


def amplitude_jitter(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    b = x.shape[0]
    scale = torch.empty(b, 1, device=x.device).uniform_(
        cfg.amp_scale_min, cfg.amp_scale_max,
    )
    shift = torch.randn(b, 1, device=x.device) * cfg.amp_shift_std
    return x * scale + shift


def gaussian_noise(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    b = x.shape[0]
    std = torch.empty(b, 1, device=x.device).uniform_(
        cfg.noise_std_min, cfg.noise_std_max,
    )
    return x + torch.randn_like(x) * std


def time_mask(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    if cfg.time_mask_max_ratio <= 0 or cfg.time_mask_n_blocks <= 0:
        return x
    b, t = x.shape
    out = x.clone()
    max_block = max(1, int(t * cfg.time_mask_max_ratio))
    for _ in range(cfg.time_mask_n_blocks):
        block_lens = torch.randint(1, max_block + 1, (b,), device=x.device)
        starts = torch.randint(0, t, (b,), device=x.device)
        ends = torch.clamp(starts + block_lens, max=t)
        for i in range(b):
            out[i, starts[i]:ends[i]] = 0.0
    return out


def maybe_polarity_flip(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    if cfg.polarity_flip_prob <= 0:
        return x
    b = x.shape[0]
    flip = (torch.rand(b, 1, device=x.device) < cfg.polarity_flip_prob).float()
    return x * (1.0 - 2.0 * flip)


def maybe_time_reverse(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    if cfg.time_reverse_prob <= 0:
        return x
    b = x.shape[0]
    rev = (torch.rand(b, device=x.device) < cfg.time_reverse_prob)
    if rev.any():
        x = x.clone()
        x[rev] = torch.flip(x[rev], dims=[-1])
    return x


def apply_augment(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    x = amplitude_jitter(x, cfg)
    x = gaussian_noise(x, cfg)
    x = time_mask(x, cfg)
    x = maybe_polarity_flip(x, cfg)
    x = maybe_time_reverse(x, cfg)
    return x
