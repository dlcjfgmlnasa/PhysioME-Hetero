# -*- coding:utf-8 -*-
"""iBOT block masking, 1D-biosignal analogue of the DINOv3 reference.

Reference: ``dinov3/data/masking.py`` (``MaskingGenerator``) and
``dinov3/data/collate.py`` lines 40-58 (batch sampling pattern).

The 2D BEiT generator from the reference samples block masks with
aspect-ratio constraints. For 1D biosignals there is no second axis,
so we degenerate to contiguous block sampling: per-crop draw a target
mask ratio from ``[mask_ratio_min, mask_ratio_max]``, then greedily
sample non-overlapping blocks of length in ``[min_block_size, max_block_size]``
until the target count is met. Only ``mask_sample_probability`` of the
``n_crops`` get a non-empty mask; the rest get all-False (matches the
``n_samples_masked = int(B * mask_probability)`` pattern in collate.py).
"""
from __future__ import annotations

import torch


def make_block_mask(n_crops: int, n_patches: int,
                    mask_ratio_min: float, mask_ratio_max: float,
                    mask_sample_probability: float,
                    min_block_size: int = 1,
                    max_block_size: int = 0,
                    device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Return ``[n_crops, n_patches]`` bool mask.

    Args:
        n_crops: number of crops in the (flattened) batch dimension.
        n_patches: number of patches per crop (after frame backbone).
        mask_ratio_min / mask_ratio_max: target mask ratios are drawn
            uniformly in this range and spread across the masked crops
            (``torch.linspace(min, max, n_masked+1)[1:]``).
        mask_sample_probability: fraction of ``n_crops`` that receive a
            non-empty mask. The rest stay all-False.
        min_block_size: minimum block length per single span.
        max_block_size: maximum block length per single span; ``0`` means
            ``n_patches`` (a single span may cover the full crop).
    """
    mask = torch.zeros((n_crops, n_patches), dtype=torch.bool, device=device)
    if (mask_sample_probability <= 0 or n_patches <= 0
            or mask_ratio_max <= 0):
        return mask

    n_masked = int(round(n_crops * mask_sample_probability))
    if n_masked <= 0:
        return mask

    perm = torch.randperm(n_crops, device=device)
    selected = perm[:n_masked].tolist()

    if max_block_size <= 0:
        max_block_size = n_patches
    min_block_size = max(1, min_block_size)
    max_block_size = max(min_block_size, min(max_block_size, n_patches))

    # Spread mask ratios linearly across the selected crops (mirrors the
    # MaskingGenerator(N * prob_max) trick that gives each crop a slightly
    # different target so the batch covers the [min, max] range densely).
    ratios = torch.linspace(
        mask_ratio_min, mask_ratio_max, steps=n_masked + 1, device=device,
    )[1:].tolist()

    for crop_idx, target_ratio in zip(selected, ratios):
        n_target = max(min_block_size,
                       int(round(target_ratio * n_patches)))
        n_target = min(n_target, n_patches)
        placed = 0
        # Greedy non-overlapping placement; bounded loop so we never spin
        # on a saturated mask.
        attempts = 0
        while placed < n_target and attempts < 8 * n_patches:
            attempts += 1
            remaining = n_target - placed
            blk = min(remaining,
                      int(torch.randint(min_block_size, max_block_size + 1,
                                        (1,), device=device).item()))
            start = int(torch.randint(0, n_patches - blk + 1,
                                      (1,), device=device).item())
            seg = mask[crop_idx, start:start + blk]
            new = (~seg).sum().item()
            if new == 0:
                continue
            mask[crop_idx, start:start + blk] = True
            placed += int(new)
    return mask
