# -*- coding:utf-8 -*-
"""iBOT patch-level loss for masked-image (here: masked-time-patch) prediction.

Mirrors ``dinov3/loss/ibot_patch_loss.py``. Two teacher distribution producers
are provided for parity (running-mean center vs. Sinkhorn-Knopp); the SK path
is the default per the reference meta-arch's assert.

``forward_masked`` matches the reference averaging convention:

  * Cross-entropy is computed only at masked patch positions.
  * Per-image weight is ``1 / n_masked_in_image`` (so each masked image
    contributes a constant total weight regardless of how many patches it
    masked), then the loss is divided by ``B`` = number of crops in the
    batch (including the ones that received an empty mask). Unmasked images
    contribute 0 to the numerator but still count in the denominator.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim: int,
                 student_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer('center', torch.zeros(1, patch_out_dim))

    # ---- Teacher distribution producers ----------------------------------

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens: torch.Tensor,
                               teacher_temp: float) -> torch.Tensor:
        return F.softmax(
            (teacher_patch_tokens - self.center) / teacher_temp, dim=-1,
        )

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor,
                               teacher_temp: float,
                               n_iterations: int = 3) -> torch.Tensor:
        """SK normalisation over the prototype axis (single-GPU variant)."""
        Q = torch.exp(teacher_output.float() / teacher_temp).t()
        n_samples = Q.shape[1]
        n_prototypes = Q.shape[0]
        sum_Q = Q.sum()
        Q /= sum_Q
        for _ in range(n_iterations):
            sum_of_rows = Q.sum(dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= n_prototypes
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= n_samples
        Q *= n_samples
        return Q.t()

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens: torch.Tensor) -> None:
        batch_center = teacher_patch_tokens.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum,
        )

    # ---- Forward ---------------------------------------------------------

    def forward_masked(self,
                       student_patch_logits: torch.Tensor,   # [N, F, K]
                       teacher_patch_distribution: torch.Tensor,  # [N, F, K]
                       patch_mask: torch.Tensor,             # [N, F] bool
                       masks_weight: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """Masked-patch cross-entropy with per-image inverse-count weighting."""
        n_crops, n_patches, _ = student_patch_logits.shape
        if not patch_mask.any():
            return student_patch_logits.new_zeros(())

        s_logp = F.log_softmax(
            student_patch_logits / self.student_temp, dim=-1,
        )
        ce = -(teacher_patch_distribution * s_logp).sum(dim=-1)  # [N, F]

        if masks_weight is None:
            # 1 / (# masked patches in that image), broadcast back to [N, F].
            n_masked = patch_mask.float().sum(dim=-1).clamp_min(1.0)  # [N]
            masks_weight = (1.0 / n_masked).unsqueeze(-1).expand_as(ce)

        # Zero out unmasked positions, then sum over patches → per-image scalar.
        per_image = (ce * patch_mask.float() * masks_weight).sum(dim=-1)  # [N]
        # Average over ALL crops in the batch (including the ones with no
        # mask, which contribute 0). Matches the reference convention.
        return per_image.sum() / float(n_crops)
