# -*- coding:utf-8 -*-
"""DINO CLS-level self-distillation loss.

Mirrors ``dinov3/loss/dino_clstoken_loss.py`` from the official Meta
reference. The reference keeps both centering paths (running-mean and
Sinkhorn-Knopp) but the meta-arch ``asserts cfg.train.centering ==
"sinkhorn_knopp"`` (ssl_meta_arch.py L41), so the training path is SK-only.
We provide the same two methods for parity and parameterise the choice;
the trainer wires the default to SK.

Hyperparameter defaults follow the reference:
  * ``student_temp = 0.1`` (hardcoded in DINOLoss.__init__)
  * ``center_momentum = 0.9`` (used only by ``softmax_center_teacher``)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int,
                 student_temp: float = 0.1,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer('center', torch.zeros(1, out_dim))

    # ---- Teacher distribution producers ----------------------------------

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output: torch.Tensor,
                               teacher_temp: float) -> torch.Tensor:
        """Running-mean DINO centering (kept for parity; not the default path)."""
        return F.softmax(
            (teacher_output - self.center) / teacher_temp, dim=-1,
        )

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor,
                               teacher_temp: float,
                               n_iterations: int = 3) -> torch.Tensor:
        """Sinkhorn-Knopp soft assignment over the prototype axis.

        Returns ``[N, K]`` non-negative matrix Q with row-sums 1 and
        column-sums N/K. Single-GPU variant of the reference's
        ``sinkhorn_knopp_teacher`` (the reference all-reduces ``sum_Q`` and
        ``sum_of_rows`` / ``sum_of_cols`` across the process subgroup; on
        a single GPU we just skip the all-reduces).
        """
        Q = torch.exp(teacher_output.float() / teacher_temp).t()  # [K, N]
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
        return Q.t()  # [N, K]

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Single-step running-mean center update.

        The reference uses an async reduce/apply queue so the center used at
        step t is the one from step t-1. On single-GPU we just do the obvious
        synchronous update; it has the same fixed point and is fine for our
        scale.
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum,
        )

    # ---- Forward ---------------------------------------------------------

    def forward(self,
                student_global_logits: torch.Tensor,   # [n_g * B, K]
                student_local_logits: torch.Tensor,     # [n_l * B, K] (may be empty)
                teacher_global_distribution: torch.Tensor,   # [n_g * B, K]
                n_global: int,
                n_local: int,
                ignore_diagonal: bool = True) -> torch.Tensor:
        """Per-pair cross-entropy with the same normalization as the reference.

        Loss = (sum of CE over all valid (teacher, student) pairs and the
        batch) / (B * (n_global * n_global - diag) + B * n_global * n_local)
        where ``diag = n_global`` when ``ignore_diagonal=True``, else 0.
        """
        if n_global < 1:
            raise ValueError('n_global must be >= 1')
        batch_size_g = student_global_logits.shape[0] // n_global

        s_g = F.log_softmax(student_global_logits / self.student_temp, dim=-1)
        s_g_v = s_g.view(n_global, batch_size_g, -1)
        t_v = teacher_global_distribution.view(n_global, batch_size_g, -1)

        # CE[t_i, s_j, b] = -(t_i[b] * s_j[b]).sum() — shape [n_g, n_g, B]
        ce_gg = -(t_v.unsqueeze(1) * s_g_v.unsqueeze(0)).sum(dim=-1)
        if ignore_diagonal:
            diag_idx = torch.arange(n_global, device=ce_gg.device)
            ce_gg[diag_idx, diag_idx] = 0.0
            n_valid_gg = n_global * n_global - n_global
        else:
            n_valid_gg = n_global * n_global

        total_sum = ce_gg.sum()
        total_pairs = n_valid_gg

        if n_local > 0 and student_local_logits.numel() > 0:
            batch_size_l = student_local_logits.shape[0] // n_local
            s_l = F.log_softmax(student_local_logits / self.student_temp, dim=-1)
            s_l_v = s_l.view(n_local, batch_size_l, -1)
            ce_gl = -(t_v.unsqueeze(1) * s_l_v.unsqueeze(0)).sum(dim=-1)
            total_sum = total_sum + ce_gl.sum()
            total_pairs += n_global * n_local

        denom = float(batch_size_g) * float(total_pairs)
        if denom <= 0:
            return student_global_logits.new_zeros(())
        return total_sum / denom
