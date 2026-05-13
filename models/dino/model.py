# -*- coding:utf-8 -*-
"""Phase-1 unimodal SSL model: DINOv3-style self-distillation.

Replaces the prior NeuroNet (TF-C: recon + time/freq/cross contrastive) and
MaskedAutoEncoderViT classes. The encoder backbone (CNN frame backbone +
RoPE Transformer) is unchanged so existing Phase-2 code that pulls
``frame_backbone`` / ``patch_embed`` / ``encoder`` / ``cls_token`` continues
to work after a ckpt-format migration.

Two public classes:

``BiosignalEncoder``
    Encoder-only module reused by Phase-2 to build the per-modal frozen
    backbone. Identical to the former ``NeuroNetEncoder``; renamed for
    accuracy now that the Phase-1 SSL objective is DINOv3, not NeuroNet's
    TF-C.

``BiosignalDINO``
    Phase-1 trainer-side wrapper. Student + EMA teacher copies of
    ``BiosignalEncoder``, a shared ``DINOHead``, the centering buffer,
    multi-crop forward, DINO + iBOT losses, and EMA update / centering
    utilities. Phase-2 only consumes ``self.student_encoder`` (saved
    standalone under ``encoder_state`` in the ckpt); the head / teacher /
    center buffer are SSL-only and discarded at Phase-2 init.
"""
from __future__ import annotations

import copy
import math
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dino.dino_head import DINOHead
from models.dino.resnet1d import FrameBackBone
from models.transformer import (
    QueryKeyProjection, RMSNorm, RotaryProjection, TransformerEncoder,
)


def _build_rope_encoder(d_model: int, num_heads: int, num_layers: int) -> TransformerEncoder:
    """uni2ts-derived encoder: GQA + GLU FFN + RMSNorm + RoPE.

    Phase-1 (60 s) checkpoints transfer to Phase-2 windows of any length
    without an absolute-pos-embed interpolation step because RoPE handles
    arbitrary sequence lengths natively.
    """
    return TransformerEncoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        pre_norm=True,
        use_glu=True,
        use_qk_norm=True,
        norm_layer=RMSNorm,
        d_cond=0,
        time_qk_proj_layer=partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            kwargs={'max_len': 4096},
        ),
    )


def frame_size(fs: int, second: int, time_window: int, time_step: float
               ) -> Tuple[int, int]:
    """Closed-form ``(num_frames, window_samples)`` for the framing params.

        F = floor((size - window) / step) + 1   if size >= window else 0
    """
    size = fs * second
    step = int(time_step * fs)
    window = int(time_window * fs)
    if size < window:
        return 0, window
    num_frames = (size - window) // step + 1
    return num_frames, window


class BiosignalEncoder(nn.Module):
    """Encoder used by both Phase-1 SSL and Phase-2 backbone reuse.

    Output shape is ``[B, num_frames + 1, encoder_embed_dim]`` with the
    leading position being the CLS token. ``frame_step`` and frame size are
    derived from ``(fs, time_window, time_step)`` via :func:`frame_size`.
    """

    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 freeze_cls_token: bool = False):
        super().__init__()
        self.fs, self.second = fs, second
        self.time_window, self.time_step = time_window, time_step
        self.window_samples = int(time_window * fs)
        self.frame_step = int(time_step * fs)
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_heads = encoder_heads
        self.encoder_depths = encoder_depths

        self.num_patches, self.frame_size = frame_size(
            fs=fs, second=second, time_window=time_window, time_step=time_step,
        )

        self.frame_backbone = FrameBackBone(fs=fs, window=time_window)
        self.patch_embed = nn.Linear(self.frame_backbone.feature_num, encoder_embed_dim)
        self.encoder = _build_rope_encoder(
            d_model=encoder_embed_dim,
            num_heads=encoder_heads,
            num_layers=encoder_depths,
        )
        self.encoder_norm = self.encoder.norm  # alias for ckpt-loading utilities
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if freeze_cls_token:
            self.cls_token.requires_grad = False

    def make_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Slice ``[B, T]`` into ``[B, F, W]`` frames via ``Tensor.unfold``."""
        return x.unfold(
            dimension=-1, size=self.window_samples, step=self.frame_step,
        ).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.make_frame(x)
        x = self.frame_backbone(x)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        return x


# ─────────────────────────────────────────────────────────────────────
# BiosignalDINO (Phase-1 SSL wrapper)
# ─────────────────────────────────────────────────────────────────────


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


@torch.no_grad()
def _ema_copy(target: nn.Module, source: nn.Module, momentum: float) -> None:
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
    for b_t, b_s in zip(target.buffers(), source.buffers()):
        b_t.data.copy_(b_s.data)


class BiosignalDINO(nn.Module):
    """DINOv3-style self-distillation for unimodal biosignal SSL.

    Forward consumes a multi-crop bundle (globals + locals) and returns the
    total SSL loss plus a dict of unweighted sub-losses for logging. The
    teacher network and centering buffer are owned here.

    Loss = dino_weight * L_dino + ibot_weight * L_ibot

      L_dino: each *student* view (global or local) predicts the sharpened-
              and-centered distribution emitted by every *teacher* global
              view, except the same-crop pairing. Mean over pairs.
      L_ibot: random patch positions in each student global crop are
              replaced by a learnable mask token. At those positions, the
              student predicts the teacher's per-patch distribution at the
              same positions on the unmasked input. Off when ``ibot_weight=0``.

    Centering: a running mean of the raw teacher prototype logits is kept
    on-device and subtracted before sharpening. Updated via
    :meth:`update_center` after each forward.
    """

    def __init__(self,
                 fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 head_hidden_dim: int = 2048,
                 head_bottleneck_dim: int = 256,
                 head_n_prototypes: int = 8192,
                 head_n_layers: int = 3,
                 student_temp: float = 0.1,
                 teacher_temp: float = 0.04,
                 center_momentum: float = 0.9,
                 ibot_weight: float = 1.0,
                 ibot_mask_ratio: float = 0.3,
                 dino_weight: float = 1.0):
        super().__init__()
        self.student_encoder = BiosignalEncoder(
            fs=fs, second=second,
            time_window=time_window, time_step=time_step,
            encoder_embed_dim=encoder_embed_dim,
            encoder_heads=encoder_heads,
            encoder_depths=encoder_depths,
            freeze_cls_token=False,
        )
        self.student_head = DINOHead(
            in_dim=encoder_embed_dim,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            n_prototypes=head_n_prototypes,
            n_layers=head_n_layers,
        )

        # Teacher: deep-copied at init, then EMA-updated from the student.
        # Frozen w.r.t. autograd (the no_grad on update + here is belt-and-suspenders).
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        _set_requires_grad(self.teacher_encoder, False)
        _set_requires_grad(self.teacher_head, False)

        # iBOT mask token applied at patch-embedded positions (post patch_embed).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.register_buffer(
            'center', torch.zeros(1, head_n_prototypes),
        )

        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.center_momentum = float(center_momentum)
        self.ibot_weight = float(ibot_weight)
        self.ibot_mask_ratio = float(ibot_mask_ratio)
        self.dino_weight = float(dino_weight)
        self.n_prototypes = int(head_n_prototypes)

    # ─────────────────────────────────────────────────────────────
    # Encoder helpers (also used by probe code via forward_latent)
    # ─────────────────────────────────────────────────────────────

    def forward_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Single-crop CLS latent for downstream probing.

        Returns ``[B, encoder_embed_dim]`` — the CLS token output of the
        student encoder. Run under no_grad; caller is responsible for
        ``self.eval()``.
        """
        z = self.student_encoder(x)
        return z[:, 0, :]

    # ─────────────────────────────────────────────────────────────
    # Loss-side forward (used during training)
    # ─────────────────────────────────────────────────────────────

    def _encode_globals_masked(self, globals_: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Student forward on global crops with iBOT-style patch masking.

        Returns:
            cls_logits  : [n_g * B, K]       — CLS prototype logits
            patch_logits: [n_g * B, F, K] — per-patch prototype logits
                          (used only at masked positions for L_ibot)
            patch_mask  : [n_g * B, F]   bool — True at masked positions
        """
        enc = self.student_encoder
        x = enc.make_frame(globals_)
        x = enc.frame_backbone(x)
        x = enc.patch_embed(x)  # [n_g * B, F, D]
        n, f, d = x.shape

        if self.ibot_weight > 0 and self.ibot_mask_ratio > 0:
            mask_prob = torch.full((n, f), self.ibot_mask_ratio, device=x.device)
            patch_mask = torch.bernoulli(mask_prob).bool()
            x = torch.where(patch_mask.unsqueeze(-1), self.mask_token, x)
        else:
            patch_mask = torch.zeros((n, f), dtype=torch.bool, device=x.device)

        cls_tokens = enc.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = enc.encoder(x)
        cls_feat, patch_feat = x[:, 0, :], x[:, 1:, :]
        cls_logits = self.student_head(cls_feat)
        if self.ibot_weight > 0:
            patch_logits = self.student_head(patch_feat.reshape(n * f, d)).reshape(n, f, -1)
        else:
            patch_logits = patch_feat.new_zeros((n, f, self.n_prototypes))
        return cls_logits, patch_logits, patch_mask

    def _encode_locals(self, locals_: torch.Tensor) -> torch.Tensor:
        """Student forward on local crops — CLS only. ``[n_l * B, K]``."""
        if locals_.shape[0] == 0:
            return locals_.new_zeros((0, self.n_prototypes))
        z = self.student_encoder(locals_)
        return self.student_head(z[:, 0, :])

    @torch.no_grad()
    def _encode_teacher(self, globals_: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher forward on globals — CLS + per-patch logits.

        Returns:
            cls_logits  : [n_g * B, K]
            patch_logits: [n_g * B, F, K]
        """
        z = self.teacher_encoder(globals_)
        cls_feat, patch_feat = z[:, 0, :], z[:, 1:, :]
        cls_logits = self.teacher_head(cls_feat)
        if self.ibot_weight > 0:
            n, f, d = patch_feat.shape
            patch_logits = self.teacher_head(patch_feat.reshape(n * f, d)).reshape(n, f, -1)
        else:
            patch_logits = patch_feat.new_zeros(
                (patch_feat.shape[0], patch_feat.shape[1], self.n_prototypes)
            )
        return cls_logits, patch_logits

    def forward(self,
                globals_: torch.Tensor,
                locals_: torch.Tensor,
                n_global: int,
                n_local: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Multi-crop self-distillation forward.

        Args:
            globals_: ``[n_global * B, T_g]`` (crops cat'd on batch axis,
                      crop-major: first B rows are crop 0, next B are crop 1, ...).
            locals_ : ``[n_local  * B, T_l]`` with same crop-major layout.
        Returns:
            loss : scalar total loss = dino_weight * L_dino + ibot_weight * L_ibot
            logs : dict of unweighted ``dino``, ``ibot`` losses for logging.
        """
        if n_global < 1:
            raise ValueError('n_global must be >= 1 for self-distillation.')
        batch_size_g = globals_.shape[0] // n_global
        if batch_size_g * n_global != globals_.shape[0]:
            raise RuntimeError('globals_ batch axis is not divisible by n_global')

        # Student
        s_g_cls, s_g_patch, patch_mask = self._encode_globals_masked(globals_)
        s_l_cls = self._encode_locals(locals_)

        # Teacher (no_grad)
        t_g_cls, t_g_patch = self._encode_teacher(globals_)

        # Centering + sharpening for the teacher CLS distribution.
        t_cls_sharp = (t_g_cls - self.center) / self.teacher_temp
        t_cls_dist = F.softmax(t_cls_sharp, dim=-1)  # [n_g*B, K]

        # Student log-softmax (full prototype distribution over K).
        s_g_logp = F.log_softmax(s_g_cls / self.student_temp, dim=-1)
        s_l_logp = (F.log_softmax(s_l_cls / self.student_temp, dim=-1)
                    if s_l_cls.shape[0] > 0 else None)

        # Reshape to per-crop tensors.
        t_cls_dist_v = t_cls_dist.view(n_global, batch_size_g, -1)
        s_g_logp_v = s_g_logp.view(n_global, batch_size_g, -1)
        if s_l_logp is not None:
            batch_size_l = locals_.shape[0] // max(n_local, 1)
            s_l_logp_v = s_l_logp.view(n_local, batch_size_l, -1)
        else:
            s_l_logp_v = None

        # ─── DINO loss: pair every teacher global with every student crop ─────
        terms: List[torch.Tensor] = []
        for i in range(n_global):
            t_i = t_cls_dist_v[i]
            # Student globals (skip same-crop pair).
            for j in range(n_global):
                if j == i:
                    continue
                terms.append(-(t_i * s_g_logp_v[j]).sum(dim=-1).mean())
            # Student locals.
            if s_l_logp_v is not None and n_local > 0:
                for j in range(n_local):
                    terms.append(-(t_i * s_l_logp_v[j]).sum(dim=-1).mean())
        if terms:
            loss_dino = torch.stack(terms).mean()
        else:
            loss_dino = globals_.new_zeros(())

        # ─── iBOT loss: masked-patch prediction on student globals ────────────
        if self.ibot_weight > 0 and patch_mask.any():
            t_patch_sharp = (t_g_patch - self.center) / self.teacher_temp
            t_patch_dist = F.softmax(t_patch_sharp, dim=-1)
            s_patch_logp = F.log_softmax(s_g_patch / self.student_temp, dim=-1)
            # Per-position CE only at masked positions.
            ce = -(t_patch_dist * s_patch_logp).sum(dim=-1)  # [n_g*B, F]
            mask_f = patch_mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            loss_ibot = (ce * mask_f).sum() / denom
        else:
            loss_ibot = globals_.new_zeros(())

        loss = self.dino_weight * loss_dino + self.ibot_weight * loss_ibot

        # Centering update (raw, un-sharpened teacher logits).
        self._update_center(t_g_cls.detach(), t_g_patch.detach() if self.ibot_weight > 0 else None)

        return loss, {'dino': loss_dino.detach(),
                      'ibot': loss_ibot.detach()}

    # ─────────────────────────────────────────────────────────────
    # EMA + centering utilities
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        _ema_copy(self.teacher_encoder, self.student_encoder, momentum)
        _ema_copy(self.teacher_head, self.student_head, momentum)

    @torch.no_grad()
    def _update_center(self, t_cls: torch.Tensor,
                       t_patch: Optional[torch.Tensor]) -> None:
        # Combine CLS + patch (if iBOT on) into a single running mean.
        batch_center = t_cls.mean(dim=0, keepdim=True)
        if t_patch is not None:
            n, f, k = t_patch.shape
            patch_center = t_patch.reshape(n * f, k).mean(dim=0, keepdim=True)
            batch_center = 0.5 * (batch_center + patch_center)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum,
        )


def cosine_schedule(start: float, end: float, step: int, total_steps: int
                    ) -> float:
    """Half-cosine schedule from ``start`` at step 0 to ``end`` at step ``total_steps``."""
    if total_steps <= 0:
        return end
    t = min(max(step, 0), total_steps) / total_steps
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))


def linear_warmup(start: float, end: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return end
    return start + (end - start) * (step / warmup_steps)


if __name__ == '__main__':
    m = BiosignalDINO(
        fs=100, second=60, time_window=3, time_step=3,
        encoder_embed_dim=256, encoder_heads=8, encoder_depths=4,
        head_n_prototypes=2048,
    )
    g = torch.randn(8, 6000)  # 2 globals x 4 batch
    l = torch.randn(16, 1500)  # 4 locals x 4 batch
    loss, logs = m(g, l, n_global=2, n_local=4)
    print(f'loss={loss.item():.4f} dino={logs["dino"].item():.4f} '
          f'ibot={logs["ibot"].item():.4f}')
