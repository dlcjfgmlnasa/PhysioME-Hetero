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

    Matches the official DINOv3 reference for the four points where v1/v2
    and v3 diverge:

      * **Separate iBOT head.** DINO (CLS) and iBOT (patch) heads are
        distinct ``DINOHead`` instances with their own hyperparameters
        (their own prototype count, hidden / bottleneck dims).
      * **Sinkhorn-Knopp centering** on the teacher logits (3 iterations)
        instead of running-mean centering. No center buffer is kept.
      * **Plain last_layer.** The DINO head's prototype linear is a plain
        ``nn.Linear(bottleneck, K, bias=False)``. The "freeze last layer
        for N epochs" stability trick is implemented as a zero-LR window
        in the trainer, NOT by wrapping the linear in weight_norm.
      * **Block-wise iBOT masking** with per-crop mask ratio drawn in
        ``[ibot_mask_ratio_min, ibot_mask_ratio_max]`` and applied only
        to ``ibot_mask_sample_probability`` of the global crops (the rest
        get empty masks). Loss is per-image-normalised (``1/n_masked_in_image``)
        then averaged over all global crops (including the unmasked ones,
        which contribute 0). Matches ``iBOTPatchLoss.forward_masked`` in
        the reference repo.

    Loss = dino_weight * L_dino + ibot_weight * L_ibot

      L_dino: every student crop (global or local) predicts the sharpened
              + SK-normalised distribution from every teacher global view,
              except the same-view (i==j) pair among globals.
      L_ibot: at masked patch positions in each global crop, the student
              predicts the teacher's per-patch distribution at the same
              positions on the unmasked input. iBOT is global-only.
    """

    def __init__(self,
                 fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 # DINO head.
                 dino_head_hidden_dim: int = 2048,
                 dino_head_bottleneck_dim: int = 256,
                 dino_head_n_prototypes: int = 8192,
                 dino_head_n_layers: int = 3,
                 # iBOT head (separate; can differ).
                 ibot_head_hidden_dim: int = 2048,
                 ibot_head_bottleneck_dim: int = 256,
                 ibot_head_n_prototypes: int = 8192,
                 ibot_head_n_layers: int = 3,
                 # Temperatures + loss weights.
                 student_temp: float = 0.1,
                 teacher_temp: float = 0.04,
                 dino_weight: float = 1.0,
                 ibot_weight: float = 1.0,
                 # iBOT block masking.
                 ibot_mask_ratio_min: float = 0.1,
                 ibot_mask_ratio_max: float = 0.5,
                 ibot_mask_sample_probability: float = 0.5,
                 ibot_min_block_size: int = 1,
                 # SK centering.
                 sinkhorn_n_iters: int = 3):
        super().__init__()
        self.student_encoder = BiosignalEncoder(
            fs=fs, second=second,
            time_window=time_window, time_step=time_step,
            encoder_embed_dim=encoder_embed_dim,
            encoder_heads=encoder_heads,
            encoder_depths=encoder_depths,
            freeze_cls_token=False,
        )
        self.student_dino_head = DINOHead(
            in_dim=encoder_embed_dim,
            hidden_dim=dino_head_hidden_dim,
            bottleneck_dim=dino_head_bottleneck_dim,
            n_prototypes=dino_head_n_prototypes,
            n_layers=dino_head_n_layers,
        )
        self.student_ibot_head = DINOHead(
            in_dim=encoder_embed_dim,
            hidden_dim=ibot_head_hidden_dim,
            bottleneck_dim=ibot_head_bottleneck_dim,
            n_prototypes=ibot_head_n_prototypes,
            n_layers=ibot_head_n_layers,
        )

        # Teacher network: deep-copied at init, then EMA-updated from the
        # student. Frozen w.r.t. autograd.
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head)
        for m in (self.teacher_encoder, self.teacher_dino_head,
                  self.teacher_ibot_head):
            _set_requires_grad(m, False)

        # iBOT mask token applied at patch-embedded positions (post patch_embed).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.dino_weight = float(dino_weight)
        self.ibot_weight = float(ibot_weight)
        self.dino_n_prototypes = int(dino_head_n_prototypes)
        self.ibot_n_prototypes = int(ibot_head_n_prototypes)
        # iBOT block masking.
        self.ibot_mask_ratio_min = float(ibot_mask_ratio_min)
        self.ibot_mask_ratio_max = float(ibot_mask_ratio_max)
        self.ibot_mask_sample_probability = float(ibot_mask_sample_probability)
        self.ibot_min_block_size = max(1, int(ibot_min_block_size))
        self.sinkhorn_n_iters = int(sinkhorn_n_iters)

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

    # ─── iBOT block-mask generator (BEiT-style) ─────────────────────────
    def _build_ibot_block_mask(self, n_crops: int, n_patches: int,
                               device: torch.device) -> torch.Tensor:
        """Per-crop block-wise mask in ``[ibot_mask_ratio_min, _max]``.

        Only ``ibot_mask_sample_probability`` of the ``n_crops`` global crops
        get a non-empty mask; the rest get all-False. For each masked crop,
        the target ratio is drawn uniformly from ``[min, max]`` and filled
        by a single contiguous block of patches at a random offset (1-D
        analogue of BEiT's 2-D block sampling; for short patch sequences a
        single block is plenty — 8 patches at most for a 60s global crop
        downsampled to 20 frames means even one block covers the full
        ``[0.1, 0.5]`` range).
        """
        mask = torch.zeros((n_crops, n_patches), dtype=torch.bool, device=device)
        if (self.ibot_weight <= 0
                or self.ibot_mask_sample_probability <= 0
                or n_patches <= 0):
            return mask
        n_masked = int(round(n_crops * self.ibot_mask_sample_probability))
        if n_masked <= 0:
            return mask
        # Which crops get masked? Pick the first n_masked after a random shuffle.
        perm = torch.randperm(n_crops, device=device)
        masked_idx = perm[:n_masked]
        # Spread mask ratios linearly across the masked crops, like
        # ``MaskingGenerator`` in dinov3/data/collate.py L40-58.
        ratios = torch.linspace(
            self.ibot_mask_ratio_min, self.ibot_mask_ratio_max,
            steps=n_masked + 1, device=device,
        )[1:]
        for k, idx in enumerate(masked_idx.tolist()):
            n_mask_tokens = max(self.ibot_min_block_size,
                                int(round(float(ratios[k]) * n_patches)))
            n_mask_tokens = min(n_mask_tokens, n_patches)
            start = int(torch.randint(0, n_patches - n_mask_tokens + 1,
                                      (1,), device=device).item())
            mask[idx, start:start + n_mask_tokens] = True
        return mask

    def _encode_globals_masked(self, globals_: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Student forward on global crops with BEiT-style block masking.

        Returns:
            cls_logits   : [n_g * B, K_dino]
            patch_logits : [n_g * B, F, K_ibot]
            patch_mask   : [n_g * B, F] bool — True at masked positions
        """
        enc = self.student_encoder
        x = enc.make_frame(globals_)
        x = enc.frame_backbone(x)
        x = enc.patch_embed(x)  # [n_g * B, F, D]
        n, f, d = x.shape

        patch_mask = self._build_ibot_block_mask(n, f, x.device)
        if patch_mask.any():
            x = torch.where(patch_mask.unsqueeze(-1), self.mask_token, x)

        cls_tokens = enc.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = enc.encoder(x)
        cls_feat, patch_feat = x[:, 0, :], x[:, 1:, :]
        cls_logits = self.student_dino_head(cls_feat)
        if self.ibot_weight > 0:
            patch_logits = self.student_ibot_head(
                patch_feat.reshape(n * f, d)
            ).reshape(n, f, -1)
        else:
            patch_logits = patch_feat.new_zeros((n, f, self.ibot_n_prototypes))
        return cls_logits, patch_logits, patch_mask

    def _encode_locals(self, locals_: torch.Tensor) -> torch.Tensor:
        """Student forward on local crops — CLS only. ``[n_l * B, K_dino]``."""
        if locals_.shape[0] == 0:
            return locals_.new_zeros((0, self.dino_n_prototypes))
        z = self.student_encoder(locals_)
        return self.student_dino_head(z[:, 0, :])

    @torch.no_grad()
    def _encode_teacher(self, globals_: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher forward on globals — CLS + per-patch logits.

        Returns:
            cls_logits  : [n_g * B, K_dino]
            patch_logits: [n_g * B, F, K_ibot]
        """
        z = self.teacher_encoder(globals_)
        cls_feat, patch_feat = z[:, 0, :], z[:, 1:, :]
        cls_logits = self.teacher_dino_head(cls_feat)
        if self.ibot_weight > 0:
            n, f, d = patch_feat.shape
            patch_logits = self.teacher_ibot_head(
                patch_feat.reshape(n * f, d)
            ).reshape(n, f, -1)
        else:
            patch_logits = patch_feat.new_zeros(
                (patch_feat.shape[0], patch_feat.shape[1], self.ibot_n_prototypes)
            )
        return cls_logits, patch_logits

    @staticmethod
    @torch.no_grad()
    def _sinkhorn_knopp(logits: torch.Tensor, eps: float,
                        n_iters: int) -> torch.Tensor:
        """Sinkhorn-Knopp soft assignment over the prototype axis.

        ``logits``: ``[N, K]`` teacher prototype logits.
        Returns ``[N, K]`` non-negative matrix Q whose rows sum to 1 and
        whose columns sum to N/K (i.e. the prototype-marginal target),
        suitable for use as the teacher's distribution in cross-entropy.
        Matches the single-GPU version of
        ``dinov3/loss/dino_clstoken_loss.sinkhorn_knopp_teacher``.
        """
        Q = torch.exp(logits / eps).t()  # [K, N]
        n_samples = Q.shape[1]
        n_prototypes = Q.shape[0]
        sum_Q = Q.sum()
        Q /= sum_Q
        for _ in range(n_iters):
            sum_of_rows = Q.sum(dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= n_prototypes
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= n_samples
        Q *= n_samples
        return Q.t()  # [N, K]

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
            loss : scalar = dino_weight * L_dino + ibot_weight * L_ibot
            logs : dict of unweighted ``dino``, ``ibot`` losses for logging.
        """
        if n_global < 1:
            raise ValueError('n_global must be >= 1 for self-distillation.')
        batch_size_g = globals_.shape[0] // n_global
        if batch_size_g * n_global != globals_.shape[0]:
            raise RuntimeError('globals_ batch axis is not divisible by n_global')

        # Student forward (with iBOT block mask) + locals.
        s_g_cls, s_g_patch, patch_mask = self._encode_globals_masked(globals_)
        s_l_cls = self._encode_locals(locals_)

        # Teacher forward (no_grad).
        t_g_cls, t_g_patch = self._encode_teacher(globals_)

        # Sinkhorn-Knopp normalisation on teacher CLS logits.
        t_cls_dist = self._sinkhorn_knopp(
            t_g_cls.float(), eps=self.teacher_temp, n_iters=self.sinkhorn_n_iters,
        ).to(t_g_cls.dtype)

        # Student log-softmax.
        s_g_logp = F.log_softmax(s_g_cls / self.student_temp, dim=-1)
        s_l_logp = (F.log_softmax(s_l_cls / self.student_temp, dim=-1)
                    if s_l_cls.shape[0] > 0 else None)

        # Reshape per-crop.
        t_cls_dist_v = t_cls_dist.view(n_global, batch_size_g, -1)
        s_g_logp_v = s_g_logp.view(n_global, batch_size_g, -1)
        if s_l_logp is not None:
            batch_size_l = locals_.shape[0] // max(n_local, 1)
            s_l_logp_v = s_l_logp.view(n_local, batch_size_l, -1)
        else:
            s_l_logp_v = None

        # ─── DINO loss: every teacher global pairs with every student crop ─────
        terms: List[torch.Tensor] = []
        for i in range(n_global):
            t_i = t_cls_dist_v[i]
            # Student globals (skip i==j same-view pair).
            for j in range(n_global):
                if j == i:
                    continue
                terms.append(-(t_i * s_g_logp_v[j]).sum(dim=-1).mean())
            if s_l_logp_v is not None and n_local > 0:
                for j in range(n_local):
                    terms.append(-(t_i * s_l_logp_v[j]).sum(dim=-1).mean())
        loss_dino = (torch.stack(terms).mean()
                     if terms else globals_.new_zeros(()))

        # ─── iBOT loss: per-image-normalised, averaged over all globals ────────
        if self.ibot_weight > 0 and patch_mask.any():
            ng, fp = patch_mask.shape  # [n_g*B, F]
            t_patch_flat = t_g_patch.reshape(ng * fp, -1).float()
            t_patch_dist = self._sinkhorn_knopp(
                t_patch_flat, eps=self.teacher_temp,
                n_iters=self.sinkhorn_n_iters,
            ).to(t_g_patch.dtype).reshape(ng, fp, -1)
            s_patch_logp = F.log_softmax(s_g_patch / self.student_temp, dim=-1)

            # Per-patch CE, then per-image inverse-mask-count weighting,
            # then average over ALL global crops (unmasked ones contribute 0).
            ce = -(t_patch_dist * s_patch_logp).sum(dim=-1)  # [n_g*B, F]
            mask_f = patch_mask.float()
            n_masked_per_image = mask_f.sum(dim=-1).clamp_min(1.0)  # [n_g*B]
            per_image = (ce * mask_f).sum(dim=-1) / n_masked_per_image
            # Zero out rows that had no mask so they contribute 0 but still
            # count in the denominator.
            per_image = per_image * (mask_f.sum(dim=-1) > 0).float()
            loss_ibot = per_image.sum() / ng
        else:
            loss_ibot = globals_.new_zeros(())

        loss = self.dino_weight * loss_dino + self.ibot_weight * loss_ibot
        return loss, {'dino': loss_dino.detach(),
                      'ibot': loss_ibot.detach()}

    # ─────────────────────────────────────────────────────────────
    # EMA teacher update
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        _ema_copy(self.teacher_encoder, self.student_encoder, momentum)
        _ema_copy(self.teacher_dino_head, self.student_dino_head, momentum)
        _ema_copy(self.teacher_ibot_head, self.student_ibot_head, momentum)


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
        dino_head_n_prototypes=2048,
        ibot_head_n_prototypes=2048,
    )
    g = torch.randn(8, 6000)   # 2 globals x 4 batch
    l = torch.randn(16, 1500)  # 4 locals x 4 batch
    loss, logs = m(g, l, n_global=2, n_local=4)
    print(f'loss={loss.item():.4f} dino={logs["dino"].item():.4f} '
          f'ibot={logs["ibot"].item():.4f}')
