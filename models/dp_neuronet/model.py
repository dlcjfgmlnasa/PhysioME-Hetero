# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from models.dp_neuronet.resnet1d import FrameBackBone
from timm.models.vision_transformer import Block
from models.utils import get_2d_sincos_pos_embed_flexible
from models.transformer import (
    TransformerEncoder, RMSNorm,
    QueryKeyProjection, RotaryProjection,
)
from models.loss import NTXentLoss
from functools import partial


def _build_projector(in_dim: int, hidden: List[int]) -> nn.Sequential:
    """Standard SSL projector: Linear -> BN -> ELU stacks ending in a bare Linear."""
    sizes = [in_dim] + list(hidden)
    layers = []
    for i, (h1, h2) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(h1, h2))
        if i != len(sizes) - 2:
            layers.append(nn.BatchNorm1d(h2))
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def _build_rope_encoder(d_model: int, num_heads: int, num_layers: int) -> TransformerEncoder:
    """uni2ts-derived encoder: GQA(MHA) + GLU FFN + RMSNorm + RoPE.

    PhysioME uses ``d_cond=0`` so layer norms are plain RMSNorm (no AdaLN).
    RoPE handles arbitrary sequence lengths natively — Phase-1 (60 s)
    checkpoints transfer to Phase-2 windows of any length without any
    interpolation step.
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


class NeuroNet(nn.Module):
    """Phase-1 unimodal SSL with **Time-Frequency Consistency** (TF-C).

    Replaces the original SimCLR pair (mask-aug contrastive + view-pair contrastive)
    with three TF-C losses (Zhang et al., NeurIPS 2022):

      * ``L_T`` — time-domain contrastive between two random masks of the same input
      * ``L_F`` — frequency-domain contrastive between two random masks of |FFT(x)|
      * ``L_TF`` — cross-domain alignment between the time-CLS and freq-CLS tokens

    Architecture: separate ``frame_backbone`` for time and freq inputs (different
    distributions / BN statistics), but **shared autoencoder** (patch_embed +
    transformer encoder + cls_token) — that shared encoder is the universal
    representation Phase-2 transfers down. The freq-magnitude vector is zero-padded
    up to the time window length so a single backbone architecture can be reused.

    Reconstruction (MAE) is still computed on both domains; the total Phase-1
    loss is ``recon + L_T + L_F + L_TF``.
    """

    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 projection_hidden: List, temperature=0.01):
        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step
        self.window_samples = int(time_window * fs)

        self.num_patches, _ = frame_size(fs=fs, second=second, time_window=time_window, time_step=time_step)
        # Time-domain backbone (raw waveform per frame).
        self.frame_backbone = FrameBackBone(fs=self.fs, window=self.time_window)
        # Frequency-domain backbone (|FFT| magnitude per frame, zero-padded to W).
        self.frame_backbone_freq = FrameBackBone(fs=self.fs, window=self.time_window)
        # Shared autoencoder: same patch_embed / encoder / cls_token / decoder for both domains.
        self.autoencoder = MaskedAutoEncoderViT(input_size=self.frame_backbone.feature_num,
                                                encoder_embed_dim=encoder_embed_dim, num_patches=self.num_patches,
                                                encoder_heads=encoder_heads, encoder_depths=encoder_depths,
                                                decoder_embed_dim=decoder_embed_dim, decoder_heads=decoder_heads,
                                                decoder_depths=decoder_depths)
        self.contrastive_loss = NTXentLoss(temperature=temperature, performance_view=False)

        # TF-C projectors: one per loss path. Each lifts the cls token into the
        # contrastive space. Separate projectors (not shared) per the TF-C paper.
        self.projector_time = _build_projector(encoder_embed_dim, projection_hidden)
        self.projector_freq = _build_projector(encoder_embed_dim, projection_hidden)
        self.projector_tfc = _build_projector(encoder_embed_dim, projection_hidden)
        self.norm_pix_loss = False

    # ------------------------------------------------------------------
    # View construction
    # ------------------------------------------------------------------
    def _frames_time(self, x: torch.Tensor) -> torch.Tensor:
        """Slice waveform into overlapping time-domain frames."""
        return self.make_frame(x)

    def _frames_freq(self, frames_time: torch.Tensor) -> torch.Tensor:
        """|FFT| magnitude per frame, zero-padded to the time-window length so
        the same FrameBackBone architecture can ingest both domains."""
        mag = torch.fft.rfft(frames_time, dim=-1).abs()        # (B, F, W//2+1)
        pad = frames_time.shape[-1] - mag.shape[-1]
        if pad > 0:
            mag = F.pad(mag, (0, pad))                          # (B, F, W)
        return mag

    # ------------------------------------------------------------------
    # Forward (TF-C 3-loss + reconstruction)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.5):
        # 1. Build time / freq views.
        frames_t = self._frames_time(x)                         # (B, F, W)
        frames_f = self._frames_freq(frames_t)                  # (B, F, W)
        feat_t = self.frame_backbone(frames_t)                  # (B, F, D)
        feat_f = self.frame_backbone_freq(frames_f)             # (B, F, D)

        # 2. Two random-masked passes per view -> v1 (with reconstruction) and v2.
        latent_t1, pred_t1, mask_t1 = self.autoencoder(feat_t, mask_ratio)
        latent_t2 = self.autoencoder.forward_encoder(feat_t, mask_ratio)[0]
        latent_f1, pred_f1, mask_f1 = self.autoencoder(feat_f, mask_ratio)
        latent_f2 = self.autoencoder.forward_encoder(feat_f, mask_ratio)[0]

        # 3. MAE reconstruction loss on both domains.
        recon_loss = self.forward_mae_loss(feat_t, pred_t1, mask_t1) \
                   + self.forward_mae_loss(feat_f, pred_f1, mask_f1)

        # 4. CLS tokens.
        o_t1, o_t2 = latent_t1[:, 0, :], latent_t2[:, 0, :]
        o_f1, o_f2 = latent_f1[:, 0, :], latent_f2[:, 0, :]

        # 5. L_T: time-domain SimCLR between the two masked views.
        l_t = self.contrastive_loss(self.projector_time(o_t1),
                                    self.projector_time(o_t2))

        # 6. L_F: frequency-domain SimCLR between the two masked views.
        l_f = self.contrastive_loss(self.projector_freq(o_f1),
                                    self.projector_freq(o_f2))

        # 7. L_TF: cross-domain alignment (time CLS vs freq CLS, same instance).
        l_tf = self.contrastive_loss(self.projector_tfc(o_t1),
                                     self.projector_tfc(o_f1))

        return recon_loss, l_t, l_f, l_tf

    # ------------------------------------------------------------------
    # Inference helpers (used by Phase-2 / probes)
    # ------------------------------------------------------------------
    def forward_latent(self, x: torch.Tensor, global_tokens=False):
        """Time-domain encoder output for downstream / Phase-2 transfer.
        The frequency path is Phase-1 only — Phase-2 (PhysioME) inherits
        ``frame_backbone`` and ``autoencoder`` and discards ``frame_backbone_freq``
        and the three projectors."""
        x = self.make_frame(x)
        x = self.frame_backbone(x)
        latent = self.autoencoder.forward_encoder(x, mask_ratio=0)[0]
        if global_tokens:
            return latent
        return latent[:, :1, :].squeeze()

    def forward_mae_loss(self,
                         real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        if self.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.e-6) ** .5

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1)
        return frame


class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, input_size: int, num_patches: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int):
        super().__init__()
        self.patch_embed = nn.Linear(input_size, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.embed_dim = encoder_embed_dim
        self.encoder_depths = encoder_depths
        self.mlp_ratio = 4.

        self.input_size = (num_patches, encoder_embed_dim)
        self.patch_size = (1, encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w

        # MAE Encoder — uni2ts-derived TransformerEncoder (GQA + GLU FFN + RMSNorm + RoPE).
        # final norm is folded into the encoder; we expose ``encoder_norm`` as an alias for
        # backwards-compat with checkpoints/utilities that look for it by name.
        self.encoder = _build_rope_encoder(
            d_model=encoder_embed_dim, num_heads=encoder_heads, num_layers=encoder_depths,
        )
        self.encoder_norm = self.encoder.norm

        # MAE Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_block = nn.ModuleList([
            Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depths)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, input_size, bias=True)
        self.initialize_weights()

    def forward(self, x, mask_ratio=0.8):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return latent, pred, mask

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.5):
        # embed patches (RoPE applies position info inside attention; no additive pos_embed here)
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # prepend cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer encoder (RoPE inside attention; final RMSNorm folded in)
        x = self.encoder(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x[:, 1:, :])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.decoder_block:
            x = block(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape  # batch, length, dim
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def initialize_weights(self):
        # Decoder still uses absolute sin-cos pos embedding (length is fixed = num_patches).
        # Encoder side relies on RoPE inside attention, so no encoder pos_embed init needed.
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1],
                                                             (self.grid_h, self.grid_w),
                                                             cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class NeuroNetEncoder(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int):
        super().__init__()
        self.mlp_ratio = 4.0
        self.fs, self.second = fs, second
        self.time_window, self.time_step = time_window, time_step
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_heads = encoder_heads
        self.encoder_depths = encoder_depths

        _, self.num_patches, self.frame_size = self.make_frame(torch.randn(1, self.fs * self.second)).shape
        self.grid_h = int(self.num_patches // 1)
        self.grid_w = int(self.encoder_embed_dim // self.encoder_embed_dim)

        self.frame_backbone = FrameBackBone(fs=fs, window=time_window)
        self.patch_embed = nn.Linear(self.frame_backbone.feature_num, encoder_embed_dim)
        # uni2ts-derived encoder: GQA + GLU FFN + RMSNorm + RoPE.
        # Variable input length is supported natively (no pos_embed table to interpolate).
        self.encoder = _build_rope_encoder(
            d_model=encoder_embed_dim, num_heads=encoder_heads, num_layers=encoder_depths,
        )
        self.encoder_norm = self.encoder.norm  # alias for ckpt-loading utilities
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        # Match the original "frozen at init" behavior — cls_token is loaded from
        # the Phase-1 NeuroNet checkpoint and not updated during Phase-2 training.
        self.cls_token.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.make_frame(x)
        x = self.frame_backbone(x)
        x = self.patch_embed(x)

        # Prepend cls token; positions are injected inside attention via RoPE.
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)
        return x

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1)
        return frame


def frame_size(fs, second, time_window, time_step):
    x = np.random.randn(1, fs * second)
    size = fs * second
    step = int(time_step * fs)
    window = int(time_window * fs)
    frame = []
    for i in range(0, size, step):
        start_idx, end_idx = i, i + window
        sample = x[..., start_idx: end_idx]
        if sample.shape[-1] == window:
            frame.append(sample)
    frame = np.stack(frame, axis=1)
    return frame.shape[1], frame.shape[2]


if __name__ == '__main__':
    m0 = NeuroNet(fs=100, second=60, time_window=3, time_step=3,
                  encoder_embed_dim=256, encoder_depths=4, encoder_heads=8,
                  decoder_embed_dim=128, decoder_depths=2, decoder_heads=4,
                  projection_hidden=[1024, 512])
    recon, l_t, l_f, l_tf = m0(torch.randn((4, 6000)), mask_ratio=0.5)
    print(f'recon={recon.item():.3f}  L_T={l_t.item():.3f}  '
          f'L_F={l_f.item():.3f}  L_TF={l_tf.item():.3f}')
