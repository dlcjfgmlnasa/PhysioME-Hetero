# -*- coding:utf-8 -*-
"""PhysioME with availability-aware loss decomposition and 3-state presence embedding.

Differences vs the originally-published PhysioME (arXiv:2510.11110) implementation:

1. **Bug fix in restoration encoder pass.** The original drop branch built the
   multimodal-encoder input from the *kept* modalities only, so the encoder
   output had ``num_kept * num_backbone_frames`` tokens. ``forward_restoration_decoder``
   then split that output by ``len(ids_restores) == num_total_modals``, which
   yielded ``split_size = num_kept * frames // num_total_modals`` — a meaningless
   slice of the kept-modal tokens. We fix this by inserting a learnable
   ``dropped_modality_token`` in the encoder input for every dropped modality so
   that the encoder output is always ``num_total_modals * frames`` tokens long.
   ``forward_decoder`` and ``forward_restoration_decoder`` are unchanged.

2. **Heterogeneous-availability training.** ``forward`` accepts arbitrary
   subsets of ``modal_names`` in ``data``. Inter-modal masked-recon and
   cross-contrastive losses are computed on whatever modalities are present in
   the batch. The synthetic-drop restoration loss is computed only when the
   batch is *complete* (all modalities real-present) — naturally-absent
   modalities never contribute to restoration, since no GT exists for them.

3. **Decoder gradient-flow fix.** The original ``forward_decoder`` and
   ``forward_restoration_decoder`` ended each per-modality block with
   ``reconstructed_patches.append(x.detach().contiguous())``, which silently
   blocked gradients from the inter-modal MAE loss and the restoration loss
   from ever reaching the decoder weights, ``mask_token``, and the new
   ``dropped_modality_token``. We replace ``.detach().contiguous()`` with just
   ``.contiguous()`` so that those losses train the decoders as the
   architecture intends. The MAE *target* (``real_tokens``) is detached
   explicitly in the caller, which is the actual intended target-stop-gradient.

3. **3-state presence embedding.** A ``nn.Embedding(3, encoder_embed_dim)`` is
   added to every modality's encoder tokens, encoding whether the modality is
   real-present (0), synthetically dropped (1), or naturally absent (2). The
   model can therefore behave differently in the three regimes.
"""
import random
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.loss import NTXentLoss
from models.transformer import (
    QueryKeyProjection, RMSNorm, RotaryProjection, TransformerEncoder,
)


PRESENCE_REAL: int = 0
PRESENCE_SYNTH_DROPPED: int = 1
PRESENCE_NATURALLY_ABSENT: int = 2
NUM_PRESENCE_STATES: int = 3


# Physiology-grouped decoder mapping.
#   cardiovascular : cardiac-driven pulsatile signals (~1 Hz cardiac cycle)
#   respiratory    : respiration-driven signals       (~0.2 Hz breath cycle)
# Within each group, the decoder *body* (transformer stack) is shared across
# modalities; only the small per-modal prediction head (Linear -> backbone_dim)
# remains modality-specific. Rationale:
#   * Param efficiency: 6 decoders -> 2 bodies + 6 heads (~3-4x param cut).
#   * Gradient sharing: rare modalities (CVP / CO2 / AWP) piggyback on
#     same-group abundant modalities' gradients instead of starving on their
#     own small share of the batch.
#   * Inductive bias: signals within a group share temporal scale and
#     morphology family, so a shared body learns reusable features.
# Override via PhysioME(..., modal_to_group=<dict>) for ablations.
DEFAULT_MODAL_TO_GROUP: Dict[str, str] = {
    'ABP': 'cardiovascular',
    'ECG': 'cardiovascular',
    'PPG': 'cardiovascular',
    'CVP': 'cardiovascular',
    'CO2': 'respiratory',
    'AWP': 'respiratory',
}


def _build_xfmr(d_model: int, num_heads: int, num_layers: int) -> TransformerEncoder:
    """uni2ts-derived TransformerEncoder used by every PhysioME stack
    (multimodal encoder, MAE decoder, restoration decoder).

    GQA + GLU FFN + RMSNorm + RoPE for time-axis position encoding.
    ``d_cond=0`` disables AdaLN — PhysioME has no conditioning vector.
    The final ``self.norm`` lives inside the encoder, so callers do not
    need to apply a separate norm afterwards.
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


class PhysioME(nn.Module):
    def __init__(self,
                 backbone_networks: Dict[str, nn.Module],
                 backbone_embed_dim: int, num_backbone_frames: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 decoder_recon_depths: int,
                 projection_hidden: List[int], temperature: float,
                 modal_to_group: Optional[Dict[str, str]] = None):
        super().__init__()
        self.modal_names = list(backbone_networks.keys())
        self.backbone_networks = nn.ModuleDict(backbone_networks)
        self.num_backbone_frames = num_backbone_frames
        self.backbone_embed_dim = backbone_embed_dim
        self.encoder_embed_dim, self.decoder_embed_dim = encoder_embed_dim, decoder_embed_dim

        # Resolve modal -> group mapping. Defaults to physiology grouping
        # (cardiovascular / respiratory). Every modal in ``backbone_networks``
        # must have a group; unknown modals fall back to a per-modal group
        # (no sharing) so custom modalities never silently collide.
        mapping = dict(modal_to_group) if modal_to_group is not None \
            else dict(DEFAULT_MODAL_TO_GROUP)
        self.modal_to_group: Dict[str, str] = {
            m: mapping.get(m, m) for m in self.modal_names
        }
        # Stable, sorted group list — order matters for ModuleDict iteration.
        self.groups: List[str] = sorted(set(self.modal_to_group.values()))

        self.input_size = (self.num_backbone_frames, self.encoder_embed_dim)
        self.patch_size = (1, self.encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w
        self.mlp_ratio = 4.

        # [Backbone Network]
        self.backbone_embedded = nn.ModuleDict({
            modal_name: nn.Sequential(
                nn.Linear(backbone_embed_dim, encoder_embed_dim),
                Rearrange('b t e -> b e t'),
                nn.BatchNorm1d(encoder_embed_dim),
                nn.ELU(),
                Rearrange('b e t -> b t e'),
                nn.Linear(encoder_embed_dim, encoder_embed_dim)
            )
            for modal_name in self.modal_names
        })
        self.modal_token_dict = nn.ParameterDict({
            modal_name: nn.Parameter(torch.zeros(1, num_backbone_frames, encoder_embed_dim))
            for modal_name in self.modal_names
        })

        # [Presence-state conditioning] — 3 states (real / synth-dropped / naturally-absent)
        self.presence_state_embed = nn.Embedding(NUM_PRESENCE_STATES, encoder_embed_dim)
        # Learnable token used in the encoder input slot of a synthetically-dropped modality.
        self.dropped_modality_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        # [MultiModal Encoder] -- RoPE handles position; no learned pos_embed.
        # ``multimodal_encoder.norm`` is the final RMSNorm inside the encoder.
        self.multimodal_encoder = _build_xfmr(
            d_model=encoder_embed_dim, num_heads=encoder_heads, num_layers=encoder_depths,
        )

        # [MultiModal Decoder] -- per-group RoPE encoder stack (body) shared
        # across modalities within the same physiology group, plus a per-modal
        # linear prediction head. The body learns the temporal/morphological
        # structure of the group; the head re-maps to each modal's specific
        # output distribution.
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.multimodal_decoder_body_dict = nn.ModuleDict({
            group: _build_xfmr(
                d_model=decoder_embed_dim, num_heads=decoder_heads, num_layers=decoder_depths,
            )
            for group in self.groups
        })
        self.multimodal_decoder_pred_dict = nn.ModuleDict({
            modal_name: nn.Linear(decoder_embed_dim, backbone_embed_dim, bias=True)
            for modal_name in self.modal_names
        })

        # [MultiModal Decoder - Restoration (for missing modality)]
        # Same grouping as the MAE decoder above.
        self.recon_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.multimodal_recon_body_dict = nn.ModuleDict({
            group: _build_xfmr(
                d_model=decoder_embed_dim, num_heads=decoder_heads, num_layers=decoder_recon_depths,
            )
            for group in self.groups
        })
        self.multimodal_recon_pred_dict = nn.ModuleDict({
            modal_name: nn.Linear(decoder_embed_dim, backbone_embed_dim, bias=True)
            for modal_name in self.modal_names
        })

        # [Contrastive Learning]
        self.backbone_projector_dict = nn.ModuleDict({
            modal_name: self.get_projection_layer([backbone_embed_dim] + projection_hidden)
            for modal_name in self.modal_names
        })
        self.fusion_projector = self.get_projection_layer([encoder_embed_dim] + projection_hidden)
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.initialize_weights()

    @staticmethod
    def get_projection_layer(projection_hidden):
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            projectors.append(nn.Linear(h1, h2))
            if i != len(projection_hidden) - 2:
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
        return nn.Sequential(*projectors)

    def initialize_weights(self):
        # No absolute pos_embed tables — RoPE injects position info inside attention.
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.dropped_modality_token, std=.02)
        torch.nn.init.normal_(self.presence_state_embed.weight, std=.02)
        for model_name, modal_token in self.modal_token_dict.items():
            self.modal_token_dict[model_name] = torch.nn.init.normal_(modal_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_presence_state(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build a presence-state tensor when none is supplied: modalities present
        in ``data`` get state 0 (real-present); the rest get state 2 (naturally-absent).
        Identical across the batch."""
        device = next(iter(data.values())).device
        batch_size = next(iter(data.values())).shape[0]
        state = torch.full(
            (batch_size, len(self.modal_names)),
            PRESENCE_NATURALLY_ABSENT,
            dtype=torch.long, device=device,
        )
        for m in data.keys():
            if m not in self.modal_names:
                continue
            mi = self.modal_names.index(m)
            state[:, mi] = PRESENCE_REAL
        return state

    # ------------------------------------------------------------------
    # Encoder / decoder paths
    # ------------------------------------------------------------------
    def forward_encoder(self, data: Dict[str, torch.Tensor],
                        presence_state: Optional[torch.Tensor] = None,
                        mask_ratio: float = 0.8):
        if presence_state is None:
            presence_state = self._default_presence_state(data)

        total_x = []
        unimodal_token_dict = {}
        mask_dict = {}
        ids_restore_dict = {}

        for unimodal_name, unimodal_x in data.items():
            modal_idx = self.modal_names.index(unimodal_name)
            encoder_out = self.backbone_networks[unimodal_name](unimodal_x).detach().contiguous()
            encoder_emb = self.backbone_embedded[unimodal_name](encoder_out).detach().contiguous()

            x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name]
            state_emb = self.presence_state_embed(presence_state[:, modal_idx])
            x = x + state_emb.unsqueeze(1)

            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            unimodal_token_dict[unimodal_name] = encoder_out
            mask_dict[unimodal_name] = mask
            ids_restore_dict[unimodal_name] = ids_restore
            total_x.append(x)

        x = torch.cat(total_x, dim=1).contiguous()
        x = self.multimodal_encoder(x)  # final RMSNorm folded inside

        return (x, mask_dict, ids_restore_dict), unimodal_token_dict

    def forward_decoder(self, data, ids_restores):
        present_modals = list(ids_restores.keys())
        if not present_modals:
            return None

        split_size = data.shape[1] // len(present_modals) if present_modals else 0
        reconstructed_patches = []

        for i, modal_name in enumerate(present_modals):
            start, end = i * split_size, (i + 1) * split_size
            x = data[:, start:end, :].contiguous()
            x = self.decoder_embed(x).contiguous()

            ids_restore = ids_restores[modal_name]
            num_missing = ids_restore.shape[1] - x.shape[1]
            if num_missing > 0:
                mask_tokens = self.mask_token.repeat(x.shape[0], num_missing, 1)
                x = torch.cat([x, mask_tokens], dim=1).contiguous()

            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            group = self.modal_to_group[modal_name]
            x = self.multimodal_decoder_body_dict[group](x)  # final RMSNorm inside
            x = self.multimodal_decoder_pred_dict[modal_name](x)
            reconstructed_patches.append(x.contiguous())

        x = torch.cat(reconstructed_patches, dim=1) if reconstructed_patches else None
        return x

    def forward_restoration_decoder(self, data, ids_restores):
        present_modals = list(ids_restores.keys())

        split_size = data.shape[1] // len(present_modals) if present_modals else 0
        reconstructed_patches = []

        for i, modal_name in enumerate(present_modals):
            start, end = i * split_size, (i + 1) * split_size
            x = data[:, start:end, :].contiguous()
            x = self.recon_embed(x).contiguous()

            ids_restore = ids_restores[modal_name]
            num_missing = ids_restore.shape[1] - x.shape[1]
            if num_missing > 0:
                mask_tokens = self.mask_token.repeat(x.shape[0], num_missing, 1)
                x = torch.cat([x, mask_tokens], dim=1).contiguous()

            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            group = self.modal_to_group[modal_name]
            x = self.multimodal_recon_body_dict[group](x)  # final RMSNorm inside
            x = self.multimodal_recon_pred_dict[modal_name](x)
            reconstructed_patches.append(x.contiguous())

        x = torch.cat(reconstructed_patches, dim=1) if reconstructed_patches else None
        return x

    # ------------------------------------------------------------------
    # Synthetic-drop restoration (the bug-fixed version)
    # ------------------------------------------------------------------
    def _synthetic_drop_restoration(self, data: Dict[str, torch.Tensor],
                                    presence_state: torch.Tensor,
                                    mask_ratio: float,
                                    restoration_only_on_complete: bool = True) -> torch.Tensor:
        """Restoration loss with availability-aware masking.

        When ``restoration_only_on_complete`` is True (default; v1 recipe), this
        returns zero on any batch that is not fully real-present.

        When False (ablation A3), restoration runs on any batch with at least
        two real-present modalities: one or more are synthetically dropped, and
        the loss is computed only on the synth-dropped positions. Naturally-
        absent modalities occupy decoder-side slots via ``dropped_modality_token``
        but contribute zero loss because no ground truth exists for them.

        Gradient design (preserves the original ``no_grad`` stop on the multimodal
        encoder for the restoration path):
          * The multimodal encoder runs under ``no_grad``, on the kept-modal
            slots only — exactly as in the original PhysioME.
          * The encoder output is then padded with the learnable
            ``dropped_modality_token`` for every synth-dropped or
            naturally-absent slot, producing a tensor of shape
            ``[B, num_modals * num_backbone_frames, encoder_dim]``. This pad-up
            is what lets ``forward_restoration_decoder`` use a correct
            ``split_size = num_backbone_frames`` per modality (the bug fix vs the
            original implementation).
          * ``forward_restoration_decoder`` receives gradients normally, so the
            restoration decoder *and* ``dropped_modality_token`` learn from this
            loss; the encoder weights do not.
        """
        device = next(iter(data.values())).device
        n_modals = len(self.modal_names)
        if n_modals < 2:
            return torch.zeros((), device=device)

        # Canonical per-modal availability — bucket sampling guarantees identical
        # presence across the batch. Sample 0 is canonical.
        canonical = presence_state[0]
        real_present = [m for i, m in enumerate(self.modal_names)
                        if int(canonical[i]) == PRESENCE_REAL]

        if restoration_only_on_complete and len(real_present) != n_modals:
            return torch.zeros((), device=device)
        if len(real_present) < 2:
            return torch.zeros((), device=device)

        max_drop = len(real_present) - 1
        num_drop = random.randint(1, max_drop)
        drop_modals = set(random.sample(real_present, num_drop))
        keep_modals = [m for m in real_present if m not in drop_modals]
        # Modalities that are simply absent from the recording (no GT).
        absent_modals = {m for m in self.modal_names
                         if int(canonical[self.modal_names.index(m)])
                         == PRESENCE_NATURALLY_ABSENT}

        synth_state = presence_state.clone()
        for m in drop_modals:
            mi = self.modal_names.index(m)
            synth_state[:, mi] = PRESENCE_SYNTH_DROPPED

        b = next(iter(data.values())).shape[0]
        L = self.num_backbone_frames

        unimodal_token_dict: Dict[str, torch.Tensor] = {}

        # ---- Encoder pass under no_grad: kept modalities only ----
        with torch.no_grad():
            encoder_inputs: List[torch.Tensor] = []
            for m in keep_modals:
                mi = self.modal_names.index(m)
                encoder_out = self.backbone_networks[m](data[m]).detach().contiguous()
                encoder_emb = self.backbone_embedded[m](encoder_out).detach().contiguous()
                x_m = (encoder_emb[:, 1:, :]
                       + self.modal_token_dict[m]
                       + self.presence_state_embed(synth_state[:, mi]).unsqueeze(1))
                encoder_inputs.append(x_m)
                unimodal_token_dict[m] = encoder_out
            # GT for synth-dropped modals (data[m] still available)
            for m in drop_modals:
                encoder_out = self.backbone_networks[m](data[m]).detach().contiguous()
                unimodal_token_dict[m] = encoder_out

            x = torch.cat(encoder_inputs, dim=1).contiguous()
            x = self.multimodal_encoder(x)  # [B, num_keep * L, encoder_dim]

        # ---- Pad encoder output up to [B, num_modals * L, encoder_dim] ----
        padded_blocks: List[torch.Tensor] = []
        ids_restore_dict: Dict[str, torch.Tensor] = {}
        recon_masks: List[torch.Tensor] = []
        keep_cursor = 0
        for m in self.modal_names:
            if m in keep_modals:
                block = x[:, keep_cursor * L:(keep_cursor + 1) * L, :]
                keep_cursor += 1
                mask_m = torch.zeros((b, L), device=device)
            elif m in drop_modals:
                block = self.dropped_modality_token.expand(b, L, -1)
                mask_m = torch.ones((b, L), device=device)
            else:  # naturally absent
                block = self.dropped_modality_token.expand(b, L, -1)
                mask_m = torch.zeros((b, L), device=device)
            padded_blocks.append(block)
            ids_restore_dict[m] = torch.arange(L, device=device).unsqueeze(0).repeat(b, 1)
            recon_masks.append(mask_m)

        x_padded = torch.cat(padded_blocks, dim=1).contiguous()

        real_tokens_list: List[torch.Tensor] = []
        for m in self.modal_names:
            if m in unimodal_token_dict:
                # detach: target should not propagate gradient
                real_tokens_list.append(unimodal_token_dict[m][:, 1:, :].detach())
            else:
                real_tokens_list.append(
                    torch.zeros((b, L, self.backbone_embed_dim), device=device)
                )
        real_tokens = torch.cat(real_tokens_list, dim=1)

        recon_tokens = self.forward_restoration_decoder(x_padded, ids_restore_dict)
        recon_masks_tensor = torch.cat(recon_masks, dim=-1)
        return self.forward_mae_loss(real_tokens, recon_tokens, recon_masks_tensor)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data: Dict[str, torch.Tensor],
                presence_state: Optional[torch.Tensor] = None,
                mask_ratio: float = 0.8,
                simulate_drop: bool = True,
                restoration_only_on_complete: bool = True):
        device = next(iter(data.values())).device

        if presence_state is None:
            presence_state = self._default_presence_state(data)

        # 1. Inter-modal masked reconstruction — present modalities only
        (fusion_tokens, masks, ids_restores), unimodal_token_dict = self.forward_encoder(
            data, presence_state=presence_state, mask_ratio=mask_ratio,
        )
        # Targets are detached so the model can't trivially set pred == target.
        real_tokens = torch.cat(
            [t[:, 1:, :].detach() for t in unimodal_token_dict.values()], dim=1,
        )
        pred_tokens = self.forward_decoder(fusion_tokens, ids_restores)
        mask_tokens = torch.cat([m for m in masks.values()], dim=-1)
        inter_recon_loss = self.forward_mae_loss(real_tokens, pred_tokens, mask_tokens)

        # 2. Cross-contrastive — fusion <-> each present unimodal
        cross_losses, cross_accs = [], []
        fusion_token = torch.mean(fusion_tokens, dim=1)
        o1 = self.fusion_projector(fusion_token)
        for unimodal_name, unimodal_tokens in unimodal_token_dict.items():
            unimodal_token = torch.mean(unimodal_tokens, dim=1)
            o2 = self.backbone_projector_dict[unimodal_name](unimodal_token)
            contra_loss, (labels, logits) = self.contrastive_loss(o1, o2)
            cross_losses.append(contra_loss)
            cross_accs.append(
                torch.mean((torch.argmax(logits, dim=-1) == labels).to(torch.float32))
            )
        cross_contra_loss = torch.stack(cross_losses, dim=-1).mean()
        cross_contra_acc = torch.stack(cross_accs, dim=-1).mean()

        # 3. Synthetic-drop restoration — gated by restoration_only_on_complete
        if simulate_drop:
            missing_recon_loss = self._synthetic_drop_restoration(
                data, presence_state, mask_ratio,
                restoration_only_on_complete=restoration_only_on_complete,
            )
        else:
            missing_recon_loss = torch.zeros((), device=device)

        return inter_recon_loss, missing_recon_loss, cross_contra_loss, cross_contra_acc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference_missing_modality(self, data: Dict[str, torch.Tensor],
                                   presence_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return fusion-token vector for inputs with possibly absent modalities.
        Internally restores naturally-absent modalities via the restoration
        decoder before re-encoding all modalities together.

        ``data`` should contain only the modalities that are actually present.
        Missing modalities are inferred from ``set(self.modal_names) - set(data)``.
        """
        device = next(iter(data.values())).device
        b = next(iter(data.values())).shape[0]
        L = self.num_backbone_frames

        present_modals = [m for m in self.modal_names if m in data]
        missing_modals = [m for m in self.modal_names if m not in data]

        if presence_state is None:
            presence_state = torch.full(
                (b, len(self.modal_names)), PRESENCE_NATURALLY_ABSENT,
                dtype=torch.long, device=device,
            )
            for m in present_modals:
                mi = self.modal_names.index(m)
                presence_state[:, mi] = PRESENCE_REAL

        # Fast path: nothing absent → pure encoder pass
        if not missing_modals:
            (fusion_tokens, _, _), _ = self.forward_encoder(
                data, presence_state=presence_state, mask_ratio=0.0,
            )
            return torch.mean(fusion_tokens, dim=1)

        # 1. Encoder pass on present modals only (matches restoration design).
        encoder_inputs = []
        for m in present_modals:
            mi = self.modal_names.index(m)
            encoder_out = self.backbone_networks[m](data[m]).detach().contiguous()
            encoder_emb = self.backbone_embedded[m](encoder_out).detach().contiguous()
            x_m = (encoder_emb[:, 1:, :]
                   + self.modal_token_dict[m]
                   + self.presence_state_embed(presence_state[:, mi]).unsqueeze(1))
            encoder_inputs.append(x_m)

        x = torch.cat(encoder_inputs, dim=1).contiguous()
        x = self.multimodal_encoder(x)

        # 2. Pad encoder output with dropped_modality_token for absent slots,
        #    then run restoration decoder.
        padded_blocks = []
        ids_restore_dict: Dict[str, torch.Tensor] = {}
        present_cursor = 0
        for m in self.modal_names:
            if m in present_modals:
                block = x[:, present_cursor * L:(present_cursor + 1) * L, :]
                present_cursor += 1
            else:
                block = self.dropped_modality_token.expand(b, L, -1)
            padded_blocks.append(block)
            ids_restore_dict[m] = torch.arange(L, device=device).unsqueeze(0).repeat(b, 1)
        x_padded = torch.cat(padded_blocks, dim=1).contiguous()

        recon = self.forward_restoration_decoder(x_padded, ids_restore_dict)
        recon_chunks = recon.chunk(chunks=len(self.modal_names), dim=1)

        # 3. Re-encode using restored signals for missing modalities.
        encoder_inputs2 = []
        for i, m in enumerate(self.modal_names):
            mi = self.modal_names.index(m)
            if m in present_modals:
                encoder_out = self.backbone_networks[m](data[m]).detach().contiguous()
                encoder_emb = self.backbone_embedded[m](encoder_out).detach().contiguous()
                x_m = encoder_emb[:, 1:, :]
            else:
                encoder_out = recon_chunks[i].detach().contiguous()
                encoder_emb = self.backbone_embedded[m](encoder_out).detach().contiguous()
                x_m = encoder_emb
            x_m = (x_m
                   + self.modal_token_dict[m]
                   + self.presence_state_embed(presence_state[:, mi]).unsqueeze(1))
            encoder_inputs2.append(x_m)

        x = torch.cat(encoder_inputs2, dim=1).contiguous()
        x = self.multimodal_encoder(x)
        return torch.mean(x, dim=1)

    def inference(self, data: Dict[str, torch.Tensor],
                  presence_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the multimodal encoder over only the modalities present in ``data``
        (no restoration). Returns the fusion-token vector."""
        valid_data = {m: v for m, v in data.items()
                      if m in self.modal_names and v is not None}
        if not valid_data:
            raise ValueError('No valid modalities provided in the input data')

        if presence_state is None:
            presence_state = self._default_presence_state(valid_data)

        (fusion_tokens, _, _), _ = self.forward_encoder(
            valid_data, presence_state=presence_state, mask_ratio=0.0,
        )
        return torch.mean(fusion_tokens, dim=1)

    # ------------------------------------------------------------------
    # Loss / utilities
    # ------------------------------------------------------------------
    @staticmethod
    def forward_mae_loss(real: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        denom = mask.sum().clamp_min(1.0)
        loss = (loss * mask).sum() / denom
        return loss

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


if __name__ == '__main__':
    # Sanity check with a tiny synthetic backbone — verifies the bug-fixed
    # restoration path on a complete batch and a hetero-bucket batch.
    class TinyBackbone(nn.Module):
        def __init__(self, num_frames: int, embed_dim: int):
            super().__init__()
            self.num_frames = num_frames
            self.embed_dim = embed_dim
            self.proj = nn.Linear(3000, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Returns [B, num_frames+1, embed_dim] (cls + frames).
            b = x.shape[0]
            cls = torch.zeros(b, 1, self.embed_dim, device=x.device)
            frames = self.proj(x).unsqueeze(1).expand(b, self.num_frames, self.embed_dim).contiguous()
            return torch.cat([cls, frames], dim=1)

    embed_dim = 64
    num_frames = 5
    # Exercise the 2-group decoder: ABP/ECG/PPG/CVP → cardiovascular,
    # CO2/AWP → respiratory. Two distinct decoder bodies must be built.
    net = PhysioME(
        backbone_networks={
            'ABP': TinyBackbone(num_frames, embed_dim),
            'ECG': TinyBackbone(num_frames, embed_dim),
            'PPG': TinyBackbone(num_frames, embed_dim),
            'CVP': TinyBackbone(num_frames, embed_dim),
            'CO2': TinyBackbone(num_frames, embed_dim),
            'AWP': TinyBackbone(num_frames, embed_dim),
        },
        backbone_embed_dim=embed_dim, num_backbone_frames=num_frames,
        encoder_embed_dim=128, encoder_heads=4, encoder_depths=2,
        decoder_embed_dim=64, decoder_heads=4, decoder_depths=2,
        decoder_recon_depths=2,
        projection_hidden=[256, 128], temperature=0.1,
    )
    print(f'[groups] {net.groups}')
    print(f'[decoder body keys] {list(net.multimodal_decoder_body_dict.keys())}')
    assert list(net.multimodal_decoder_body_dict.keys()) == ['cardiovascular',
                                                              'respiratory']

    B = 4
    complete = {m: torch.randn(B, 3000) for m in net.modal_names}
    inter, miss, contra, acc = net(complete, mask_ratio=0.5)
    print(f'[complete] inter={inter.item():.3f}  miss={miss.item():.3f}  '
          f'contra={contra.item():.3f}  acc={acc.item():.3f}')

    hetero = {
        'ECG': torch.randn(B, 3000),
        'PPG': torch.randn(B, 3000),
    }
    inter, miss, contra, acc = net(hetero, mask_ratio=0.5)
    print(f'[hetero ] inter={inter.item():.3f}  miss={miss.item():.3f}  '
          f'contra={contra.item():.3f}  acc={acc.item():.3f}')

    out = net.inference_missing_modality({'ECG': torch.randn(B, 3000)})
    print(f'[inference_missing_modality output] shape={tuple(out.shape)}')
