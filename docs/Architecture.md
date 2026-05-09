# Architecture

## Overview

PhysioME-Hetero is composed of:

```
   ABP ─► [Backbone_ABP] ─► [Adapter_ABP] ─┐
   ECG ─► [Backbone_ECG] ─► [Adapter_ECG] ─┤
   PPG ─► [Backbone_PPG] ─► [Adapter_PPG] ─┼──► [Multimodal Encoder] ──► fusion tokens
   CVP ─► [Backbone_CVP] ─► [Adapter_CVP] ─┤            (1, shared)            │
   CO2 ─► [Backbone_CO2] ─► [Adapter_CO2] ─┤                                   │
   AWP ─► [Backbone_AWP] ─► [Adapter_AWP] ─┘                                   ▼
                                       ┌─► [MAE Decoder × 6]   ─► inter-modal recon loss
                                       └─► [Recon Decoder × 6] ─► restoration loss
                                                                  (synth-drop / missing modal)
```

Implementation: `models/physiome/model.py` (`PhysioME` class).

## Module breakdown (4-modal example; v2 = 6-modal)

| Module | Shared / per-modality | Count (v2) | Role |
|---|---|---|---|
| `backbone_networks` | per-modality | 6 | Phase-1 NeuroNet encoders, frozen + LoRA |
| `backbone_embedded` | per-modality | 6 | backbone_dim → encoder_dim adapter |
| `modal_token_dict` | per-modality | 6 | learnable modality identifier |
| `presence_state_embed` | shared | 1 | 3-state embedding (real / synth-dropped / absent) |
| `dropped_modality_token` | shared | 1 | placeholder token for empty slots |
| `multimodal_encoder` | **shared** | 1 | self-attention across all modal tokens |
| `decoder_embed` | shared | 1 | encoder_dim → decoder_dim |
| `recon_embed` | shared | 1 | encoder_dim → recon_dim |
| `mask_token` | shared | 1 | MAE mask placeholder |
| `multimodal_decoder_dict` (MAE) | per-modality | 6 | inter-modal masked recon |
| `multimodal_decoder_pred_dict` | per-modality | 6 | MAE pred head (Linear) |
| `multimodal_recon_dict` (Restoration) | per-modality | 6 | missing-modality token recon |
| `multimodal_recon_pred_dict` | per-modality | 6 | restoration pred head (Linear) |
| `backbone_projector_dict` | per-modality | 6 | unimodal contrastive projector |
| `fusion_projector` | shared | 1 | fusion-token contrastive projector |

**Architecture (type) is identical across modalities** — only weight
instances are split. The `decoder_*` yaml hyperparameters are scalars
shared across modalities; per-modality decoder hparams would require a
yaml dict + `__init__` rewire.

## Two decoder kinds

| | MAE Decoder | Restoration Decoder |
|---|---|---|
| dict attr | `multimodal_decoder_dict` | `multimodal_recon_dict` |
| depth (yaml) | `decoder_depths` (default 3) | `decoder_recon_depths` (default 8) |
| input | `decoder_embed(fusion_token)` + mask token | `recon_embed` + `dropped_modality_token` for empty slots |
| target | masked backbone tokens (intra-modality recon) | dropped/missing modality tokens (inter-modality recon) |
| loss | `inter_recon_loss` | `missing_recon_loss` |

Both stacks built from the same `_build_xfmr` factory:
GQA + GLU FFN + RMSNorm + RoPE (uni2ts-derived `TransformerEncoder`).

## 3-state presence embedding

`nn.Embedding(NUM_PRESENCE_STATES=3, encoder_embed_dim)`:

| state | int | meaning |
|---|---|---|
| `PRESENCE_REAL` | 0 | modality is genuinely measured for this subject |
| `PRESENCE_SYNTH_DROPPED` | 1 | measured, but masked for restoration training |
| `PRESENCE_NATURALLY_ABSENT` | 2 | not measured (no GT exists) |

Added to every modality token before they enter the multimodal encoder.
Lets the encoder *know which slots to trust* and treat the three regimes
differently (vs the original PhysioME which had no such signal).

## `dropped_modality_token`

Single learnable `[1, 1, encoder_embed_dim]` parameter that fills any
encoder/decoder slot whose modality is dropped or absent. Acts like a
BERT `[MASK]` token but at the modality level. Its learnable nature lets
the restoration decoder use it as a *query* asking *"if this slot were
present, what would it look like?"*.

Critical role: keeps encoder output shape `[B, N_total × L, D]` even when
only `N_kept < N_total` modalities were actually present in the encoder.
Without this padding, `forward_restoration_decoder` splits the encoder
output by `N_total` and gets meaningless per-modal slices — the original
PhysioME's restoration bug.

## Heterogeneous-availability training

`forward(data, presence_state, mask_ratio, restoration_only_on_complete)`:

1. **Inter-modal masked recon** on whatever modalities are in `data`.
2. **Cross-contrastive** (NTXent) between fusion token and each present
   unimodal token — only on present modalities.
3. **Synth-drop restoration**:
   - Default (`restoration_only_on_complete=True`): only when the batch
     bucket has all `N` modalities real-present.
   - A3 ablation (`False`): on any batch with ≥2 real-present modalities;
     synth-drops one or more, naturally-absent slots occupy decoder slots
     via `dropped_modality_token` but contribute zero loss.

`BucketBatchSampler` ensures every batch is *uniform in availability
pattern*, so the loss decisions above can be made once per batch.

## Bug fixes vs the original PhysioME (arXiv:2510.11110)

1. **Restoration encoder split bug**: original encoder output was
   `[B, N_kept × L, D]` but `forward_restoration_decoder` split by
   `N_total` — every per-modal decoder received a meaningless slice.
   Fixed by padding to `[B, N_total × L, D]` with `dropped_modality_token`.
2. **Decoder gradient stop bug**: `forward_decoder` /
   `forward_restoration_decoder` ended each per-modal block with
   `.detach().contiguous()`, silently blocking inter-recon and
   restoration losses from updating decoder weights / `mask_token` /
   `dropped_modality_token`. Only the contrastive head actually trained
   the model. Fixed by removing the `.detach()`; targets are explicitly
   detached in the caller (the genuine intent of "stop-gradient on target").
3. (cosmetic) `train.py` referenced `SVC` without importing it — never
   actually called in working flows but cleaned up.

These are why the v1 A1 baseline must re-run with our bug-fixed code,
not the published numbers.

## Phase-1 backbones (NeuroNet, TF-C SSL)

Each backbone is independently pretrained on a single modality with the
TF-C objective (Zhang et al. NeurIPS 2022):

```
forward(x, mask_ratio) → (recon_loss, L_T, L_F, L_TF)
```

* `recon_loss`: MAE-style time-domain reconstruction.
* `L_T`: time-domain contrastive (NT-Xent over time-view pairs).
* `L_F`: frequency-domain contrastive (NT-Xent over freq-view pairs from
  `frame_backbone_freq` consuming |FFT|-magnitude per frame).
* `L_TF`: cross-domain alignment (time CLS vs freq CLS of the same
  instance).

Architecture: `frame_backbone` (ResNet1D) → patch_embed → shared
`autoencoder` (MAE encoder + cls_token + decoder) + three projectors.

Phase-2 transfers `frame_backbone`, `autoencoder.patch_embed`,
`autoencoder.encoder`, and `autoencoder.cls_token` per modality.
Frequency path is Phase-1 only.

## Phase-2 LoRA

After Phase-1 backbones are loaded into the multimodal model, every
`Linear` named `out_proj` (the GQA output projection) is wrapped by
`models/transformer/lora.LoRALinear` (rank-r residual, gaussian init A /
zero init B, rsLoRA scaling `alpha / sqrt(r)`). All other backbone
parameters are frozen. Only LoRA adapters + multimodal encoder + decoders
+ embeddings + projectors are trainable in Phase-2.

State-dict keys are clean: `encoder.layers.<i>.self_attn.out_proj.{base.weight,
lora_A, lora_B}` — no `base_model.model.<...>.base_layer.weight` prefix
that peft would produce.
