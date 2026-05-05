---
tags: [hub, architecture]
---

# 02 Architecture — Hub

> 모델 컴포넌트와 핵심 설계 결정 7종.

## 모델 스택 개요

```
Phase-1 (per-modality SSL)
    NeuroNet + TF-C
        time backbone   ─┐
        freq backbone   ─┼→ shared autoencoder (RoPE TransformerEncoder)
                         │      ↓
                         │   3 projectors (time / freq / tfc)
                         │      ↓
                         │   L_T + L_F + L_TF + recon

Phase-2 (multimodal SSL)
    PhysioME hetero
        unimodal backbones (LoRA out_proj) — frozen base
            ↓
        +3-state presence embedding +dropped_modality_token
            ↓
        multimodal encoder (RoPE TransformerEncoder)
            ↓
        per-modal MAE decoder ─┐
                                │→ inter-recon + miss-recon + cross-contra
        per-modal restoration  ─┘   (availability-aware)

Downstream
    PhysioMEClassifier (frozen encoder + fc head)
        forward via inference_missing_modality (auto-restoration)
        7-subset enumeration
```

## 핵심 설계 결정 (narrative 순서)

이 순서대로 읽으면 *왜* 이 모델 모양인지 이해 가능:

1. [[Decision_Hetero_Bucket]] — 학습 데이터부터 real heterogeneous 분포 (paper의 핵심)
2. [[Decision_Presence_Embedding]] — 3가지 modality 상태를 모델이 구분
3. [[Decision_BFM_Transformer_Port]] — timm Block → uni2ts-derived TransformerEncoder (RMSNorm + GQA + GLU FFN + RoPE)
4. [[Decision_RoPE_Everywhere]] — 학습형 pos_embed 모두 제거, 임의 길이 네이티브 지원
5. [[Decision_TFC_over_SimCLR]] — Phase-1을 SimCLR에서 Time-Frequency Consistency로 교체
6. [[Decision_FreqProj_Linear]] — TF-C freq view의 zero-padding hack을 학습된 linear lift으로 교체
7. [[Decision_HandRolled_LoRA]] — peft 의존성 제거, `models/transformer/lora.py`로 직접 구현

## 컴포넌트 위치 (코드 ↔ wiki)

| 컴포넌트 | 코드 | 결정 노트 |
|---|---|---|
| RMSNorm | `models/transformer/norm.py` | [[Decision_BFM_Transformer_Port]] |
| GQA | `models/transformer/attention.py::GroupedQueryAttention` | [[Decision_BFM_Transformer_Port]] |
| GLU FFN | `models/transformer/ffn.py::GatedLinearUnitFeedForward` | [[Decision_BFM_Transformer_Port]] |
| RoPE | `models/transformer/position/attn_projection.py::QueryKeyProjection + RotaryProjection` | [[Decision_RoPE_Everywhere]] |
| TransformerEncoder | `models/transformer/transformer.py` | [[Decision_BFM_Transformer_Port]] |
| LoRA | `models/transformer/lora.py::LoRALinear, apply_lora` | [[Decision_HandRolled_LoRA]] |
| NeuroNet | `models/dp_neuronet/model.py::NeuroNet` | [[Decision_TFC_over_SimCLR]], [[Decision_FreqProj_Linear]] |
| PhysioME | `models/physiome/model.py::PhysioME` | [[Decision_Hetero_Bucket]], [[Decision_Presence_Embedding]] |
| PhysioMEClassifier | `downstream/model.py` | — |

## Bug-fixes 기록 (원본 PhysioME 대비)

원본 PhysioME 코드에서 발견되어 우리 refactor에서 수정된 버그 (A1 baseline 비교 시 중요):

1. **`forward_restoration_decoder`의 split_size 불일치** — 원본은 encoder가 *kept modal* 만 인코딩한 결과를 `len(num_total_modals)` 으로 split해 의미없는 슬라이스를 각 per-modal decoder에 전달. **Fix**: `dropped_modality_token`으로 dropped/absent 슬롯을 채워 항상 `num_total_modals * num_backbone_frames` 길이 보장.

2. **decoder forward 끝의 `.detach().contiguous()`** — inter-modal MAE loss와 missing-recon loss가 decoder 가중치 / `mask_token` / `dropped_modality_token` 학습에 기여하지 못함. **Fix**: `.contiguous()`만 유지, target은 caller에서 명시적 `.detach()`.

3. **`SVC` import 누락** — `pretrained/physiome/train.py::linear_probing`에서 `SVC()` 사용하지만 import 없음. **Fix**: `train_hetero.py`에 import 추가.

→ 자세한 컨텍스트: `C:\Users\SNUH_VitalLab_LEGION\.claude\projects\C--Projects-PhysioME\memory\project_physiome_state.md`
