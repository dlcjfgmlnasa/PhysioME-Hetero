---
tags: [decision, architecture]
date: 2026-05-04
status: locked
---

# Decision: RoPE everywhere (학습형 pos_embed 제거)

## 결정

NeuroNet 인코더 + PhysioME 멀티모달 인코더 + MAE 디코더 + restoration 디코더 **모두**에서 학습형 1D positional embedding 테이블을 제거하고, RoPE (`QueryKeyProjection` + `RotaryProjection`)만 사용한다.

## Why

원본 PhysioME는 두 종류 pos_embed를 사용:
- `multimodal_encoder_pos_embed: nn.Parameter(zeros, requires_grad=...)` — sincos init
- `multimodal_decoder_pos_embed: nn.Parameter(randn, requires_grad=...)` — sincos init

문제:
1. **고정 길이 가정** — 학습 시 `num_modals * num_frames` 길이로 init되는데, 멀티모달에서는 이 길이가 *modality 수에 비례*. testtime에 다른 부분집합을 넣으면 길이가 다름 → 매번 `interpolate_1d_pos_embed` 호출 필요.
2. **중복 표현** — RoPE도 position 정보를 attention QK projection에 주입함. 학습형 pos_embed와 RoPE 둘 다 있으면 redundant.
3. **NeuroNet의 BFM 포팅 부산물** — BFM의 TransformerEncoder는 RoPE를 attention 내부에서 처리하므로 외부에서 더 더할 이유가 없음.

## What changed

`models/physiome/model.py`:
- `multimodal_encoder_pos_embed` 파라미터 **제거**
- `multimodal_decoder_pos_embed` 파라미터 **제거**
- `multimodal_*_norm` LayerNorm들도 **제거** (BFM TransformerEncoder가 final norm을 fold)
- `initialize_weights()` 에서 `get_2d_sincos_pos_embed_flexible` 호출 제거 (PhysioME 측만 — NeuroNet의 MAE 디코더는 여전히 사용)

`models/utils.py`:
- `interpolate_1d_pos_embed` 함수 삭제 (더 이상 호출자 없음)
- 사용 안 하는 `import torch.nn.functional as f` 제거

## NeuroNet MAE decoder는 예외

`models/dp_neuronet/model.py::MaskedAutoEncoderViT`의 디코더는 여전히 timm `Block` + 학습형 `decoder_pos_embed`를 사용. RoPE 디코더로 교체 가능하지만:
- MAE는 정해진 길이로 동작 (Phase-1은 60s 윈도우 고정)
- 디코더는 Phase-2로 transfer되지 않음 (인코더만 transfer)

→ 우선순위 낮아 v1에서 그대로 유지.

## 임의 길이 native 지원 효과

7-subset inference에서 시퀀스 길이가 `1*frames`, `2*frames`, `3*frames` 로 변하지만 RoPE는 각 길이를 native로 처리. interpolation 호출 없이 `inference_missing_modality` 가 7-subset 모두에 same code path.

검증: `experiments/smoke_test_downstream.py` 7-subset 모두 forward 성공.

## 원본

- 결정 메모: `memory/project_physiome_state.md` "RoPE applied to PhysioME multimodal stack too"
- 코드: `models/physiome/model.py` (모든 forward_*), `models/transformer/position/`
