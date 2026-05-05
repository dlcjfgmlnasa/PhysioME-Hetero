---
tags: [decision, architecture]
date: 2026-05-04
status: locked
---

# Decision: Phase-1 SSL을 SimCLR → TF-C로 교체

## 결정

NeuroNet의 두 NT-Xent loss (random mask view-pair + augmented input pair) 를 제거하고, **Time-Frequency Consistency** (Zhang et al., NeurIPS 2022)로 교체한다.

3가지 contrastive loss + 1 reconstruction loss:

| Loss | 양 view | 의미 |
|---|---|---|
| `L_T` | time-domain mask₁ vs mask₂ | 시간 도메인 view 일관성 |
| `L_F` | freq-domain mask₁ vs mask₂ | 주파수 도메인 view 일관성 |
| `L_TF` | time-CLS vs freq-CLS (same instance) | **cross-domain alignment** ← TF-C 핵심 |
| `recon` | MAE on time + MAE on freq | reconstruction supervision |

## Why

**SimCLR 기반의 두 가지 문제**:

1. **False-negative 압박이 biosignal에 부적합**. Sleep-EEG augmentation (segment crop, permutation)으로 만든 view가 같은 환자/같은 수술타입의 다른 segment와 contrastive에서 강하게 push apart 됨 → 임상적으로 *비슷해야 할* 신호들이 표현 공간에서 멀어짐.

2. **Augmentation의 invariance 가정이 약함**. Crop/permutation이 의미를 보존한다는 가정이 sleep stage에서는 ok였지만 ABP/ECG/PPG 같은 hemodynamic signal에는 의문 — short crop은 cardiac cycle을 깨뜨리고, permutation은 시간 순서를 destroy함.

**TF-C의 장점 (biosignal-natural)**:

- Cardiac periodicity / HR / HRV / 호흡 modulation은 **주파수 도메인에서 더 명시적**. ABP/ECG/PPG에 freq-view supervision은 자연스러움.
- Cross-domain alignment (`L_TF`)가 view augmentation 없이 supervisory signal 제공 — augmentation의 의미 invariance 가정 불필요.
- 같은 instance의 time-CLS와 freq-CLS를 align하라는 task는 **trivial이 아님** (서로 다른 표현이지만 같은 신호의 다른 측면) — 풍부한 학습 신호.

## Architecture

```
input x (B, T)
    │
    ├── _frames_time(x)       (B, F, W)    raw waveform per frame
    │       └── frame_backbone (time)
    │
    └── _frames_freq(frames_t)             |FFT| magnitude per frame
            └── freq_proj (W//2+1 → W)     learned linear lift
                    └── frame_backbone_freq (freq)

Both → SHARED autoencoder (patch_embed + transformer encoder + cls_token + decoder)
    ↓
projector_time / projector_freq / projector_tfc (separate per TF-C paper)
    ↓
NT-Xent for L_T, L_F, L_TF
```

핵심: **shared autoencoder**. Phase-2 PhysioME가 transfer하는 universal representation.
- `frame_backbone` + `frame_backbone_freq` 는 도메인별 (다른 BN statistics)
- `autoencoder.encoder` + `cls_token` + decoder는 공유

## What was removed

- `pretrained/dp_neuronet/augmentation.py::DataAugmentationNeuroNet` 사용 안 됨 (보관용으로만 남김)
- 기존 NeuroNet의 view-pair contrastive (random mask 두 번 + augmented input pair) 제거

## 검증 (smoke)

`experiments/smoke_test_neuronet_tfc.py` 통과 (2026-05-05):
- 4 loss 모두 finite, positive at init: recon=0.88 / L_T=0.90 / L_F=1.11 / L_TF=4.26
- 7개 module 모두 gradient routing 정상 (projector_time/freq/tfc, frame backbones, encoder, decoder)
- 30-step 고정 batch 학습으로 total loss **5.74 → 0.14** 수렴

## 후속 결정

- [[Decision_FreqProj_Linear]] — `_frames_freq`의 zero-padding hack 제거 (2026-05-05)

## 원본

- 결정 메모: `memory/project_physiome_state.md` "Phase-1 SSL switched from SimCLR to TF-C"
- 코드: `models/dp_neuronet/model.py::NeuroNet`
- 논문: Zhang et al., "Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency", NeurIPS 2022
- Method 노트: [[../90_Paper/PhysioME-Hetero/11. Phase-1 TF-C]]
