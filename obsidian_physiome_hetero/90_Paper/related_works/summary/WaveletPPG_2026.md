---
arxiv: 2601.12215
title: "Wavelet-Driven Masked Multiscale Reconstruction for PPG Foundation Models"
authors: Thukral, Tanade, Lee, Lee, Zhou, Chun, Gwak, Nathan, Rahman, Zhu, Morshed, Venkatraman, Desai
venue: arXiv 2026 (Samsung Research)
year: 2026
domain: PPG foundation model (wearable)
tags: [related-works, PPG, foundation-model, wavelet, multi-scale, masked-reconstruction, wearable]
---

# Wavelet-Driven Masked Multiscale Reconstruction for PPG (Samsung)

## TL;DR
PPG 의 **wavelet 다중 스케일 분해** 의 coefficient 를 mask + reconstruct 하는 self-supervised pretraining. ~17 M PPG segments / ~32K smartwatch 사용자. 19 task 중 17 task 에서 baseline 동등 이상.

## Domain & Task
- PPG (smartwatch wearable)
- 19 health-related downstream

## Method
- **Wavelet decomposition** : PPG → 시간-주파수 hierarchical scale
- **Masked Multiscale Reconstruction (MMR)** : random 으로 wavelet coefficient mask → 복원
- Transformer encoder + wavelet-based feature
- 17 M unlabeled segments

## Datasets
- ~17 M PPG segments, ~32,000 smartwatch users
- 다양한 health task (cardiovascular, sleep, fitness 등)

## Key Results
- 19 downstream 중 17 task 에서 baseline 동등 이상
- Spectral 다중 스케일 활용이 일반 time-domain masked reconstruction 능가

## Relation to 본 연구
- **Frequency-domain masked pretraining** 의 PPG 적용 — 본 연구의 Multi-Resolution STFT Loss 와 같은 철학
- 단, wavelet 기반 vs STFT 기반 방법론 차이
- Wearable scale (32K user) — 본 연구 ICU clinical scale 과 보완적

## Limitations / 차이
- PPG 단일 modality
- Cross-modal 결합 미지원
- Wavelet 의 specific 한 frequency band 의존성 (PPG-specific)

## Citation 위치 후보
- §Related Work / Frequency-domain biosignal SSL
- §Loss Formulations / Multi-Resolution STFT Loss 의 spectral SSL 사례
