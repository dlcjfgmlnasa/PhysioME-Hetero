---
arxiv: 2405.18765
title: "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"
authors: Jiang, Zhao, Lu
venue: ICLR 2024
year: 2024
domain: EEG foundation model
tags: [related-works, EEG, foundation-model, BCI, vector-quantization, channel-patch]
---

# LaBraM — Large Brain Model for EEG (ICLR 2024)

## TL;DR
~2,500 시간 EEG (~20 datasets) 로 학습한 **EEG foundation model**. 채널 수·sample 길이가 데이터셋마다 다른 EEG 의 unification challenge 를 **channel patch + vector-quantized neural spectrum** 으로 해결.

## Domain & Task
- EEG (BCI 도메인, 다양한 task)
- Abnormal detection, event classification, emotion recognition, gait prediction 등 다중 downstream

## Method
- **Channel patch tokenization** : EEG 를 channel × time patch 로 분할 → 다양한 channel count 호환
- **Vector-quantized neural spectrum prediction** : raw EEG patch → semantic neural code 로 tokenize
- **Pretraining objective** : masked neural code prediction (BERT-style on quantized tokens)
- Architecture : Neural Transformer

## Datasets
- ~20 EEG datasets, 총 ~2,500 hours
- Channel 수 다양 (가변 montage 호환)

## Key Results
- 4 가지 downstream (이상감지·이벤트·감정·보행) 모두에서 강한 transfer
- Cross-dataset 일반화 입증

## Relation to 본 연구
- **EEG foundation model 의 대표** — 본 연구가 EEG 를 입력에서 제외하고 cardiovascular 에 집중한 결정과 대조 사례
- Channel patch + vector quantization 접근 = 본 연구의 **modality + spatial Dual Embedding** 과 다른 paradigm
- 본 연구의 미래 확장 (EEG 추가) 시 주요 baseline

## Limitations / 차이
- EEG-only — 다른 modality (ECG, ABP 등) 미지원
- Vector quantization 의 정밀 파형 복원 한계 (Chronos 와 같은 단점)
- Cross-modal 학습 메커니즘 부재

## Citation 위치 후보
- §Related Work / EEG foundation models
- §Future Work / EEG 등 multi-system 확장의 prior art
- §Discussion / scope 결정 (EEG 제외 결정의 대조)
