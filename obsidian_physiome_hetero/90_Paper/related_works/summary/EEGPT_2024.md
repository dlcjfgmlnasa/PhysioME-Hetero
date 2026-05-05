---
venue: NeurIPS 2024
title: "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals"
authors: Wang, Liu, He, Xu, Ma, Li
year: 2024
domain: EEG foundation model
tags: [related-works, EEG, foundation-model, pretrained-transformer, BCI, dual-self-supervised]
---

# EEGPT — Pretrained Transformer for Universal EEG (NeurIPS 2024)

## TL;DR
**10 M parameter EEG foundation model**. Mask-based **dual self-supervised** (masked reconstruction + spatio-temporal alignment) pretraining. Hierarchical structure 로 spatial / temporal 분리 처리.

## Domain & Task
- EEG (BCI, sleep, neurology 등)
- 다양한 downstream task linear-probing

## Method
- **Mask-based dual SSL** :
  - Masked reconstruction : raw EEG patch 일부 mask → 복원
  - Spatio-temporal representation alignment : high-SNR semantic representation level alignment (pixel level 이 아닌 representation level masked target)
- **Hierarchical** : spatial pooling + temporal modeling 분리 → 효율 + flexibility
- 10 M parameters

## Datasets
- EEG-specific dataset 모음 (구체적 list 본문 확인)

## Key Results
- Linear-probing 으로 SOTA on 다양한 EEG task
- 효율적 모델 크기 (10M) 로 큰 성능
- Open-source : https://github.com/BINE022/EEGPT

## Relation to 본 연구
- **Dual SSL** (masked + alignment) = 본 연구의 4 손실 결합과 같은 multi-objective 철학
- Hierarchical spatial/temporal 분리 = 본 연구의 Dual Embedding + temporal patch 의 다른 형태
- EEG-only — 본 연구의 cardiovascular scope 와 분리
- LaBraM 과 함께 EEG FM 의 양대 reference

## Limitations / 차이
- EEG 단일 modality
- Cross-modal coupling 미지원
- Cardiovascular signal 미지원

## Citation 위치 후보
- §Related Work / EEG foundation models (LaBraM 과 짝)
- §Objectives Function (Dual SSL paradigm 의 사례)
- §Discussion / 향후 EEG 확장 시 baseline
