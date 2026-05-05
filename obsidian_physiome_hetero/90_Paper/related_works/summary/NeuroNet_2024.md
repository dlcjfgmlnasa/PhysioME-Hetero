---
arxiv: 2404.17585
title: "NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG"
authors: Lee, Kim, Han, Jung, Yoon, Kim
venue: arXiv 2024
year: 2024
domain: Single-channel EEG sleep staging
tags: [related-works, EEG, sleep, self-supervised, hybrid, contrastive, masked-prediction, mamba]
---

# NeuroNet — Hybrid SSL for Single-channel EEG Sleep Staging

## TL;DR
**Contrastive + Masked prediction** 의 hybrid SSL 을 단일 채널 EEG 의 sleep stage 분류에 적용. **Mamba** 기반 temporal context module 로 epoch 간 관계 모델링. 3 PSG dataset 에서 SOTA SSL 방법 능가.

## Domain & Task
- Single-channel EEG (single-lead, polysomnography)
- Sleep stage classification (5-class: W/N1/N2/N3/REM)

## Method
- **Hybrid SSL** : contrastive (instance discrimination) + masked prediction (BERT-style)
- **Encoder** : CNN-based feature extractor
- **Mamba-based temporal context module** : sleep epoch 간 long-range dependency 학습
- Two-stage : intra-epoch (contrastive + masked) + inter-epoch (Mamba)

## Datasets
- 3 PSG (Polysomnography) datasets (Sleep-EDF, SHHS 등 추정)

## Key Results
- 기존 EEG SSL 방법 능가
- Limited labeled data 에서 supervised 동등 성능

## Relation to 본 연구
- **Contrastive + Masked 의 hybrid** = 본 연구의 4 손실 결합 ($\mathcal{L}_{\mathrm{recon}} + \mathcal{L}_{\mathrm{next}} + \mathcal{L}_{\mathrm{cross}} + \mathcal{L}_{\mathrm{con}}$) 의 EEG-domain precedent
- Mamba 사용 — 본 연구는 Transformer 채택, architecture 측면 대조 사례
- Sleep staging 은 본 연구의 downstream 에 포함되지 않으나, *"hybrid SSL"* paradigm 의 인용 가치

## Limitations / 차이
- 단일 채널 EEG 만 — multi-modal 미지원
- Sleep staging 단일 task focus
- Foundation model 규모 사전학습 아님

## Citation 위치 후보
- §Related Work / Hybrid SSL biosignal pretraining
- §Objectives Function (다중 손실 결합의 baseline)
