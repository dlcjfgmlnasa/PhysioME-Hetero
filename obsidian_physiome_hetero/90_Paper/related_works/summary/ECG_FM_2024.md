---
arxiv: 2408.05178
title: "ECG-FM: An Open Electrocardiogram Foundation Model"
authors: McKeen, Masood, Toma, Rubin, Wang
venue: arXiv 2024 (UHN Toronto)
year: 2024
domain: ECG foundation model (open-source)
tags: [related-works, ECG, foundation-model, transformer, contrastive, generative, open-weight]
---

# ECG-FM — Open ECG Foundation Model (Toronto)

## TL;DR
**1.5 M ECG** 로 사전학습한 **open-weight** ECG foundation model. **Hybrid contrastive + generative** SSL strategy. AFib AUROC 0.996, LVEF 예측 등 임상 task SOTA. 모델·코드·benchmark 모두 공개.

## Domain & Task
- 12-lead ECG
- Pretraining → 다중 임상 task (AFib detection, LVEF prediction, ECG interpretation)

## Method
- **Transformer** based encoder
- **Hybrid SSL** : contrastive (instance discrimination) + generative (masked reconstruction)
- 1.5 M ECGs 사전학습

## Datasets
- Pretraining : 1.5 M ECGs (UHN Toronto institutional)
- Benchmark : MIMIC-IV-ECG

## Key Results
- AFib detection : AUROC **0.996**
- LVEF prediction 등 다양한 임상 task SOTA
- Small-to-medium label regime 에서 task-specific 모델 능가
- Latent space + saliency map 분석 제공

## Relation to 본 연구
- **ECG foundation model 의 직접적 baseline** — 본 연구의 cardiovascular FM 과 가장 가까운 비교 대상 중 하나
- Hybrid SSL (contrastive + generative) = 본 연구의 4 손실 결합과 같은 철학
- Open-weight + benchmark 공개 = 본 연구의 reproducibility 정책 reference (npj DM 합격 확률 ↑ 의 open science 측면)
- 단, ECG-only 단일 modality — 본 연구는 6 종 cardiovascular signal 통합

## Limitations / 차이
- ECG 단일 modality
- Cross-modal 학습 메커니즘 부재
- Patient-level aggregation 메커니즘 명시 부재 (window-level focus)

## Citation 위치 후보
- §Related Work / ECG foundation models (대표 baseline)
- §Experiments / Arrhythmia Classification baseline 후보
- Discussion / open-weight release 정책의 reference
