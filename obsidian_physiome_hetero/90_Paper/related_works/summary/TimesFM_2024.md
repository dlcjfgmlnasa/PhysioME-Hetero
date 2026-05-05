---
arxiv: 2310.10688
title: "A Decoder-only Foundation Model for Time-series Forecasting"
authors: Das, Kong, Sen, Zhou
venue: ICML 2024 (Google Research)
year: 2024
domain: General time-series forecasting
tags: [related-works, foundation-model, time-series, decoder-only, residual-mlp, block-prediction]
---

# TimesFM — Decoder-only Foundation Model for Time-series Forecasting (Google)

## TL;DR
**Decoder-only Transformer** 기반 시계열 forecasting foundation model. **Residual MLP patch tokenizer** (1-hidden-layer MLP + skip connection) 와 **non-autoregressive block prediction** (output patch length > input patch length) 도입. 200 M parameters, 100 B+ time points 사전학습.

## Domain & Task
- 일반 시계열 forecasting (energy, weather, traffic 등)
- Zero-shot forecasting

## Method
- **Decoder-only** Transformer (GPT-style)
- **Residual MLP patch tokenizer** : raw time-series patch (32 values + 32 mask bits) → 1280-dim token. **본 연구의 Residual MLP Patch Projection 의 직접 출처**
- **Non-autoregressive block prediction** : input patch = 32, output patch = 128. 한 번의 forward 로 더 긴 미래 예측. **본 연구의 Block Next Prediction 의 직접 출처**
- Pretraining : 100 B+ real time points + 합성 데이터

## Datasets
- 100 B+ real time points (Google internal time-series + public archive)

## Key Results
- Zero-shot forecasting 에서 task-specific SOTA 동등 또는 우위
- 200 M parameters scaling 효과 입증
- HuggingFace open weights

## Relation to 본 연구 (architecture 의 직접 prior)
- **Residual MLP Patch Projection** (§Patch Encoder) — TimesFM 의 patch tokenizer 채택
- **Block Next Prediction** (§Objectives) — TimesFM 의 non-AR block prediction 채택
- 두 component 모두 본 연구의 reference list 에 이미 등록됨 ([[reference#TimesFM]])

## Limitations / 차이
- 일반 시계열 — biosignal-specific 임상 결합 부재
- Forecasting 단일 task (masked reconstruction, contrastive 미지원)
- Cross-modal coupling 부재

## Citation 위치 후보 (이미 본 연구 본문에 인용됨)
- §Patch Encoder / Residual MLP Patch Projection
- §Block Next Prediction
- §Related Work / Time-series foundation models
