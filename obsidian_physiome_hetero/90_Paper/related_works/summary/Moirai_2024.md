---
arxiv: 2402.02592
title: "Unified Training of Universal Time Series Forecasting Transformers"
authors: Woo, Liu, Kumar, Xiong, Savarese, Sahoo
venue: ICML 2024 (Salesforce)
year: 2024
domain: General multivariate time-series forecasting
tags: [related-works, foundation-model, time-series, forecasting, large-scale]
---

# Moirai — Universal Time Series Forecasting Transformer (Salesforce)

## TL;DR
**27B+ observations across 9 domains (LOTSA archive)** 로 학습한 universal time-series forecasting foundation model. Cross-frequency learning, variable-length, heterogeneous distribution 의 세 challenge 를 동시 해결.

## Domain & Task
- 일반 multivariate time-series **forecasting** (energy, weather, finance, traffic, healthcare 등 9 도메인)
- Zero-shot forecasting 평가

## Method
- **Masked encoder Transformer** (BERT-style, decoder 없음)
- **LOTSA dataset** : 9 domain × 27B+ observations 의 거대 archive 자체 구축
- Multi-frequency 처리 : 다양한 sampling rate 통합
- Variable-length input : flexible context window
- Multi-variate : channel mixing 지원

## Datasets
- **LOTSA (Large-scale Open Time Series Archive)** : 27B+ obs, 9 domains
- Healthcare 도메인도 포함하지만 일반 시계열 위주

## Key Results
- Zero-shot forecasting 에서 task-specific SOTA 와 동등 또는 우월
- Scale (27B obs) 의 효과 입증

## Relation to 본 연구
- **General time-series FM 의 대표** — 본 연구의 cardiovascular 특화 FM 과 직접 비교 baseline
- 본 연구의 zero-shot Waveform Forecasting 평가에서 Moirai 가 baseline 후보
- Architecture 측면에서 masked encoder + multi-frequency 처리 → 본 연구의 patch-based encoder + LSCNorm 과 비교 가능
- 단점 : healthcare-specific 설계 부재 (특히 cross-modal 임상 결합 미고려)

## Limitations / 차이
- 일반 시계열 — biosignal 의 임상적 문맥 (loc/scale, modality identity, cross-modal coupling) 미고려
- Forecasting 단일 task focus — masked reconstruction 등 다양한 SSL objective 결합 부재
- 데이터 규모 비교 : 27B obs (general) vs 본 연구 ~100B+ tokens (cardiovascular-specific)

## Citation 위치 후보
- §Related Work / General time-series foundation models
- §Block Next Prediction (forecasting FM 의 대표 baseline)
- §Experiments / Waveform Forecasting baseline 비교
