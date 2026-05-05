---
arxiv: 2402.03885
title: "MOMENT: A Family of Open Time-series Foundation Models"
authors: Goswami, Szafer, Choudhry, Cai, Li, Dubrawski
venue: ICML 2024 (CMU)
year: 2024
domain: General time-series foundation model
tags: [related-works, foundation-model, time-series, masked-reconstruction, open-weight, family-of-models]
---

# MOMENT — Family of Open Time-series Foundation Models (CMU)

## TL;DR
**Open-source family of time-series foundation models** (40M / 125M / 385M). **T5-encoder + masked reconstruction** pretraining. Time Series Pile (1.13 B time points) 사전학습. Forecasting, classification, anomaly detection, imputation 의 4 task family 통합.

## Domain & Task
- 일반 multivariate time-series
- 4 task family : forecasting, classification, anomaly detection, imputation
- Zero-shot / linear probe / fine-tuning 평가

## Method
- **T5 encoder** backbone (다양한 크기 family)
- **Masked reconstruction** SSL (BERT-style on continuous values)
- **Time Series Pile** : 1.13 B time points across multiple domains 자체 구축
- 모든 task 에 동일 모델 적용

## Datasets
- **Time Series Pile** (자체 collection, 1.13 B time points)

## Key Results
- 4 task 모두에서 task-specific SOTA 동등 이상
- Linear probe 만으로 강한 성능
- Open-weight, open-data — 본 연구 reproducibility 의 reference

## Relation to 본 연구
- **정규화 도메인의 한계 사례** — 본 연구 §Conditional Layer Normalization 에서 **MOMENT 의 *"정규화 후 절대 스케일 정보 손실"*** 이슈를 motivation 으로 인용 ([[reference#MOMENT]])
- 일반 시계열 FM 의 대표 baseline — 본 연구의 cardiovascular FM 과 직접 비교 대상
- 4 task family 통합 paradigm = 본 연구의 9 downstream task 통합과 같은 철학
- 단, biosignal-specific 임상 component (Loc/Scale conditioning, Binary Bias, Cross-Modal) 모두 부재

## Limitations / 차이
- 일반 시계열 — 임상 cardiovascular 특화 없음
- 정규화 도메인 의존 — 본 연구가 LSCNorm 으로 극복하는 한계
- Cross-modal coupling 부재

## Citation 위치 후보 (이미 본 연구 본문에 인용됨)
- §Conditional Layer Normalization (정규화 도메인 한계 motivation)
- §Related Work / General time-series foundation models
- §Experiments / 일반 시계열 FM 비교 baseline
