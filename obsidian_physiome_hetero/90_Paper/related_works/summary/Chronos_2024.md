---
arxiv: 2403.07815
title: "Chronos: Learning the Language of Time Series"
authors: Ansari, Stella, Turkmen, Zhang, Mercado et al. (Amazon Science)
venue: arXiv 2024 (TMLR 2024)
year: 2024
domain: General time-series forecasting
tags: [related-works, foundation-model, time-series, language-model, tokenization, forecasting]
---

# Chronos — Learning the Language of Time Series (Amazon)

## TL;DR
시계열 값을 **scaling + quantization** 으로 fixed vocabulary token 으로 변환한 뒤, **T5 계열 LM 을 cross-entropy loss 로 사전학습**. 시계열을 *"다른 언어"* 로 보는 LM 적 접근. 20M ~ 710M parameter, 42 datasets + synthetic GP 데이터.

## Domain & Task
- 일반 시계열 forecasting (도메인 무관)
- Zero-shot forecasting 평가

## Method
- **Tokenization** : continuous value → bin (quantization) → token id
- **Architecture** : T5 family (encoder-decoder LM, 기존 pretrained 활용)
- **Pretraining loss** : Cross-entropy (categorical, language model 식)
- 모델 크기 : 20M / 60M / 200M / 710M
- 학습 데이터 : 42 public dataset + synthetic Gaussian Process samples

## Datasets
- 42 public time-series datasets (Monash, M-competitions 등)
- 합성 Gaussian Process 데이터로 데이터 augmentation

## Key Results
- In-distribution / out-of-distribution 양쪽에서 강한 zero-shot
- LM-style tokenization 이 continuous regression head 보다 일반화 우수

## Relation to 본 연구
- **시계열을 LM 으로 처리하는 대표 사례** — 본 연구는 continuous patch regression 채택 (Chronos 와 다른 paradigm)
- Patch token 의 *"보편적 어휘"* narrative (본 연구 §Curriculum Learning) 와 conceptual 유사
- 단, biosignal 의 임상적 정확성 (수치 스케일 보존) 측면에서 quantization 은 단점 — 본 연구가 이를 회피

## Limitations / 차이
- Biosignal-specific 설계 없음
- Quantization 이 정밀한 파형 복원에는 부적합 — 임상 deployment 어려움
- Multi-modal cross-modal 결합 미지원

## Citation 위치 후보
- §Related Work / Time-series foundation models (LM-style 접근의 대표)
- §Block Next Prediction (forecasting FM baseline)
- Discussion / *"왜 patch regression 을 채택했는가"* 의 대조 사례
