---
arxiv: 2410.20542
title: "PaPaGei: Open Foundation Models for Optical Physiological Signals"
authors: Pillai, Spathis, Kawsar, Malekzadeh
venue: ICLR 2025 (Nokia Bell Labs)
year: 2025
domain: PPG foundation model (open-source)
tags: [related-works, PPG, foundation-model, optical, open-weight, domain-aware]
---

# PaPaGei — Open Foundation Models for PPG (ICLR 2025)

## TL;DR
**PPG (photoplethysmography) 단일 modality 의 첫 open foundation model**. 57,000+ hours / 20 M unlabeled segments 로 학습. 20 task × 10 dataset 평가에서 contrastive 기반 baseline 대비 +6.3% (분류) / +2.9% (회귀) 개선. **70× 작은 모델 규모로 competitive 경쟁 모델 능가**.

## Domain & Task
- PPG (광학 박동) 단일 modality
- 20 task : 심혈관 건강, 수면 장애, 임신 모니터링, wellbeing 평가

## Method
- **Domain-aware representation learning** : PPG 의 형태학적 특성 (박동 윤곽, 변동) 을 명시적으로 활용
- 일반 contrastive 가 아닌 PPG-morphology aware
- Skin tone bias benchmark 포함 (fairness)

## Datasets
- 57,000+ hours, 20 M PPG segments (공개 데이터 합산)

## Key Results
- 20 task 평균 분류 +6.3%, 회귀 +2.9%
- 70× 작은 모델로 task-specific 모델 능가 (효율성)
- Skin tone 별 bias 분석 (fairness)
- Open-weight, open-data

## Relation to 본 연구
- **PPG foundation model 의 대표** — 본 연구의 PPG channel 처리 baseline
- Domain-aware morphology representation = 본 연구의 **patch-level 보편적 어휘 학습** narrative 와 같은 철학
- Open-weight + fairness 분석 = 본 연구의 reproducibility 정책 reference
- 단, PPG-only — 본 연구는 6 종 cardiovascular 통합

## Limitations / 차이
- PPG 단일 modality
- Cross-modal 결합 미고려
- Foundation 규모는 70× 작음 (효율성 강점이지만 scale 측면 약점)

## Citation 위치 후보
- §Related Work / PPG / Optical biosignal foundation models
- §Experiments / Hypotension·Sepsis 등 PPG 활용 task baseline 후보
- Discussion / open-weight + fairness 정책 reference
