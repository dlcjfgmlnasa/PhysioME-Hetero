---
arxiv: 2312.05409
title: "Large-scale Training of Foundation Models for Wearable Biosignals"
authors: Abbaspourazad, Elachqar, Miller, Emrani, Nallasamy, Shapiro
venue: ICLR 2024
year: 2024
domain: Wearable PPG + ECG foundation model
tags: [related-works, foundation-model, PPG, ECG, wearable, contrastive, large-scale]
---

# Apple Wearable Foundation Model (PPG + ECG)

## TL;DR
Apple Heart and Movement Study 의 **141K 참가자, 3 년치 데이터** 로 학습한 PPG · ECG 사전학습 foundation model. Wearable consumer 디바이스 환경에 특화된 첫 large-scale biosignal foundation model.

## Domain & Task
- Wearable 디바이스 PPG 와 ECG (smartwatch 등)
- Self-supervised pretraining → 인구통계, 건강 상태 추론, 다양한 downstream

## Method
- **Self-supervised contrastive learning**
- 핵심 design choices:
  - **Participant-level positive pair selection** (같은 사용자의 다른 시간 segment)
  - Stochastic augmentation
  - Regularized contrastive loss
  - Momentum encoder (BYOL/MoCo style)
- 단일 framework 가 PPG, ECG 양쪽에 일반화

## Datasets
- Apple Heart and Movement Study (~141K participants, ~3 years)
- 거대 규모 — 임상 데이터셋 (PTB-XL ~21K) 대비 7배 이상

## Key Results
- 사전학습 표현이 인구통계 (나이, 성별), 건강 조건 (BMI, 부정맥 등) 정보를 자연스럽게 인코딩
- Linear probe 만으로 다양한 downstream 에서 강한 성능
- 사용자 수준 label efficiency 입증

## Relation to 본 연구
- **Wearable consumer scale 의 biosignal FM** — 본 연구 (ICU/OR clinical scale) 와 보완적
- Participant-level positive pair = 본 연구의 **same sample + different time** 와 유사한 원리
- 단, 본 연구는 **modality 간 alignment** (cross-modal contrastive) 까지 확장 — Apple 모델은 단일 modality 내 contrastive

## Limitations / 차이
- Wearable 환경 (외래 / 일상) — ICU/OR 의 invasive catheter 신호 미지원
- ECG 와 PPG 두 modality 만, ABP/CVP/PAP/ICP 같은 침습적 신호 없음
- Cross-modal 학습 메커니즘 명시 부재 (modality 별 separate encoder 가능성)

## Citation 위치 후보
- §Related Work / Foundation models for biosignals (wearable scale 의 대표)
- §Datasets / Pretraining 규모 비교 (141K participants 대 본 연구 3.5K+ ICU 환자)
- §Cross-Modal Contrastive Learning (participant-level positive 의 선조)
