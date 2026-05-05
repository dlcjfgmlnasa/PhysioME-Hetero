---
arxiv: 2005.13249
title: "CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients"
authors: Kiyasseh, Zhu, Clifton
venue: ICML 2021
year: 2021
domain: ECG self-supervised learning
tags: [related-works, ECG, contrastive, self-supervised]
---

# CLOCS — Contrastive Learning of Cardiac Signals Across Space, Time, and Patients

## TL;DR
ECG 의 **세 가지 invariance** (spatial = lead 간, temporal = 시간 segment 간, patient = 환자 간) 를 contrastive 학습 신호로 활용해 SimCLR/BYOL 보다 강한 ECG 표현을 학습. 25% labeled data 만으로 SOTA 동등 성능.

## Domain & Task
- 12-lead ECG 분류 (arrhythmia, AFIB 등)
- Self-supervised pretraining → linear probe / fine-tuning

## Method
- **세 contrastive variant** :
  - **CMSC** (Contrastive Multi-Segment Coding) : 같은 환자·같은 lead 의 다른 시간 segment 를 positive
  - **CMLC** (Contrastive Multi-Lead Coding) : 같은 환자·같은 시간 segment 의 다른 lead 를 positive
  - **CMSMLC** : 둘을 결합 — 같은 환자 + 다른 lead + 다른 시간 segment 를 positive
- Encoder : 1D-CNN (ResNet-style)
- Loss : NT-Xent (SimCLR 식 InfoNCE)

## Datasets
- PhysioNet 2017 (single-lead ECG)
- Chapman ECG (12-lead)
- PhysioNet 2020 / Cardiology
- 100K+ ECG, 다양한 임상 코호트

## Key Results
- BYOL, SimCLR 대비 downstream linear probe AUROC +2~5%
- 25% labeled training data 로 fully-supervised 100% 동등 성능
- Patient-specific embedding 의 클러스터링이 임상 의미를 반영 (interpretability)

## Relation to 본 연구
- **Cross-Modal Contrastive Learning** 의 직접적 conceptual 선조 — 본 연구는 *"같은 sample (환자) + 같은 time + 다른 modality"* 를 positive 로 사용 → CLOCS 의 CMLC variant 의 modality-수준 일반화로 볼 수 있음
- CLOCS 는 ECG 단일 modality 내 lead invariance, 본 연구는 6 종 cardiovascular modality 간 invariance 로 확장
- Related Work 에서 **"contrastive pretraining of physiological signals"** 항목의 주요 인용 후보

## Limitations / 차이
- ECG 단일 modality 만 다룸 — 본 연구의 multi-modal scope 와 차이
- Patch-based transformer 가 아닌 1D-CNN encoder — architecture 측면에서 다른 계열

## Citation 위치 후보 (paper 내)
- §Cross-Modal Contrastive Learning : positive pair 정의의 inspiration 출처
- §Related Work / Self-supervised biosignal pretraining
