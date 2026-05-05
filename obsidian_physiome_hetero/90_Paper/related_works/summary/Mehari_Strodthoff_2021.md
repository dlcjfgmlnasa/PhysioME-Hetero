---
arxiv: 2103.12676
title: "Self-supervised representation learning from 12-lead ECG data"
authors: Mehari, Strodthoff
venue: Computers in Biology and Medicine 141 (2022) 105114
year: 2022
domain: ECG self-supervised learning
tags: [related-works, ECG, self-supervised, contrastive, CPC]
---

# Self-supervised representation learning from 12-lead ECG data

## TL;DR
12-lead ECG 에 컴퓨터 비전·음성의 대표 SSL 방법 (SimCLR, BYOL, SwAV, **CPC**) 을 체계적으로 비교한 첫 systematic benchmark. **Contrastive Predictive Coding (CPC)** 변형이 linear probe 성능 supervised 대비 -0.5%, fine-tuning 시 +1% 로 가장 우수.

## Domain & Task
- 12-lead ECG (PTB-XL 등)
- Multi-label arrhythmia / cardiology classification
- Linear probe / fine-tuning / label efficiency / robustness 평가

## Method
- 4 가지 SSL 방법 비교: SimCLR, BYOL, SwAV (instance discrimination 계열) + CPC (latent forecasting 계열)
- Encoder : 1D-CNN (ResNet variant) — domain-standard
- 각 방법별 ECG-specific augmentation 적용
- CPC variant 가 ECG 의 시간적 구조에 가장 잘 맞음

## Datasets
- PTB-XL (대규모 12-lead ECG 데이터셋, 약 21K records)
- Chapman, ICBEB 등 외부 검증

## Key Results
- **CPC variant** : linear evaluation 0.5% below supervised, fine-tuning +1% above supervised
- **Label efficiency** : 적은 labeled data 에서 더 큰 이득
- **Physiological noise robustness** : SSL pretrained 가 supervised 대비 noise 에 강건
- GitHub : https://github.com/tmehari/ecg-selfsupervised

## Relation to 본 연구
- ECG-domain SSL 의 systematic benchmark — Related Work 의 자연스러운 reference 후보
- **CPC 의 latent forecasting** 은 본 연구의 **Block Next Prediction** 과 conceptual 유사 (다음 시점 예측 신호)
- 단점 : ECG-only, 1D-CNN encoder, multi-modal 미지원

## Limitations / 차이
- ECG 단일 modality
- Patch-based transformer 가 아닌 1D-CNN
- Cross-modal 결합 미고려

## Citation 위치 후보
- §Related Work / Self-supervised biosignal pretraining
- §Block Next Prediction (CPC 가 forecasting-based SSL 의 baseline)
