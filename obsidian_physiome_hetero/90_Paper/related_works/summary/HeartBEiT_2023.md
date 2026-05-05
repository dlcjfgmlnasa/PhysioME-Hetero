---
doi: 10.1038/s41746-023-00840-9
title: "A foundational vision transformer improves diagnostic performance for electrocardiograms"
authors: Vaid, Jiang, Sawant, et al.
venue: npj Digital Medicine 6, 108 (2023)
year: 2023
domain: ECG vision transformer foundation model
tags: [related-works, ECG, foundation-model, vision-transformer, masked-image-modeling, npj-DM]
---

# HeartBEiT — Foundational Vision Transformer for ECG (npj Digital Medicine 2023)

## TL;DR
**ECG waveform 을 image 로 변환** 후 **masked image modeling (BEiT-style)** 으로 사전학습한 vision transformer. **8.5 M ECGs** 사전학습. CNN 대비 hypertrophic cardiomyopathy, low LVEF, STEMI 진단에서 우수.

## Domain & Task
- 12-lead ECG → image representation
- HCM (hypertrophic cardiomyopathy), low LVEF, STEMI 등 임상 진단

## Method
- **ECG → image** 변환 (waveform 을 grid plot 으로 시각화)
- **BEiT-style Masked Image Modeling** : 일부 patch token mask → tokenizer 의 visual token 예측
- Vision Transformer (ViT) backbone
- 8.5 M ECG image 사전학습

## Datasets
- 8.5 M ECGs (Mount Sinai 등)

## Key Results
- 표준 CNN architecture 대비 진단 성능 향상
- 특히 small/medium label regime 에서 큰 이득

## Relation to 본 연구
- **npj Digital Medicine publication** — 본 연구의 target venue. paper format · narrative · evaluation 표준 reference
- **ECG-as-image** paradigm = 본 연구의 1D waveform patch 처리와 다른 접근. 두 방식의 trade-off 가 discussion 거리
- 단일 ECG modality 에 한정 — 본 연구의 6 종 cardiovascular FM 과 scope 차이

## Limitations / 차이
- ECG 단일 modality
- Image 변환 시 정밀한 수치 정보 손실 가능
- Patient-level aggregation 메커니즘 없음

## Citation 위치 후보
- §Related Work / ECG foundation models (npj DM precedent)
- §Discussion / Image-based vs Waveform-based ECG FM 비교
- npj DM submission 시 venue-specific reference 로 거의 필수
