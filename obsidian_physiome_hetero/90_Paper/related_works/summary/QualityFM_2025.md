---
arxiv: 2509.06516
title: "QualityFM: a Multimodal Physiological Signal Foundation Model with Self-Distillation for Signal Quality Challenges in Critically Ill Patients"
authors: Guo, Chen, Ferrario
venue: arXiv 2025
year: 2025
domain: ICU/OR multi-modal physiological FM (PPG + ECG)
tags: [related-works, foundation-model, ICU, PPG, ECG, self-distillation, signal-quality, multi-modal]
---

# QualityFM — Multi-modal Physiological FM with Self-Distillation (ICU/OR)

## TL;DR
ICU·OR 환경의 **poor signal quality** (motion artifact, sensor noise) 를 self-distillation 으로 정면 대응. PPG + ECG dual-track Transformer 로 21 M waveforms / 179,757 hours 사전학습. False alarm 감소, AFib detection, ABP estimation 평가.

## Domain & Task
- **ICU·OR** 환경의 PPG + ECG (**본 연구와 직접 같은 domain**)
- VTac false alarm detection, AFib detection, ABP estimation
- 신호 품질 저하에 robust 한 학습

## Method
- **Self-distillation** : 고품질 신호 encoder 가 저품질 신호 encoder 를 guide
- **Dual-track architecture** : PPG track + ECG track parallel
- **Windowed sparse attention** Transformer
- 모델 크기 **9.6 M ~ 319 M** (3 variant)
- Loss : distillation + spectral reconstruction

## Datasets
- **21 M waveform segments (30 sec)**, **179,757 hours**
- ICU monitoring 위주

## Key Results
- VTac false alarm 감지에서 SOTA
- AFib detection, ABP estimation 도 우수
- 저품질 신호에서 high-quality model 의 성능 retain

## Relation to 본 연구 (가장 직접적 baseline 중 하나)
- **ICU/OR domain + PPG + ECG multi-modal FM** → 본 연구와 **scope 겹침** (단, 본 연구는 6 종, QualityFM 은 2 종)
- Dual-track 분리 vs 본 연구의 단일 unified encoder + Binary Attention Bias — architecture 측면 대조
- **신호 품질 측면 explicit 처리** : 본 연구에 부재한 차별점, 향후 추가 가능
- 모델 크기 9.6–319M 은 본 연구의 capacity 비교 baseline

## Limitations / 차이
- PPG + ECG 두 modality 만 — CVP, PAP, ICP 등 invasive pressure signal 미지원
- Self-distillation 은 noise robust 측면 강점이지만 cross-modal coupling 학습 부족
- Cross-modal patch modeling 등 본 연구의 핵심 contribution 부재

## Citation 위치 후보
- §Related Work / Multi-modal physiological FM (가장 직접적 baseline)
- §Datasets / Pretraining 규모 비교 (179K hours)
- §Experiments / AFib, ABP 관련 baseline
- Discussion / 향후 신호 품질 robust 학습 확장 가능성
