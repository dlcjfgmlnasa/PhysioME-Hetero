---
arxiv: 2405.19366
title: "ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text"
authors: Yu, Guo, Sano
venue: arXiv 2024 (revised Oct 2024)
year: 2024
domain: ECG + LLM multi-modal foundation model
tags: [related-works, ECG, multi-modal, LLM, contrastive, captioning]
---

# ESI — ECG Semantic Integrator with LLM-Enhanced Text

## TL;DR
12-lead ECG 사전학습에 **LLM 으로 생성한 cardiological text description** 을 페어링해 multi-modal contrastive + captioning 으로 학습. ECG-텍스트 cross-modal foundation model.

## Domain & Task
- 12-lead ECG + 텍스트 (LLM 생성 caption)
- Arrhythmia detection, subject identification

## Method
- **CQA (Cardio Query Assistant)** : Retrieval-Augmented Generation 으로 ECG 별 텍스트 caption 생성 — 인구통계, 파형 패턴 묘사 포함
- **ESI (ECG Semantics Integrator)** : ECG encoder + 텍스트 encoder, contrastive (CLIP-style) + captioning loss 결합
- ECG → 임상 의미 텍스트로 alignment

## Datasets
- 대규모 ECG + LLM 생성 텍스트 페어 (구체적 수치 abstract 미명시)

## Key Results
- Supervised, self-supervised, 기존 multi-modal 모두 능가
- 부정맥 진단·subject ID 에서 향상

## Relation to 본 연구
- **Multi-modal pretraining 의 ECG-text 변형** — 본 연구의 multi-modal 은 6 종 생체신호 간 alignment, ESI 는 ECG-text alignment
- 본 연구의 CardioWave 와 직접 비교 baseline 으로는 약함 (text modality 사용)
- 향후 본 모델 + clinical text alignment 확장의 reference

## Limitations / 차이
- ECG-only signal modality, 다른 cardiovascular waveform 미지원
- Text alignment 가 임상 deployment 시 LLM 의존성 증가
- Foundation model scale 은 명시 부재

## Citation 위치 후보
- §Related Work / Multi-modal biosignal pretraining (signal-text alignment 의 사례)
- §Discussion / Future work — clinical text 와의 alignment 확장
