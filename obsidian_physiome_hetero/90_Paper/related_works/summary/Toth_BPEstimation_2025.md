---
arxiv: 2502.17460
title: "Finetuning and Quantization of EEG-Based Foundational BioSignal Models on ECG and PPG Data for Blood Pressure Estimation"
authors: Tóth, Senti, Ingolfsson, Zweidler, Elsig, Benini, Li
venue: EMBC 2025
year: 2025
domain: EEG → ECG/PPG transfer for BP estimation
tags: [related-works, EEG, ECG, PPG, BP-estimation, transfer-learning, quantization]
---

# Tóth et al. — EEG FM → ECG/PPG Transfer for BP Estimation (EMBC 2025)

## TL;DR
EEG foundation model 의 표현이 ECG·PPG 로도 fine-tuning 만으로 transfer 됨을 입증. MIMIC-III + VitalDB 에서 SBP MAE 2.72 mmHg, DBP MAE 1.57 mmHg. INT8 quantization 으로 모델 크기 3.5× 감축, wearable 배포.

## Domain & Task
- 입력 : ECG, PPG
- Pretrain source : EEG 기반 foundation model
- Task : 연속 비침습 혈압 (BP) 추정 — systolic/diastolic regression

## Method
- EEG foundation model (사전학습) → ECG/PPG 로 fine-tuning
- Cross-modality transfer 입증
- INT8 dynamic quantization (배포 효율)

## Datasets
- MIMIC-III, **VitalDB** (본 연구와 같은 데이터셋!)

## Key Results
- DBP MAE : **1.57 mmHg**
- SBP MAE : **2.72 mmHg** (SOTA)
- 모델 크기 3.5× 감축 (INT8)

## Relation to 본 연구
- **VitalDB 사용** — 본 연구와 동일 데이터셋, 직접적 비교 가능
- Cross-modality transfer (EEG → ECG/PPG) = 본 연구의 cross-modal 학습과 다른 접근 (transfer vs joint pretraining)
- BP estimation 은 본 연구의 직접 downstream 은 아니지만 **Hypotension prediction 의 prequel** (위험 vs 절대값)
- 단, EEG → ECG/PPG transfer 는 본 연구의 cardiovascular-only scope 와 다른 paradigm

## Limitations / 차이
- BP estimation 단일 task
- Joint pretraining 이 아닌 transfer 만 — modality 간 cross-talk 학습 부족
- Foundation 규모 미명시 (EEG FM 의 크기)

## Citation 위치 후보
- §Related Work / Cross-modality transfer for biosignals
- §Experiments / Hypotension Prediction 의 BP-related baseline
- Discussion / 비침습 wearable 배포 가능성 논의
