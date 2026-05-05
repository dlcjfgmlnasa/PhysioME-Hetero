---
doi: 10.1056/AIoa2500164
title: "A Foundation Transformer Model with Self-Supervised Learning for ECG-Based Assessment of Cardiac and Coronary Function"
authors: Moody, Poitrasson-Rivière, Renaud, Hagio, Alahdab, Al-Mallah, Vanderver, Goonewardena, Ficaro, Murthy
venue: NEJM AI 2(12) (2025)
year: 2025
domain: ECG foundation model — cardiac/coronary function
tags: [related-works, ECG, foundation-model, vision-transformer, NEJM-AI, MIMIC-IV-ECG]
---

# Foundation Transformer for ECG-based Cardiac/Coronary Function (NEJM AI 2025)

## TL;DR
**Modified Vision Transformer** 를 **MIMIC-IV-ECG (N=800,035)** 로 self-supervised pretraining. Cardiac function 과 coronary disease 평가에 적용. NEJM AI 게재.

## Domain & Task
- 12-lead ECG
- 심장 기능 (cardiac function) 과 관상동맥 질환 (coronary disease) 평가
- 임상 의사결정 지원 측면 강조 (NEJM AI publication)

## Method
- **Modified Vision Transformer** (ViT 변형, 구체 변형 paper 본문 확인 필요)
- Self-supervised pretraining
- Sample 800,035 ECG (MIMIC-IV-ECG)

## Datasets
- **MIMIC-IV-ECG** : 800K ECG (PhysioNet)

## Relation to 본 연구
- **NEJM AI publication** — 본 연구가 npj DM 외 도전 가능한 sister venue. Paper format/narrative reference
- ViT-based (HeartBEiT 와 비슷한 paradigm) — 본 연구 1D patch transformer 와 다른 계열
- MIMIC-IV-ECG 800K sample — 본 연구의 K-MIMIC scale 비교 baseline

## Limitations / 차이
- ECG 단일 modality
- Cardiac/coronary function focus, 본 연구의 multi-task (Hypotension·Sepsis·Mortality·AKI 등) 와 scope 차이
- Multi-modal alignment 부재

## Citation 위치 후보
- §Related Work / ECG foundation models (NEJM AI venue 의 대표)
- §Discussion / clinical translation reference (NEJM AI 가 임상 전이 강조)
