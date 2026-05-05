---
arxiv: 2203.06889
title: "Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram"
authors: Oh, Chung, Kwon, Hong, Choi
venue: CHIL 2022
year: 2022
domain: ECG self-supervised learning
tags: [related-works, ECG, self-supervised, lead-masking, local-global]
---

# Lead-agnostic SSL for Local and Global ECG Representations (Oh et al., CHIL 2022)

## TL;DR
ECG SSL 의 두 가지 한계 — (1) 대부분 global context 만 학습, (2) 고정된 lead 구성 가정 — 를 동시에 해결. **Random lead masking** 과 **local + global 학습** 으로 임의의 lead 구성에 robust 한 모델 학습.

## Domain & Task
- 12-lead ECG (PTB-XL 등)
- 부정맥·심혈관 질환 분류
- **임의의 lead subset 입력에 robust** 한 generalization

## Method
- **Random lead masking** : pretraining 중 일부 lead 를 random 으로 mask → 임의의 lead 구성에 적응
- **Local + Global 학습** : patch-level local representation 과 sequence-level global representation 동시 학습
- Encoder : Transformer-based
- 임의 lead 수 입력 가능한 lead-agnostic 설계

## Datasets
- PTB-XL
- PhysioNet 2020 challenge
- 다중 데이터셋 cross-evaluation

## Key Results
- 기존 SSL 방식 대비 lead 가용성 변화 시 성능 강건
- Local + global representation 결합이 단일 representation 대비 향상

## Relation to 본 연구
- **임의 lead 구성에 robust** 한 학습 = 본 연구의 **Dual Additive Embedding (modality + spatial)** 과 같은 motivation
- ECG 단일 modality 안의 *"lead-agnostic"* → 본 연구는 6 종 modality 와 그 안의 spatial channel 까지 일반화
- Local + Global 학습 = 본 연구의 patch-level reconstruction + contrastive 의 dual-scale 학습과 유사 철학

## Limitations / 차이
- ECG 단일 modality
- Cross-modal alignment 미고려
- Foundation model scale 사전학습 아님

## Citation 위치 후보
- §Related Work / Lead-agnostic / Channel-flexible biosignal SSL
- §Dual Additive Embedding (lead/spatial 가변성에 대한 robust 설계의 baseline)
