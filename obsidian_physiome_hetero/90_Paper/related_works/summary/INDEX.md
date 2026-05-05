---
tags: [related-works, index]
project: biosignal-foundation-model
---

# Related Works — Summary Index

본 디렉토리는 `papers/` 의 24 개 PDF 를 분석한 summary 모음. 각 파일은 표준 format (TL;DR / Domain / Method / Datasets / Key Results / Relation to 본 연구 / Citation 위치 후보) 으로 작성됨.

향후 paper 작성 시 §Related Work, §Methods, §Experiments 의 baseline·인용 후보 검색용 index.

## 카테고리별 분류

### 🟦 General Time-Series Foundation Models (cardiovascular 비특화)

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[TST_2020]] | Transformer for Multivariate Time Series Representation Learning | 2020 (KDD'21) | Masked input reconstruction, 첫 TS Transformer SSL |
| [[TimesFM_2024]] | A Decoder-only Foundation Model for Time-series Forecasting | 2024 (ICML, Google) | **Residual MLP Patch + Block Next Prediction 의 출처** |
| [[MOMENT_2024]] | MOMENT: A Family of Open Time-series Foundation Models | 2024 (ICML, CMU) | T5 encoder + masked reconstruction. **본 연구가 정규화 한계 motivation 으로 인용** |
| [[Chronos_2024]] | Chronos: Learning the Language of Time Series | 2024 (Amazon) | Tokenization + LM pretraining (T5 family) |
| [[Moirai_2024]] | Unified Training of Universal Time Series Forecasting Transformers | 2024 (ICML, Salesforce) | 27B+ obs, 9 domains LOTSA |

### 🟥 ECG Foundation Models / Self-Supervised Learning

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[HeartBEiT_2023]] | A foundational vision transformer improves diagnostic performance for electrocardiograms | 2023 (npj DM) | 8.5M ECG → image, BEiT MIM. **npj DM target venue reference** |
| [[ECG_FM_2024]] | ECG-FM: An Open Electrocardiogram Foundation Model | 2024 (Toronto) | 1.5M ECG, hybrid contrastive+generative, **AFib AUROC 0.996** |
| [[NEJMAI_ECGFM_2025]] | A Foundation Transformer Model for ECG-Based Cardiac/Coronary Function | 2025 (NEJM AI) | Modified ViT, MIMIC-IV-ECG 800K |
| [[ECG_SemanticIntegrator_2024]] | ECG Semantic Integrator (ESI): LLM-Enhanced Cardiological Text | 2024 | ECG + LLM-generated text contrastive |
| [[Mehari_Strodthoff_2021]] | Self-supervised representation learning from 12-lead ECG data | 2022 (Comp Bio Med) | SimCLR/BYOL/SwAV/CPC 비교, CPC 우수 |
| [[PCLR_2022]] | Patient Contrastive Learning of Representations | 2022 (PLOS CB) | **Patient-level positive pair** — 본 연구 Cross-Modal Contrastive 의 prequel |
| [[3KG_2021]] | 3KG: Contrastive Learning with Physiologically-Inspired Augmentations | 2021 (ML4H) | VCG 공간 회전 augmentation |
| [[CLOCS_2021]] | CLOCS: Contrastive Learning Across Space, Time, Patients | 2021 (ICML) | Spatial/temporal/patient 3-axis contrastive |
| [[Oh_LeadAgnostic_2022]] | Lead-agnostic Self-supervised Learning for Local and Global ECG | 2022 (CHIL) | Random lead masking, local+global |

### 🟩 EEG Foundation Models

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[LaBraM_2024]] | Large Brain Model for Learning Generic Representations | 2024 (ICLR) | ~2,500h EEG, channel patch + vector quantized |
| [[EEGPT_2024]] | EEGPT: Pretrained Transformer for Universal Reliable EEG | 2024 (NeurIPS) | 10M params, dual SSL (masked + alignment) |
| [[NeuroNet_2024]] | Hybrid SSL for Sleep Stage Classification | 2024 | Contrastive + masked + Mamba |

### 🟨 PPG / Optical Foundation Models

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[PaPaGei_2025]] | PaPaGei: Open Foundation Models for Optical Physiological Signals | 2025 (ICLR) | 57K hours PPG, **70× 작은 모델로 SOTA** |
| [[WaveletPPG_2026]] | Wavelet-Driven Masked Multiscale Reconstruction for PPG | 2026 (Samsung) | 17M PPG, wavelet decomposition + masked |
| [[Apple_Wearable_FM_2024]] | Large-scale Training of Foundation Models for Wearable Biosignals | 2024 (ICLR, Apple) | **141K participants, 3 years**, PPG+ECG, participant-level positive |

### 🟪 Multi-biosignal / ICU Foundation Models (가장 직접적 baseline)

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[BIOT_2023]] | BIOT: Biosignal Transformer for Cross-data Learning in the Wild | 2023 (NeurIPS) | EEG+ECG+HAR, channel × time patch tokenization |
| [[QualityFM_2025]] | QualityFM: Multimodal Physiological Signal FM with Self-Distillation | 2025 | **ICU/OR PPG+ECG 21M waveform, 179K hours** — 본 연구와 가장 직접적 baseline |
| [[DigitalStethoscope_FM_2024]] | Foundation models for cardiovascular disease detection via digital stethoscopes | 2024 (npj Cardio) | Acoustic + ECG, npj Cardiovascular Health venue |

### 🟫 Cross-modality Transfer

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[Toth_BPEstimation_2025]] | Finetuning EEG FM on ECG/PPG for BP Estimation | 2025 (EMBC) | EEG → ECG/PPG transfer, **MIMIC-III + VitalDB** (본 연구와 같은 데이터셋) |

### 🟧 VitalLab @ SNUH — npj Digital Medicine 시리즈 (Hyung-Chul Lee 저자, **본 연구의 stylistic reference**)

> 본 연구의 **Introduction / Discussion 작성 시 문체·narrative pattern 참고용**. npj DM 의 perioperative / ICU AI paper 표준 어휘·구조의 직접 reference.

| 파일 | 제목 | 연도 | 핵심 |
|---|---|---|---|
| [[VitalLab_PostopMortality_2022]] | Multi-center validation of ML model for postop mortality | 2022 (npj DM) | **한국 4 기관 454K 환자**, light XGBoost, AUROC 0.94 — multi-center generalization narrative reference |
| [[VitalLab_AIVE_2023]] | Reinforcement learning for ventilation control during anesthesia emergence | 2023 (npj DM) | Offline RL, 1-초 단위 14K 사례 + 외부 검증 — **continuous waveform-based real-time decision support** narrative reference |
| [[VitalLab_CardiacArrestHRV_2023]] | Real-time ML for in-hospital cardiac arrest using HRV in ICU | 2023 (npj DM) | **VitalDB ECG-only LGBM**, AUROC 0.881 — 본 연구의 cardiac arrest task 직접 baseline |
| [[VitalLab_DELPHI_EEG_2025]] | Deep learning for postop delirium from intraop EEG | 2025 (npj DM) | 6-lead intraop EEG deep learning, AUROC 0.870 — **2025 latest npj DM raw biosignal DL precedent** |

## 본 연구와의 관련도 (priority for citation)

### ⭐⭐⭐ Tier 1 — 직접 baseline / 본 연구의 핵심 contribution 의 prequel

- [[TimesFM_2024]] — Residual MLP Patch + Block Next Prediction 출처 (이미 인용됨)
- [[MOMENT_2024]] — 정규화 도메인 한계 motivation (이미 인용됨)
- [[BIOT_2023]] — multi-biosignal unified Transformer 의 직접 선조
- [[QualityFM_2025]] — ICU/OR multi-modal physiological FM, scope 가장 가까움
- [[ECG_FM_2024]] — open ECG FM, hybrid SSL paradigm
- [[Apple_Wearable_FM_2024]] — large-scale PPG+ECG FM, participant-level contrastive
- [[HeartBEiT_2023]] — npj DM target venue, ECG FM precedent
- [[VitalLab_CardiacArrestHRV_2023]] — VitalDB cardiac arrest 의 직접 baseline (본 연구 downstream task)
- **VitalLab npj DM series (4 papers)** — npj DM 의 perioperative/ICU AI 작성 표준 어휘·structure 의 stylistic reference

### ⭐⭐ Tier 2 — Related Work 카테고리 대표

- [[CLOCS_2021]] — Cross-Modal Contrastive Learning 의 conceptual 출처
- [[PCLR_2022]] — Patient-level positive pair 의 출처
- [[Oh_LeadAgnostic_2022]] — Dual Embedding lead/spatial robust 학습 baseline
- [[Mehari_Strodthoff_2021]] — ECG SSL systematic benchmark
- [[3KG_2021]] — Physiologically-inspired contrastive aug
- [[PaPaGei_2025]] — Open PPG FM, fairness benchmark
- [[Moirai_2024]] / [[Chronos_2024]] — General time-series FM baseline
- [[NEJMAI_ECGFM_2025]] — NEJM AI venue precedent

### ⭐ Tier 3 — 보조 reference

- [[TST_2020]] — masked time-series transformer SSL 의 원전
- [[LaBraM_2024]] / [[EEGPT_2024]] — EEG FM baseline (본 연구 future work)
- [[NeuroNet_2024]] — hybrid SSL paradigm 사례
- [[ECG_SemanticIntegrator_2024]] — text alignment future work
- [[WaveletPPG_2026]] — frequency-domain SSL
- [[DigitalStethoscope_FM_2024]] — acoustic+ECG, future modality 확장
- [[Toth_BPEstimation_2025]] — VitalDB 사용 비교 baseline

## 주요 매핑 — 본 연구 component 별 관련 논문

| 본 연구 component | 직접 prior |
|---|---|
| Residual MLP Patch Projection | TimesFM |
| Block Next Prediction | TimesFM |
| Conditional Layer Normalization (LSCNorm) | (FiLM, AdaLN-Zero — already in reference.md) |
| Binary Attention Bias | BIOT (channel × time tokenization) |
| Cross-Modal Patch Modeling | BIOT (cross-data), Apple Wearable |
| Cross-Modal Contrastive Learning | PCLR, CLOCS, Apple Wearable |
| Sequence Packing (FFD) | (Krell-2021 — already in reference.md) |
| Multi-Resolution STFT Loss | WaveletPPG (다른 spectral SSL 사례) |
| Patient-level Aggregator | (downstream level, no direct prior) |

## Action Items (보완 필요)

- [[DigitalStethoscope_FM_2024]] — Nature 직접 fetch 실패. PDF 직접 확인 후 저자·architecture·데이터셋 보완 필요
- [[NEJMAI_ECGFM_2025]] — 정확한 architecture (modified ViT 의 변형 종류) 본문 확인 필요
- [[PCLR_2022]] — 정확한 저자 list 본문 확인 필요
- 모든 summary 의 *"Datasets"* / *"Key Results"* 수치는 abstract level — paper 작성 시 본문 직접 확인하여 정확한 N·AUROC 값 인용 권장
