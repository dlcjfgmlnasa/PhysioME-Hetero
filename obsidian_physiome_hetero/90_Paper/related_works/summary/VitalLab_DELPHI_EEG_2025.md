---
nature: 10.1038/s41746-025-02033-y
pmcid: PMC12623934
pmid: 41249487
title: "Development of a deep learning-based prediction model for postoperative delirium using intraoperative electroencephalogram in adults"
authors: Jang Ho Ahn, Hyeonhoon Lee, Pedro Gambus, Hyun-Kyu Yoon, Jae-Woo Ju, Hyung-Chul Lee (last, corresponding)
venue: npj Digital Medicine 2025 (SNUH)
year: 2025
domain: Anesthesia outcome — postoperative delirium prediction from intraoperative EEG
tags: [related-works, vitallab, npj-digital-medicine, hyung-chul-lee, anesthesia, EEG, deep-learning, postoperative-delirium, writing-style-reference]
---

# VitalLab — DELPHI-EEG: Deep Learning for Postoperative Delirium from Intraop EEG (npj DM 2025)

## TL;DR
**6-lead intraoperative EEG waveform** 만으로 postoperative delirium (POD) 을 예측하는 deep learning 모델 (DELPHI-EEG). SNUH 단일기관 34,550 사례 (POD event 267 사례) 에서 5-fold CV AUROC **0.870** (95% CI 0.789-0.935). burst suppression ratio 기반 logistic regression baseline 대비 유의 우월 (0.870 vs 0.729, p=0.004). 본 연구의 **EEG biosignal-based deep learning 임상 의사결정 지원** 의 npj DM 최신 reference (2025 publication).

## Domain & Task
- 성인 환자의 **postoperative delirium (POD)** 예측 — binary classification
- Input : **6-lead intraoperative EEG waveform** (수술 중 마취 모니터링)
- 임상 motivation : POD 가 morbidity/mortality 증가, 입원 연장, 장기 인지 저하와 연관됨에도 사전 예측 도구 미비

## Method
- **모델** : Deep learning network (DELPHI-EEG, raw waveform 처리)
- 6-channel intraoperative EEG → POD 확률
- baseline 비교 : burst suppression ratio 기반 logistic regression
- **검증** : 5-fold cross-validation (단일 기관 cohort)

## Datasets
- **SNUH 단일 기관, 2022-2024** 사이 6-channel intraoperative EEG 모니터링 받은 성인 환자
- 34,550 사례 (POD event **267 사례**, 0.77% imbalance)
- CAM-ICU label subset 4,268

## Key Results
| 지표 | DELPHI-EEG | LR baseline (BSR) | p-value |
|---|---|---|---|
| AUROC (95% CI) | **0.870** (0.789-0.935) | 0.729 (0.624-0.825) | 0.004 |
| AUPRC (95% CI) | 0.038 (0.017-0.084) | 0.013 (0.007-0.026) | 0.002 |

- **AUROC 0.870 vs 0.729** — deep learning 이 hand-crafted EEG feature 대비 유의 우월
- AUPRC 가 절대치로 낮은 것은 극심한 class imbalance (event ratio 0.77%) 때문

## Relation to 본 연구
- **EEG biosignal 기반 deep learning 임상 prediction 의 npj DM 2025 reference** — 본 연구는 EEG 를 현재 scope 에서 제외하지만, 향후 확장 가능성 (Decision_EEG_Excluded → 미래 task) 의 stylistic precedent
- **VitalLab 의 deep learning + clinical biosignal 조합 paper 의 가장 최신 사례**
- **Single-center single-modality** 의 한계를 본 연구가 *"multi-source K-MIMIC + multi-modal 9 종 통합"* 으로 어떻게 해결하는지 대조 narrative 제공

## Limitations / 차이
- Single-center, single-task (POD only) — 본 연구의 multi-task FM 와 paradigm 차이
- 6-lead EEG 단일 모달리티 — 본 연구의 9 종 cardiorespiratory 와 다른 scope
- 외부 검증 부재 — 본 연구의 cross-domain (한국 → 미국) generalization 의 차별점
- Hand-crafted feature baseline 만 비교 — foundation model paradigm 미사용

## Citation 위치 후보 (본 연구)
- §Discussion — *"VitalLab 의 deep learning + biosignal 임상 prediction 의 일관된 series"* 의 일원으로 본 연구의 위치 정당화
- §Future Work — EEG modality 확장 시 reference (intraoperative 6-lead EEG 의 임상 가치 입증)
- §Related Work — npj DM 2025 의 raw biosignal deep learning 흐름 사례

## ⭐ Writing Style 참고 — Intro/Discussion 작성용

### 1. **임상 결과 (POD) 의 morbidity/mortality 연관성으로 motivation 시작**
   - *"Postoperative delirium (POD) is associated with increased morbidity and mortality"* — abstract 첫 문장
   - 임상 결과 → 사회적 부담 → AI 개입 필요성의 표준 progression
   - 본 연구 적용 : Introduction Para 1 의 임상 사건 (저혈압, 부정맥, 심정지) 나열 패턴과 같은 구조

### 2. **"Significantly outperforming" 의 단언적 비교 (p-value 명시)**
   - *"significantly outperforming the logistic regression model using burst suppression ratio with AUROC of 0.729 (p = 0.004)"*
   - **abstract 에 baseline 모델명, 정확한 수치, p-value 까지** 포함
   - 본 연구 적용 : Abstract 의 본 연구 vs ECG-FM / BIOT / MOMENT baseline 비교를 같은 식으로 *"significantly outperforming X with AUROC Y vs Z (p = ...)"* 형태로 작성

### 3. **Discussion 의 "Risk predictor → Targeted intervention" framing**
   - *"might serve as a risk predictor for postoperative delirium, potentially enabling targeted preventive interventions for surgical patients"*
   - 모델의 임상적 actionability 를 명시 — 단순 prediction X, **개입 가능 시점/방법 시사**
   - 본 연구 적용 : Discussion 의 *"본 모델은 ICU 의사가 5-30 min ahead 로 hypotension·arrhythmia·ICH 위험 환자를 식별하여 예방적 개입 (혈관수축제·항부정맥제·체위 변경) 의 시점을 결정 지원"* 로 actionability 강조

### 4. **External validation 부재의 투명한 인정**
   - *"nonetheless, external validation in diverse clinical settings is required"*
   - Abstract 마지막 한 줄에 명시적 한계 고백 — defensiveness 회피
   - 본 연구 적용 : Abstract / Discussion 마지막에 *"K-MIMIC 단독 pretrain 의 cross-population fairness 는 제한적이며, 향후 다국가 ICU consortium federated pretrain 으로 확장 필요"* 로 명시

### 5. **Hand-crafted feature 와의 비교로 deep learning 정당화**
   - *"DELPHI-EEG (deep learning) vs logistic regression with burst suppression ratio (hand-crafted feature)"*
   - End-to-end learning 의 stylistic 우월성 입증 패턴
   - 본 연구 적용 : ECG/HRV hand-crafted feature baseline (위 CardiacArrestHRV paper 의 LGBM) 대비 본 연구 FM 의 학습 표현 우월성 비교

### 6. **AUROC 와 AUPRC 동시 보고 (clinical imbalance 고려)**
   - 극심한 class imbalance (POD 0.77%) 환경에서 AUPRC 도 동시 보고
   - 본 연구 적용 : Cardiac Arrest, Sepsis 등 imbalanced task 에서 AUROC + AUPRC 병기 (이미 paper 에 명시됨, npj DM 표준 패턴 정합)

## 원본 자료
- PDF: `obsidian/90_Paper/related_works/papers/VitalLab_NPJDM_2025_DELPHI_EEG.pdf`
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12623934/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/41249487/
- Nature: https://www.nature.com/articles/s41746-025-02033-y
