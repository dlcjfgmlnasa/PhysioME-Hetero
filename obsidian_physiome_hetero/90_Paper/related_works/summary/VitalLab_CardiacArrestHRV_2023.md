---
nature: 10.1038/s41746-023-00960-2
pmcid: PMC10665411
title: "Real-time machine learning model to predict in-hospital cardiac arrest using heart rate variability in ICU"
authors: Hyeonhoon Lee, Hyun-Lim Yang, Ho Geol Ryu, Chul-Woo Jung, Youn Joung Cho, Soo Bin Yoon, Hyun-Kyu Yoon, Hyung-Chul Lee (corresponding, last)
venue: npj Digital Medicine 2023 (SNUH, VitalDB)
year: 2023
domain: ICU cardiac arrest prediction — ECG/HRV-based ML
tags: [related-works, vitallab, npj-digital-medicine, hyung-chul-lee, ICU, cardiac-arrest, HRV, vitaldb, real-time, writing-style-reference]
---

# VitalLab — Real-Time ICU Cardiac Arrest Prediction via HRV (npj DM 2023)

## TL;DR
**ECG-only LGBM 모델**로 **VitalDB SNUH ICU 데이터** (4,821 환자, 5,679 ICU stays) 의 in-hospital cardiac arrest 를 0.5-24 시간 전 실시간 예측. AUROC **0.881** (95% CI 0.875-0.887). ECG 만 사용함으로써 multi-source EMR 통합의 임상 운용 부담을 제거. 본 연구의 **VitalDB cardiac arrest downstream task 의 직접 baseline** + **HRV 기반 ECG-only 임상 framing** 의 reference.

## Domain & Task
- **ICU 입실 환자의 in-hospital cardiac arrest 실시간 예측** (binary, 0.5-24 시간 horizon)
- ECG 5 분 epoch + 5 분 sliding interval 으로 streaming inference
- 임상 motivation : 제한된 ICU 자원 + 심정지 원인의 다양성 → 단일 신호 만으로 광범위 적용 가능한 모델 필요

## Method
- **모델** : Light Gradient Boosting Machine (LGBM)
  - Bayesian hyperparameter optimization
  - Beta calibration
- **Feature engineering** : 43 개 HRV measure (24 time-domain, 9 frequency-domain, 15 nonlinear)
  - BorutaShap → 33 개 selected
  - Neurokit2 Python library
- **평가** : AUROC, AUPRC, sensitivity, specificity, precision, F1-score, SHAP, Kendall's tau (시간 추이)

## Datasets
- **VitalDB** (SNUH prospective registry)
- 5,771 환자 (6,982 ICU stays) → quality control 후 **4,821 환자, 5,679 ICU stays**
- Cardiac arrest 사건 : **107 건 (1.88%)**
- 80/20 patient-level split
- Epoch 수 : 개발 634,396 / 검증 139,663

## Key Results
| 지표 (0.5-24h) | 수치 (95% CI) |
|---|---|
| AUROC | **0.881** (0.875-0.887) |
| AUPRC | 0.104 (0.093-0.116) |
| Sensitivity | 0.817 (0.800-0.834) |
| Specificity | 0.800 (0.798-0.802) |
| Precision | 0.053 (0.051-0.056) |
| F1-score | 0.100 (0.095-0.104) |

- **임상 변수 baseline 모델 (HR, SBP, DBP, MAP, SpO₂, RR) 대비** : AUROC 0.881 vs 0.735 (p < 0.001)
- 가장 중요한 feature Top 6 : TINN, HTI, IALS, Prc20NN, MinNN, IQRNN

## Relation to 본 연구
- **본 연구의 Cardiac Arrest Prediction downstream task 의 직접 baseline**
  - 같은 outcome (cardiac arrest), 같은 dataset 계열 (VitalDB ICU subset), 같은 시간 horizon 설정
  - 본 연구 §Cardiac Arrest 의 5/15/30 min horizon, risk-set matching 설계가 이 paper 와 비교 평가 가능
- **ECG-only baseline** — 본 연구의 multi-modal 통합 (ECG + ABP + PPG + CVP) 가 ECG-only 대비 얼마나 향상시키는지의 직접 비교 가능
- **HRV 의 nonlinear time dynamics 가 ML 로 학습 가능** — 본 연구 foundation model 의 implicit 학습 가설을 정당화
- **VitalDB 활용 학술 표준** — 본 연구의 OR holdout / downstream 평가에서 VitalDB 인용 시 같은 cohort 의 prior 표준

## Limitations / 차이
- ECG-only — 본 연구의 9 종 multi-modal 과 직접 대조
- Hand-crafted HRV feature — 본 연구의 end-to-end 학습 표현과 paradigm 차이
- 단일 코호트 (VitalDB SNUH) 후향적 — 본 연구는 K-MIMIC pretrain + MIMIC-III WDB 외부 검증으로 cross-population 검증

## Citation 위치 후보 (본 연구)
- §Cardiac Arrest Prediction (Supplementary Methods) — VitalDB 기반 ECG-only baseline 비교 명시
- §Discussion — multi-modal 표현이 ECG-only HRV 대비 향상시키는 시점·횟수 정량화 시
- §Related Work — VitalDB 활용 ICU prediction 의 prior

## ⭐ Writing Style 참고 — Intro/Discussion 작성용

### 1. **"의료 자원 제약" 프레이밍 으로 임상 motivation 강화**
   - *"limited ICU resources and diverse causes"* — 기술적 우월성보다 **임상적 필요성** 을 먼저 제시
   - Introduction 에서 *"기존 vital sign 기반 EWS (NEWS, MEWS) 의 한계"* → *"new approach needed"* 로 이행
   - 본 연구 적용 : Introduction 에서 *"ICU 의 다중 신호를 임상의가 통합 해석하기 어려운 자원 제약"* → *"automated decision support 필요"* 로 이미 강조 (강화 가능)

### 2. **단일 데이터 source 의 "접근성/이전성" 강조**
   - *"ECG, widely used for continuous monitoring... can simplify the process and ensure constant, real-time monitoring"*
   - **이미 임상 표준 도구를 활용함으로써** *"high accessibility and transferability to other healthcare settings"* 어필
   - 본 연구 적용 : *"6 cardiovascular + 3 respiratory 모두 ICU/OR 표준 모니터로 측정되어 추가 hardware 불요"* 로 transferability 어필

### 3. **비선형 / 복잡 동역학을 ML 도입 정당화로 활용**
   - *"HRV measures ... render them cumbersome in conventional statistical models"*
   - → *"ML-based models handle complex relationships without requiring prespecified assumptions"*
   - 본 연구 적용 : *"심혈관-호흡 결합의 비선형 인과 동역학 (heart-lung coupling, baroreflex, ventilator-induced ABP variation) 은 hand-crafted feature 로 포착 불가"* → *"foundation model 의 implicit 학습으로만 가능"* 정당화

### 4. **Limitation → Constructive Future Path 패턴**
   - 단순 한계 나열 X → *"Future research should focus on validating ... in larger multi-center studies"* + *"incorporating clinical factors may further assist the model"*
   - 한계 + 후속 단계를 한 문장으로 묶어 momentum 유지
   - 본 연구 적용 : *"K-MIMIC 단일 cohort pretrain → 향후 MIMIC-IV-WDB · eICU · 다국가 ICU consortium 으로 federated pretrain 확장"* 같은 형태

### 5. **Calibration / 시계열 동역학 시각화의 중요성**
   - 단순 AUROC 만 X → calibration plot, *"changes in HRV measures over time until event"* 분석
   - 본 연구 적용 : Discussion 에서 *"npj DM 의 권장 사항인 calibration 과 DCA (Decision Curve Analysis) 를 supplementary 에 포함"* — 본 연구가 이미 계획한 사항의 stylistic 정당화

### 6. **임상 운용성의 명시적 어필**
   - *"As our model uses only ECG data, it can be easily applied in clinical practice"* — abstract 의 마지막 한 줄
   - 본 연구 적용 : Abstract 마지막에 *"single architecture handles 9 modalities + 9 downstream tasks → minimal deployment overhead in ICU"* 같은 매듭

## 원본 자료
- PDF: `obsidian/90_Paper/related_works/papers/VitalLab_NPJDM_2023_CardiacArrestHRV.pdf`
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10665411/
- Nature: https://www.nature.com/articles/s41746-023-00960-2
