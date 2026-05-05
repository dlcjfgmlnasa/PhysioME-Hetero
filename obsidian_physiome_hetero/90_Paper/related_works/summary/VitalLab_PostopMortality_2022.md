---
nature: 10.1038/s41746-022-00625-6
pmcid: PMC9276734
title: "Multi-center validation of machine learning model for preoperative prediction of postoperative mortality"
authors: Seung Wook Lee, Hyung-Chul Lee (2nd), Jungyo Suh, Kyung Hyun Lee, Heonyi Lee, Suryang Seo, Tae Kyong Kim, Sang-Wook Lee, Yi-Jun Kim
venue: npj Digital Medicine 2022 (SNUH / Multi-center 한국)
year: 2022
domain: Perioperative ML — preoperative mortality prediction
tags: [related-works, vitallab, npj-digital-medicine, hyung-chul-lee, perioperative, multi-center, writing-style-reference]
---

# VitalLab — Multi-center Postop Mortality Prediction (npj DM 2022)

## TL;DR
**한국 4 기관 (SNUH, AMC, EUMC, BRMH) 의 454,404 명 비심장 수술 환자**에 대해 12-18 개 임상 변수만으로 30-일 사망률을 예측하는 light XGBoost 모델. SNUH-trained 모델이 EUMC 외부 검증에서 AUROC **0.941** 로 단일기관 학습의 generalization 가능성을 입증. 본 연구 (cardiorespiratory FM) 의 **multi-center generalization narrative** 와 **한국 다기관 코호트 framing** 의 직접 reference.

## Domain & Task
- 비심장 수술 환자의 **30-일 사망률** binary classification
- Preoperative 만 사용 (수술 전 EMR 일부 임상 변수 12-18 개)
- 수술 결정·자원 배분의 임상 의사결정 지원

## Method
- **모델**: Logistic Regression / Random Forest / XGBoost / 5-layer DNN 비교 → XGBoost 최종 채택
- **변수**: 12-18 개 (인구통계, 수술 전 혈액검사, 마취 유형, 응급 여부 등)
- **검증**: Bootstrap, 10-fold CV, Grid search, **다기관 외부 검증** (cross-institution)

## Datasets
- 4 개 한국 기관 통합:
  - SNUH 223,905
  - AMC 66,522
  - EUMC 131,867
  - BRMH 32,110
- 총 **454,404 명** (18세 이상)
- 30 일 사망률 0.2-0.4% (기관별 변동)

## Key Results
- SNUH 자체 검증 AUROC **0.9376** (AUPRC 0.1593)
- SNUH-trained → EUMC 외부 검증 AUROC **0.941** (단일기관 학습으로 다른 기관에서 작동)
- 가장 중요한 변수: 모든 기관에서 공통 — **알부민, 프로트롬빈 시간, 나이**
- 임상 비교 baseline 대비 light 변수 set 으로 동등 이상 성능

## Relation to 본 연구
- **한국 다기관 코호트 generalization 전략의 직접 prequel** — 본 연구의 K-MIMIC (3,500+ 환자) → MIMIC-III WDB (BIDMC 외부 검증) generalization narrative 와 정합
- 본 연구 §Datasets 의 *"미국·유럽 코호트 (MIMIC-III/IV, eICU 등) 에 편중된 cardiovascular AI 데이터 지형에 한국·아시아 다기관 인구의 표현 추가"* 메시지의 **선행 연구**
- npj DM 의 perioperative ML / cross-institution generalization 가 본 연구의 acceptance frame 과 일치

## Limitations / 차이
- 단일 시점 (preoperative) 만 사용 — 본 연구의 연속 waveform pretrain 와 다른 paradigm
- Tabular EMR 변수 — 본 연구는 high-frequency continuous biosignal
- Foundation model 없음 — task-specific shallow model

## Citation 위치 후보 (본 연구)
- §Introduction — 한국 다기관 perioperative ML generalization 의 선행 사례
- §Discussion — cross-institution validation (한국 → 미국 ICU) narrative 의 정당화
- §Datasets — K-MIMIC 가 multi-center 한국 코호트라는 점 강조 시

## ⭐ Writing Style 참고 — Intro/Discussion 작성용

본 연구의 Introduction / Discussion 작성 시 모방 가능한 stylistic 요소:

### 1. **Problem-Solution-Evidence 의 명확한 progression**
   - Introduction 에서 기존 위험도 평가 도구 (ASA-PS, POSSUM, ACS-NSQIP) 의 구체적 한계를 4-5 개 나열 + 각 한계 뒤에 citation
   - 그 후 *"This study thus aimed to..."* 로 자신의 contribution 을 차별화
   - 본 연구 적용 : *"Existing time-series FMs ... ECG-only FMs ... lack X. This study thus aims to..."* 의 구조

### 2. **다기관 generalization 의 가치를 강조하는 명시적 어휘**
   - *"validated against multi-centered rather than single-centered data"*
   - *"an important milestone in the development of generalized, robust models applicable to multiple hospitals"*
   - *"overfitting"*, *"transferability"*, *"real-world heterogeneity"* 같은 용어로 necessity 를 framing
   - 본 연구 적용 : K-MIMIC (한국) → MIMIC-III WDB (미국) cross-domain validation 을 *"transferability across institutions and populations"* 로 framing

### 3. **"Minimal" / "light" 같은 효율성 강조 형용사 반복**
   - *"manageable amount of clinical information"*
   - *"light predictive model using only minimal preoperative information"*
   - 임상 적용성 (feasibility) 을 핵심 가치로 position
   - 본 연구 적용 : *"light backbone with K-MIMIC's manageable token budget"* 등의 framing 가능

### 4. **한국 주도 연구의 국제적 기여를 자연스럽게 제시**
   - 먼저 *"four independent institutions"* 같은 기관 중립적 표현
   - Methods/Results 에서 구체적 기관명 (SNUH, AMC, EUMC, BRMH) 으로 투명성 확보
   - Discussion 에서 *"hospital-dependent and diversely distributed"* 같은 현실 제약 인정 → 글로벌 문제 해결 narrative
   - 본 연구 적용 : K-MIMIC 의 IMPACT 컨소시엄 (KHIDI grant) framing 을 *"Korean multi-center cohort underrepresented in current cardiovascular AI"* 로 글로벌 issue 화

### 5. **Limitation 을 "기회" 프레임으로 전환**
   - *"Although...has the advantage of being able to predict..."* (단점을 역설적으로 장점으로 재해석)
   - *"various techniques ... are expected to be developed through future research"* (e.g. federated learning 언급)
   - 한계를 폐쇄 아닌 *"개선 경로"* 로 표현
   - 본 연구 적용 : Discussion 의 *"K-MIMIC token budget 한계"* / *"cross-population validation 부재"* 를 *"open data initiatives 와 federated learning 을 통한 확장 가능 경로"* 로 재포지셔닝

### 6. **객관적 수치와 학술 언어의 균형**
   - 한국 연구진 주도임에도 영어 medical English 의 형식성 유지
   - 단언적 주장 회피, 지속적인 *"may"*, *"could"*, *"is expected to"* 사용
   - 본 연구 적용 : Foundation model contribution claims 도 hedge 어휘로 보수적 표현

## 원본 자료
- PDF: `obsidian/90_Paper/related_works/papers/VitalLab_NPJDM_2022_PostopMortality.pdf`
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9276734/
- Nature: https://www.nature.com/articles/s41746-022-00625-6
