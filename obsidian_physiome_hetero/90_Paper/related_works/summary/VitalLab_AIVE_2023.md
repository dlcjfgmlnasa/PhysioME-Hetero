---
nature: 10.1038/s41746-023-00893-w
pmcid: PMC10425339
title: "Development and validation of a reinforcement learning model for ventilation control during emergence from general anesthesia"
authors: Hyeonhoon Lee (co-first), Hyun-Kyu Yoon (co-first), Jaewon Kim (co-first), Ji Soo Park, Chang-Hoon Koo, Dongwook Won, Hyung-Chul Lee (corresponding, last)
venue: npj Digital Medicine 2023 (SNUH, multi-center)
year: 2023
domain: Anesthesia ventilation control — offline RL
tags: [related-works, vitallab, npj-digital-medicine, hyung-chul-lee, anesthesia, reinforcement-learning, multicenter-validation, writing-style-reference]
---

# VitalLab — AIVE: RL for Ventilation Control during Anesthesia Emergence (npj DM 2023)

## TL;DR
**Offline RL (Conservative-Q Learning) 기반 환기 제어 AI** (AIVE). 14,306 사례 (1-초 단위 6.7M data points) 로 학습, 별도 학술병원 406 사례로 외부 검증. AIVE 정책의 estimated reward 가 임상의 정책보다 **유의하게 높고** (내부 0.185 vs −0.406, 외부 0.506 vs 0.154), 정책 불일치가 클수록 cardiorespiratory instability 가 증가함을 통계적으로 입증. 본 연구의 **continuous waveform-based real-time clinical decision support** narrative 의 모범적 reference.

## Domain & Task
- **마취 각성 (emergence from general anesthesia) 중 환기 제어** (ON/OFF) 의 최적 timing 결정
- 1-초 단위 ventilatory + hemodynamic parameter 기반 sequential decision-making
- 임상 motivation : 인간 임상의의 주의 산만·피로로 인한 cardiorespiratory instability 위험 회피

## Method
- **Conservative-Q learning** (offline RL, distributional shift 안전성)
- 신경망 기반 architecture, 매 초 환자 상태 → action 매핑
- **State space (10 features)** : propofol/remifentanil 효과부위 농도, sevoflurane 농도, peak inspiratory pressure, tidal volume, airway pressure, ETCO₂, HR, SpO₂, SBP, 자발호흡 유무, apnea 누적 시간, 환기·extubation 상태
- **Action** : 환기 ON/OFF 이원
- **평가** : Fitted-Q-evaluation (FQE) bootstrapping (300 모델), 95% confidence bounds

## Datasets
- 내부 (SNUH 추정) : 2016-2019 14,306 사례 (6,763,535 1-초 data points)
  - 85% 훈련 (12,160), 15% 내부 검증 (2,146)
- 외부 (다른 학술병원, 2022) : **406 사례** (162,656 1-초 data points)

## Key Results
| 지표 | 내부 검증 | 외부 검증 |
|---|---|---|
| AIVE 정책 reward (95% lower bound) | **0.185** | **0.506** |
| 임상의 정책 reward (95% upper bound) | −0.406 | 0.154 |
| 정책 불일치 ↔ cardiorespiratory instability 상관 | r=0.252 (P<0.001) | r=0.216 (P<0.001) |
| Feature importance #1 | airway pressure | airway pressure |

## Relation to 본 연구
- **연속 ventilatory + hemodynamic waveform 기반 임상 의사결정** — 본 연구 cardiorespiratory FM 의 핵심 활용 시나리오와 정합
- **Airway pressure 가 핵심 feature** — 본 연구가 AWP 를 9 종 신호에 포함시킨 결정의 임상적 정당화
- **외부 다기관 cohort 으로 generalization 검증** — 본 연구의 K-MIMIC → MIMIC-III WDB cross-domain narrative 의 직접 prequel
- **인간 임상의 한계 (피로, 주의 산만)** 를 motivation 으로 활용 — 본 연구의 *"임상의가 다중 신호를 일관되게 통합 해석하는 것은 현실적으로 어려움"* narrative 의 stylistic reference

## Limitations / 차이
- Task-specific RL (단일 task, 단일 시점) — 본 연구의 multi-task FM 와 다른 paradigm
- Sevoflurane / propofol 등 **수술 중 약물 변수**도 state 에 포함 — 본 연구는 신호만 사용
- AIVE 는 ventilation control 이라는 **action prediction** — 본 연구는 representation pretrain

## Citation 위치 후보 (본 연구)
- §Introduction — 임상의 의사결정 지원 자동화의 ICU/OR 영역 선행 사례
- §Discussion — *"ventilation control 같은 sequential decision task"* 의 사전학습 표현 활용 잠재력
- §Datasets — VitalDB / SNUH cohort 의 multi-modal vital data 활용 선례

## ⭐ Writing Style 참고 — Intro/Discussion 작성용

### 1. **임상 문제의 인간적 취약성을 motivation 으로 framing**
   - *"Human factors may affect the risks of anesthesia emergence"*
   - 임상의의 주의 산만·피로 언급으로 AI 개입 필요성 확보
   - 본 연구 적용 : Introduction Para 1 에 *"의사가 다중 신호를 일관되게 통합 해석하는 것은 현실적으로 어려움"* 으로 이미 적용된 패턴 — 더 강화 가능

### 2. **AI/RL 의 임상적 동치성 명시 (수용성 증대)**
   - *"누적 예상 보상 최대화는 임상의의 의사결정 과정과 유사하다"* 같은 직접 비유
   - 알고리즘과 임상 목표 일치성 강조
   - 본 연구 적용 : *"Cross-Modal Patch Modeling 은 임상의가 ECG·ABP·PPG 의 인과 사슬을 통합 해석하는 reasoning 을 모델 표현 공간에 명시적으로 주입"* 같은 구조

### 3. **95% Confidence Bound 의 보수적 비교**
   - 단순 점 추정이 아닌 *"95% lower bound for proposed > 95% upper bound for baseline"* 로 비교
   - *"보수적 신뢰 구간 비교"* 로 통계적 엄밀성 신호
   - 본 연구 적용 : Results 의 baseline 비교에서 patient-level bootstrap 1,000 회 + 95% CI 적용 (이미 paper 에 명시됨) — Discussion 에서 같은 어휘로 강조 가능

### 4. **Limitation 의 투명한 사전 제시 + Constructive Direction**
   - retrospective bias / exclusion criteria / external cohort 작은 크기 등 투명히 인정
   - *"Future research should address ..."* 로 후속 연구 방향 명시
   - 본 연구 적용 : Discussion 의 limitation 섹션을 *"단일 코호트 기반 / RESP 신호 가용 환자 비율 한계 / cross-population fairness 미검증"* 식으로 투명히 + *"federated cross-institutional pretraining / 다국가 ICU consortium 확장"* 의 future direction

### 5. **임상 구현의 현실적 장벽 명시**
   - *"monitoring 지연, 통신 지연, action 전달 지연"* 등 실제 임상 환경 제약 명시
   - *"distributional shift"* 같은 ML-specific 용어로 기술적 신뢰도 강화
   - 본 연구 적용 : Discussion 에서 *"실시간 추론 시 sliding window latency, packed batch overhead, multi-source channel availability variation"* 등의 구현 장벽 언급

### 6. **Multi-center 외부 검증의 narrative 모범 (소규모라도 명시)**
   - 외부 cohort 가 406 사례로 작아도 *"different academic hospital, different year (2022)"* 라는 분리 명시
   - 본 연구 적용 : MIMIC-III WDB / VitalDB 가 *"독립 데이터셋, 분리된 환자 모집단, 다른 SR (125 vs 500 Hz), 다른 monitor 제조사"* 로 분리됨을 강조

## 원본 자료
- PDF: `obsidian/90_Paper/related_works/papers/VitalLab_NPJDM_2023_AIVE.pdf`
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10425339/
- Nature: https://www.nature.com/articles/s41746-023-00893-w
- GitHub: https://github.com/HyeonhoonLee/AIVE
