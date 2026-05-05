---
tags: [paper, references]
project: biosignal-foundation-model
title: References
---

# References

본 문서는 논문에서 참조되는 외부 reference 를 정리한다.
draft 에서는 `[[reference#키]]` 형식의 Obsidian wikilink 로 연결한다.

> **Citation key 컨벤션**: 모델/메서드 이름이 잘 알려진 경우 그대로 사용 (예: `RevIN`, `MOMENT`). 그렇지 않은 경우 `Author-Year` 형식 사용 (예: `Vaswani-2017`).

---

## K-MIMIC

**IMPACT Consortium / KHIDI** (2021–).
*K-MIMIC: Korea Medical Information Mart for Intensive Care*.
한국 보건복지부 / KHIDI 한국보건기술 R&D 사업 (Grant: HI21C1074).
URL: https://sites.google.com/view/k-mimic/home

> 한국 다기관 ICU 컨소시엄 데이터베이스. IMPACT consortium 주관, 19개 병원 협력 계획. EMR(다기관, 3,234명/3,738 ICU stay), 의료영상, vital signal 의 세 가지 sub-database 로 구성. 본 논문에서는 vital recording subset (`K-MIMIC-MORTAL`, ~3,500명/228K .vital, 880GB) 을 사전학습 단독 데이터로 사용 (§Datasets / K-MIMIC-MORTAL).

---

## VitalDB

**Lee, H.-C., Park, Y., Yoon, S. B., Yang, S. M., Park, D., & Jung, C.-W.** (2022).
*VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*.
Scientific Data, 9, 279.
URL: https://www.nature.com/articles/s41597-022-01411-5
Dataset: https://vitaldb.net/

> 서울대학교병원(SNUH) 수술실에서 수집된 6,388건의 전신마취 사례에 대한 고해상도 다중 채널 vital signs 공개 데이터셋. ECG, ABP, EEG, PPG, CVP, CO₂, AWP 등 7종 신호 동시 측정. 본 논문에서는 사전학습 holdout 으로 분리하여 OR 도메인 일반화 평가 및 intraop downstream task (Hypotension, AKI) 의 자연 benchmark 로 활용 (§Datasets / VitalDB Open).

---

## Vaswani-2017

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017).
*Attention Is All You Need*.
NeurIPS 2017.
arXiv: https://arxiv.org/abs/1706.03762

> Transformer 의 원전. 위치 정보를 sinusoidal positional encoding 으로 표현하여 토큰 임베딩에 element-wise 가산하는 **additive injection** 패턴의 정형적 사례. 본 논문에서 거부 대안으로 논의되는 *"학습 가능한 projection 으로 변환된 보조 정보를 패치 토큰에 가산하는"* 방식의 canonical reference (§Instance Normalization with Loc/Scale Injection).

---

## MIMIC-III-WDB

**Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I.** (2020).
*MIMIC-III Waveform Database* (Version 1.0).
PhysioNet. DOI: https://doi.org/10.13026/c2607m
URL: https://physionet.org/content/mimic3wdb/1.0/

> 67,000+ recordings · 3M+ hours of waveform data, BIDMC ICU 입실 환자 대상 침상 모니터 raw waveform 공개 데이터베이스. 본 논문에서는 외부 병원 일반화 검증용 ICU 도메인 holdout 으로 사용 (§Datasets / MIMIC-III Waveform Database). 인용 시 MIMIC-III 본 논문 (Johnson et al. 2016, *Sci Data* 3:160035) 도 함께 인용 권장.

---

## TimesFM

**Das, A., Kong, W., Sen, R., & Zhou, Y.** (2024).
*A Decoder-only Foundation Model for Time-series Forecasting*.
ICML 2024.
arXiv: https://arxiv.org/abs/2310.10688
GitHub: https://github.com/google-research/timesfm

> Google Research 의 decoder-only 시계열 파운데이션 모델. 패치 입력을 **residual MLP block** (1-hidden-layer MLP + skip connection) 으로 token 으로 변환하는 patch tokenizer 와, output patch length 가 input patch length 보다 큰 **non-autoregressive block prediction** 헤드를 도입. 본 논문의 Residual MLP Patch Projection (§Patch Encoder) 와 Block Next Prediction (§Objectives) 의 구조적 출처.

---

## Krell-2021

**Krell, M. M., Kosec, M., Perez, S. P., & Fitzgibbon, A.** (2021).
*Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance*.
arXiv: https://arxiv.org/abs/2107.02027

> 동일 컨텍스트 윈도우 내에 가변 길이 시퀀스를 cross-contamination 없이 packing 하는 학습 효율화 기법. NLLB 와 BERT 사전학습에서 padding 비율 ≈ 0% 달성. 본 논문의 §Sequence Packing (FFD bin-packing + attention mask 분리) 의 직접 출처.

---

## Mukkamala-2015

**Mukkamala, R., Hahn, J.-O., Inan, O. T., Mestha, L. K., Kim, C.-S., Töreyin, H., & Kyal, S.** (2015).
*Toward Ubiquitous Blood Pressure Monitoring via Pulse Transit Time: Theory and Practice*.
IEEE Transactions on Biomedical Engineering, 62(8), 1879–1901.
DOI: https://doi.org/10.1109/TBME.2015.2441951

> 심전기적 활동 (ECG) 과 말초 맥파 (PPG / ABP) 의 결합을 pulse transit time (PTT) 으로 정량화하는 표준 review. 본 논문에서는 ECG ↔ ABP, ECG ↔ PPG, ABP ↔ PPG cross-modal pair 의 생리학적 결합 근거 (electromechanical coupling, pulse propagation) 로 인용 (§Cross-Modal Patch Modeling [[draft_v00003#Table-CrossModalPairs|Table 2]]).

---

## Czosnyka-2004

**Czosnyka, M., & Pickard, J. D.** (2004).
*Monitoring and Interpretation of Intracranial Pressure*.
Journal of Neurology, Neurosurgery, and Psychiatry, 75(6), 813–821.
DOI: https://doi.org/10.1136/jnnp.2003.033126

> 두개내압 (ICP) 모니터링과 뇌관류압 항등식 CPP = MAP − ICP 의 임상 reference. 본 논문에서는 ABP ↔ ICP cross-modal pair 의 생리학적 결합 (cerebral hemodynamics) 근거로 인용 (§Cross-Modal Patch Modeling Table 2).

---

## Magder-2018

**Magder, S.** (2018).
*Right Atrial Pressure in the Critically Ill: How to Measure, What is the Value, What are the Limitations?*.
Chest, 153(4), 1056–1066.
DOI: https://doi.org/10.1016/j.chest.2017.07.013

> 우심방·우심실 충만압 (CVP) 과 폐동맥압 (PAP) 의 동일 압력 시스템 측정 review. CVP-PAP 박동 결합과 우심 전후부하의 임상 해석을 다룬다. 본 논문에서는 CVP ↔ PAP, ABP ↔ PAP cross-modal pair 의 생리학적 결합 근거로 인용 (§Cross-Modal Patch Modeling Table 2).

---

## RevIN

**Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J.** (2022).
*Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*.
ICLR 2022.
URL: https://openreview.net/forum?id=cGDAkQo1C0p

> 입력에서 instance normalization을 적용하고 출력에서 동일 통계량으로 역정규화하는 paradigm. 본 논문에서는 "출력단 역정규화" 거부 대안의 대표 사례로 인용 (§Instance Normalization with Loc/Scale Injection).

---

## MOMENT

**Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A.** (2024).
*MOMENT: A Family of Open Time-series Foundation Models*.
ICML 2024.
arXiv: https://arxiv.org/abs/2402.03885

> Patch + masked reconstruction 기반 시계열 foundation model. 정규화 도메인에서만 동작하므로 절대 스케일 정보가 표현에서 소실됨 — 본 논문이 극복하려는 한계의 대표 사례 (§Instance Normalization with Loc/Scale Injection).

---

## FiLM

**Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A.** (2018).
*FiLM: Visual Reasoning with a General Conditioning Layer*.
AAAI 2018.
arXiv: https://arxiv.org/abs/1709.07871

> Feature-wise Linear Modulation: 조건 벡터로 feature map의 affine transform $(\gamma, \beta)$를 산출해 곱·합산하는 multiplicative conditioning. LSCNorm의 이론적 모태.

---

## AdaLN-Zero

**Peebles, W., & Xie, S.** (2023).
*Scalable Diffusion Models with Transformers*.
ICCV 2023 (DiT).
arXiv: https://arxiv.org/abs/2212.09748

> Adaptive LayerNorm의 affine 파라미터를 zero-init하여 학습 시작 시점에 modulation이 항등으로 작동하도록 보장. 본 논문 LSCNorm의 zero-init 전략 출처 (§Instance Normalization with Loc/Scale Injection).

---

## AdaIN

**Huang, X., & Belongie, S.** (2017).
*Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization*.
ICCV 2017.
arXiv: https://arxiv.org/abs/1703.06868

> Style transfer에서 instance normalization의 affine 통계 $(\mu, \sigma)$를 외부 conditioning으로 대체하는 adaptive instance normalization. AdaLN/FiLM 계열의 normalization-기반 modulation의 시초.

---

## VitalLab-2022-Mortality

**Lee, S. W., Lee, H.-C., Suh, J., Lee, K. H., Lee, H., Seo, S., Kim, T. K., Lee, S.-W., & Kim, Y.-J.** (2022).
*Multi-center validation of machine learning model for preoperative prediction of postoperative mortality*.
npj Digital Medicine, 5, 91.
DOI: https://doi.org/10.1038/s41746-022-00625-6
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9276734/

> 한국 4 기관 (SNUH, AMC, EUMC, BRMH) 454,404 명 비심장 수술 코호트에서 12-18 개 임상 변수만으로 30-일 사망률을 예측하는 light XGBoost 모델. SNUH-trained 모델이 EUMC 외부 검증에서 AUROC 0.941. 본 논문에서는 한국 다기관 perioperative deep learning 의 임상적 유효성 선례로 인용 (§Introduction Para 1).

---

## VitalLab-2023-AIVE

**Lee, H., Yoon, H.-K., Kim, J., Park, J. S., Koo, C.-H., Won, D., & Lee, H.-C.** (2023).
*Development and validation of a reinforcement learning model for ventilation control during emergence from general anesthesia*.
npj Digital Medicine, 6, 145.
DOI: https://doi.org/10.1038/s41746-023-00893-w
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10425339/

> Conservative-Q learning 기반 offline RL 환기 제어 AI (AIVE). 14,306 사례 1-초 단위 데이터로 학습, 외부 코호트 406 사례 검증에서 임상의 정책 대비 reward 우월 (0.506 vs 0.154, 95% bound 비교). 본 논문에서는 연속 waveform 기반 임상 의사결정 지원의 선행 사례로 인용 (§Introduction Para 1).

---

## VitalLab-2023-CardiacArrest

**Lee, H., Yang, H.-L., Ryu, H. G., Jung, C.-W., Cho, Y. J., Yoon, S. B., Yoon, H.-K., & Lee, H.-C.** (2023).
*Real-time machine learning model to predict in-hospital cardiac arrest using heart rate variability in ICU*.
npj Digital Medicine, 6, 215.
DOI: https://doi.org/10.1038/s41746-023-00960-2
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10665411/

> VitalDB SNUH ICU 데이터 (4,821 환자, 5,679 stays, 107 cardiac arrest events) 의 in-hospital cardiac arrest 를 ECG-based HRV (33 features) + LGBM 으로 0.5-24 시간 ahead 예측. AUROC 0.881 (95% CI 0.875-0.887). 본 논문의 Cardiac Arrest downstream task 의 직접 baseline 이자, 한국 ICU continuous monitoring 의 임상 deep learning 선행 사례 (§Introduction Para 1, §Cardiac Arrest Prediction).

---

## VitalLab-2025-DELPHI

**Ahn, J. H., Lee, H., Gambus, P., Yoon, H.-K., Ju, J.-W., & Lee, H.-C.** (2025).
*Development of a deep learning-based prediction model for postoperative delirium using intraoperative electroencephalogram in adults*.
npj Digital Medicine.
DOI: https://doi.org/10.1038/s41746-025-02033-y
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12623934/
PubMed: https://pubmed.ncbi.nlm.nih.gov/41249487/

> 6-lead intraoperative EEG waveform 만으로 postoperative delirium 을 예측하는 deep learning 모델 (DELPHI-EEG). SNUH 단일 기관 34,550 사례 (267 events). AUROC 0.870 vs burst suppression ratio LR baseline 0.729 (p=0.004). 본 논문에서는 raw biosignal deep learning 의 npj DM 2025 최신 사례로 인용 (§Introduction Para 1).
