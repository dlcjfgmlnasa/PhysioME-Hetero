---
tags: [paper, draft]
project: physiome-hetero
title: PhysioME-Hetero — Scalable and Availability-Aware Self-Supervised Learning for Heterogeneous Physiological Signals with Missing Modalities
author: 이철희
status: draft
version: 00001
parent: draft.md
review: review_v00001.md
---

# PhysioME-Hetero — Scalable and Availability-Aware Self-Supervised Learning for Heterogeneous Physiological Signals with Missing Modalities

작성자: 이철희

> ⚠️ **v00001 base.** PhysioME [@PhysioME] 의 두 가지 한계 — (i) 사전 정의된 모달리티 조합 외 일반화 미검증, (ii) 모달리티 수에 선형 비례하는 복원 디코더 비용 — 를 해소하기 위한 후속 프레임워크 PhysioME-Hetero 의 first-pass 드래프트. 위키 재정비·번역·참고문헌 아카이브 작업 로그는 [[review_v00001]] 참조.

## Abstract

(작성 예정)

## Abbreviations

| Category | Abbreviation | Full Term |
|:---|:---|:---|
| **Setting** | ICU | Intensive Care Unit |
| | OR | Operating Room |
| | EMR | Electronic Medical Record |
| **Datasets** | Sleep-EDFX | Sleep European Data Format Expanded |
| | VitalDB | Vital signs Database (SNUH) |
| | MIMIC-IV-WDB | MIMIC-IV Waveform Database |
| | SHHS | Sleep Heart Health Study |
| **Cardiovascular Signals** | ECG | Electrocardiogram |
| | ABP | Arterial Blood Pressure |
| | PPG | Photoplethysmogram |
| **Sleep Signals** | EEG | Electroencephalogram |
| | EOG | Electrooculogram |
| | EMG | Electromyogram |
| **Models / Methods** | SSL | Self-Supervised Learning |
| | FM | Foundation Model |
| | DP-NeuroNet | Dual-Path NeuroNet |
| | MAE | Masked Autoencoder |
| | ViT | Vision Transformer |
| | LoRA | Low-Rank Adaptation |
| | LSCNorm | (참고용; 본 연구는 사용하지 않음) |
| | NT-Xent | Normalized Temperature-scaled Cross Entropy |
| | MSE | Mean Squared Error |
| **Statistics** | ACC | Accuracy |
| | AUC | Area Under the ROC Curve |
| | MAV | Mean Absolute Value (full-modality 대비 평균 절대 저하) |
| | ECE | Expected Calibration Error |
| **Reporting** | TRIPOD+AI | Transparent Reporting of multivariable prediction models for Individual Prognosis Or Diagnosis (AI extension) |

---

## Introduction

(작성 예정)

---

## Related Work

본 절은 PhysioME-Hetero 와 인접한 선행 연구를 세 축으로 정리하고, 각 축에서 본 연구가 갖는 차별점을 명시한다.

### Single-Modality Biosignal Foundation Models

생체신호 파운데이션 모델 연구의 다수는 단일 신호 계열에 집중되어 왔다. ECG 에서는 BIOT [@BIOT] · ECG-FM · HuBERT-ECG · ST-MEM 이, PPG 에서는 PaPaGei · GPT-PPG · Pulse-PPG 가, EEG 에서는 BENDR · LaBraM · EEGPT [@EEGPT] 등이 대규모 자기지도 사전학습으로 각 도메인 내 일반화 가능한 표현을 학습하였다. 이 단일 모달리티 FM 들은 자기 도메인 안에서 강력하지만 (a) 동시 측정된 다른 모달리티 신호에 접근하지 않으므로 cross-modal 보강을 학습할 수 없고, (b) 한 모달리티의 결측 상황에서 대체 정보를 제공하지 못한다. ICU·OR·sleep lab 의 임상 의사결정은 단일 신호가 아니라 동기 측정된 ECG·ABP·PPG·EEG·EOG 등의 *결합* 위에서 이루어지므로, 단일 모달리티 FM 은 다중 모달리티 task 의 전이에서 본질적 ceiling 을 갖는다.

### Multi-Modal Biosignal SSL with Missing Modalities

다중 모달리티 결측 학습은 본 연구의 직접 선행이며 두 설계 축으로 환원된다 — *결측을 어떻게 표현할지* 와 *결측 신호를 어떻게 복원할지*. **MultiMAE** [@MultiMAE] 와 **CroSSL** [@CroSSL] 은 입력 또는 잠재 단계에서 단일 학습 가능 mask token 으로 결측을 표현한다. **RobustSsF** [@RobustSsF] 는 시나리오별 fusion 분기와 coupled regularization 으로 dedicated-style 강건성을 구성한다. **MaskMentor** [@MaskMentor] 는 완전 입력의 teacher 와 부분 결측 입력의 student 사이의 distillation 을 사용한다. **Reza et al.** [@Reza2024] 은 동결된 사전학습 백본 위에서 parameter-efficient adaptation modulator 로 결측을 보정한다. **Wu et al.** [@Wu2024Survey] 는 분야 전반을 서베이한다. 그러나 이들 중 어느 것도 (i) 모달리티 간 복원 디코더 파라미터를 *공유하면서 센서 인지를 유지* 하거나, (ii) 임상 결측률 사전확률로 복원 손실을 *재가중* 하거나, (iii) 결측의 *원인* 을 복원의 명시적 조건으로 삼지 않는다.

### PhysioME and the Per-Modality Restoration Decoder

본 연구의 직접 baseline 인 **PhysioME** [@PhysioME] 는 NeuroNet [@NeuroNet] 의 hybrid masked-prediction + contrastive SSL 백본을 dual-path 로 확장한 **DP-NeuroNet** 위에 ViT 기반 다중모달 인코더와 *모달리티별* 복원 디코더를 결합하여, Sleep-EDFX 와 VitalDB 의 결측 모달리티 시나리오에서 single-model 베이스라인 대비 일관된 강건성을 달성하였다. 그러나 PhysioME 는 두 가지 구조적 한계를 §Discussion 에서 명시한다. 첫째, 평가가 사전 정의된 모달리티 조합으로 한정되어 새로운 조합·도메인 전이는 검증되지 않았다. 둘째, 모달리티마다 독립적인 ViT 복원 디코더를 인스턴스화하므로 디코더 파라미터 수가 모달리티 수 $M$ 에 선형 증가하며, 새로운 센서를 추가하려면 (i) 새 디코더, (ii) 재사전학습, (iii) 유사 모달리티에서 학습된 표현 공유 불가의 세 가지 비용이 동반된다.

### PhysioME-Hetero 의 위치

PhysioME-Hetero 는 PhysioME 의 DP-NeuroNet 백본과 contrastive + masked-prediction SSL 목적은 그대로 계승하면서, 위 두 한계를 정조준하는 세 가지 작은 구성 변경 — **Hetero-bucket restoration decoder**, **availability-aware reconstruction loss**, **presence embedding** — 을 도입한다. 이 변경들은 (a) 모달리티 간 복원 파라미터를 *물리적 신호 계열* 단위로 공유하면서 모달리티 type embedding 으로 센서 인지를 유지하고 (= Hetero-bucket), (b) 임상 코호트의 경험적 결측률을 손실 가중에 반영하여 자주 결측되는 모달리티의 복원을 우선 보장하며 (= availability-aware), (c) 단일 mask token 대신 결측 사유 (금기·드롭아웃·아티팩트) 를 구조화된 임베딩으로 모델에 명시적으로 신호한다 (= presence embedding). 세 변경은 독립 ablation 이 가능하도록 설계되어, 결측 강건성·확장성·메타데이터 활용 세 축 각각에서의 효과를 정량화한다.

---

## Methods

### Datasets

본 연구의 사전학습과 in-distribution downstream 평가는 PhysioME [@PhysioME] 와 동일하게 **Sleep-EDFX** [@SleepEDFX] 와 **VitalDB** [@VitalDB] 를 사용하여 직접 비교가 가능하도록 한다. 외부 코호트 일반화 검증에는 **MIMIC-IV-WDB**, **SHHS**, 또는 사내 ICU 데이터셋 가운데 한 가지를 사용할 예정이다 (v00002 에서 확정).

#### Sleep-EDFX

Sleep-EDFX [@SleepEDFX] 는 153 개 야간 polysomnography (PSG) 기록과 미국수면의학회 (AASM) 기준의 5 단계 sleep stage 라벨 (W, N1, N2, N3, REM) 을 제공한다. 본 연구는 PhysioME 와 동일하게 SC 서브셋과 EEG Fpz-Cz, EEG Pz-Oz, 수평 EOG 의 3 채널을 사용하고, 모든 신호에 0–40 Hz 대역통과 필터를 적용한다. Sleep-EDFX 는 본 연구의 *높은 결측률 EOG* 시나리오 — 옛 기록에서 EOG 가 누락된 케이스 — 의 자연스러운 코호트로 작용하며, 이는 §Pretraining 의 availability prior $\pi$ 추정에 직접 반영된다.

#### VitalDB

VitalDB [@VitalDB] 는 SNUH 가 공개한 수술 중 생리학적 기록 데이터셋으로, 일반외과·흉부외과·신경외과 등 다양한 임상 스펙트럼을 포괄하는 6,388 건의 전신마취 사례와 486,451 개의 파형 세그먼트를 포함한다. 본 연구는 PhysioME 와 동일하게 ABP, ECG, PPG 세 모달리티를 사용해 *5 분 내 수술 중 저혈압* 을 예측하는 downstream task 를 수행한다. VitalDB 는 본 연구의 *임상 금기에 의한 결측* 시나리오 — 항응고 환자의 ABP 미적용 — 를 포함하며, 이는 EOG 와는 질적으로 다른 결측 사유 (`contraindicated` vs `sensor_dropout`) 의 대비를 가능하게 하여 §Presence Embedding 의 cause embedding $\kappa^{(c)}$ 의 효용을 직접 검증할 수 있게 한다.

#### External Cohort (TBD)

PhysioME §Discussion 에서 명시된 빈틈인 *완전히 새로운 모달리티 조합 또는 도메인 전이* 를 평가하기 위해, 본 연구는 사전학습에 사용되지 않은 외부 코호트에서 leave-one-site-out 평가를 추가한다. 후보군은 (a) **MIMIC-IV-WDB** — 미국 BIDMC ICU 환경, ECG·ABP·PPG·CVP·ICP 등 가용 채널이 풍부하지만 sleep 신호 부재, (b) **SHHS** — 다기관 수면 코호트, EEG·EOG·EMG·심전도 가용, (c) **사내 ICU 데이터셋** — Sleep 과 ICU 양쪽 모달리티를 동시 보유하는 가장 강한 검증 자원이지만 IRB·DUA 일정에 따라 사용 가능 시점이 다름. v00002 에서 후보군 가운데 하나를 확정한다.

세 데이터셋의 sampling rate 는 PhysioME 와 동일하게 모달리티별 표준 (Sleep-EDFX 100 Hz, VitalDB 100 Hz / 128 Hz mixed) 으로 유지하고, 윈도우 분할·라벨링·5-fold subject-group cross-validation 의 상세 절차는 PhysioME [@PhysioME] 의 §Experiments 와 동일하다.

---

### Model Architecture

PhysioME-Hetero 는 PhysioME [@PhysioME] 의 3 단계 파이프라인 — modality encoder → multimodal encoder → restoration decoder — 을 그대로 계승하며, 세 단계 가운데 (1) multimodal encoder 의 *입력 부호화* 와 (2) restoration decoder 의 *파라미터화* 두 곳만 수정한다. 이하 §5.1 은 PhysioME 와 공유되는 표기를 기술하고, §5.2 ~ §5.4 는 PhysioME-Hetero 가 도입하는 세 변경을 차례로 정의한다.

#### Notation and Inherited Pipeline

$\mathcal{M} = \{1, \dots, M\}$ 을 모달리티 집합으로 둔다. 각 입력 윈도우마다 $\mathcal{M}_{\text{obs}} \subseteq \mathcal{M}$ 가 관측되고 $\mathcal{M}_{\text{miss}} = \mathcal{M} \setminus \mathcal{M}_{\text{obs}}$ 가 결측된다. PhysioME 와 동일하게 사전학습된 DP-NeuroNet [@NeuroNet] 인코더 $f_{\text{neuronet enc}}$ 가 모달리티 인코더 역할을 수행하여, 입력 $x^{(m)}$ 으로부터 길이 $N$ 의 토큰 시퀀스 $e^{(m)} = \{e_i^{(m)}\}_{i=1}^N$ 을 생성한다. 모달리티 인코더의 일부 레이어에는 LoRA [@LoRA] 가 적용되고 나머지는 동결된다.

각 $e^{(m)}$ 은 MLP 와 위치 인코딩 $pe$, 모달리티 인코딩 $mt^{(m)}$ 을 거쳐 $z^{(m)}$ 으로 변환된 뒤, 학습 시 random sampling 또는 drop modality 가 무작위로 적용되어 결측 시나리오를 시뮬레이션한다. 이후 ViT 기반 다중모달 인코더 $f_{\text{multimodal enc}}$ 가 모든 $z^{(m)}$ 을 통합하여 fusion 토큰 시퀀스 $o$ 를 생성한다. 학습 시점에는 모달리티 디코더 $f_{\text{modality dec}}^{(m)}$ 가 random sampling 으로 빠진 토큰을 복원하며 (intra-modal reconstruction; SSL 학습 종료 후 폐기), 복원 디코더 $f_{\text{restoration dec}}$ 는 drop modality 가 적용된 모달리티의 토큰을 복원한다 (cross-modal restoration; 추론 시 사용). 이 파이프라인은 PhysioME 와 일치하며, 본 연구의 변경은 후자의 두 단계 — 결측 표현과 복원 디코더 — 에 한정된다.

#### Hetero-Bucket Restoration Decoder

PhysioME [@PhysioME] 는 모달리티마다 독립된 ViT 기반 복원 디코더 $f_{\text{restoration dec}}^{(m)}$ ($m \in \mathcal{M}$) 를 인스턴스화한다. 이 설계는 모달리티별 출력 통계의 정밀 복원을 가능하게 하지만 두 가지 비용을 동반한다. 첫째, 디코더 파라미터 수가 $\mathcal{O}(M \cdot D_{\text{dec}}^2)$ 로 모달리티 수에 선형 증가한다 — $M = 6$ 인 본 연구 설정에서도 이미 multimodal encoder 와 비슷한 규모의 파라미터를 디코더가 차지한다. 둘째, 새로운 센서 종류 $m'$ 을 추가하려면 새 디코더 인스턴스를 생성하고 처음부터 사전학습을 다시 수행해야 하며, 동일한 신호 계열의 기존 모달리티 (예: PPG 와 새로운 부위의 PPG) 에서 학습된 복원 표현을 재사용할 경로가 존재하지 않는다.

본 연구는 모달리티를 신호 특성으로 정의되는 *버킷* (bucket) 으로 묶고, 모달리티별 디코더 대신 *버킷별 디코더* 하나를 사용한다. 버킷 할당 함수 $b(\cdot): \mathcal{M} \to \{1, \dots, K\}$ 는 (i) 신호 종류 (1D 생체전기 vs 1D 혈역학 vs 다채널 EEG …), (ii) 일반적 sampling rate, (iii) 채널 토폴로지를 기준으로 사전 큐레이션된다. 본 연구의 6 모달리티 설정은 다음과 같이 $K = 3$ 버킷으로 묶인다.

- $b = 1$ (electric 1D): EEG Fpz-Cz, EEG Pz-Oz, EOG, ECG
- $b = 2$ (hemodynamic 1D): ABP
- $b = 3$ (optical 1D): PPG

버킷 내부 모달리티는 학습 가능한 *모달리티 type embedding* $\rho^{(m)} \in \mathbb{R}^{D_\rho}$ 으로 구분되며, 동일 버킷 내 두 모달리티는 디코더 가중치를 공유하지만 type embedding 이 출력 통계의 모달리티 특이성을 보존한다. 버킷 디코더 $f_{\text{restoration dec}}^{[b]}$ 의 출력은 다음과 같이 정의된다.

$$
g^{(m)} = \mathrm{MLP}\!\left(f_{\text{restoration dec}}^{[b(m)]}\bigl(o^{(m)} \,\Vert\, \rho^{(m)}\bigr)\right),\quad m \in \mathcal{M}_{\text{miss}}
$$

여기서 $\Vert$ 는 특징 차원 결합이다. 디코더 파라미터 수는 $\mathcal{O}(M \cdot D_{\text{dec}}^2)$ 에서 $\mathcal{O}(K \cdot D_{\text{dec}}^2 + M \cdot D_\rho \cdot D_{\text{dec}})$ 로 줄어들며, $D_\rho \ll D_{\text{dec}}$ 인 본 연구 설정 ($D_{\text{dec}} = 256$, $D_\rho = 32$) 에서는 약 2 배 감소이다. 더 결정적으로, 동일 버킷에 속하는 새로운 모달리티 $m'$ 의 추가는 (a) 작은 type embedding $\rho^{(m')}$ 의 학습과 (b) 모달리티 인코더 $f_{\text{modality enc}}^{(m')}$ 의 fine-tuning 만 요구하며, 버킷 디코더 자체는 재학습 없이 재사용된다 — 즉 *zero-shot 결측 모달리티 복원* 이 동일 버킷 안에서 가능해진다.

이 설계의 핵심 가정은 *복원의 저주파 구조* 가 개별 모달리티보다 신호 계열에 의해 대체로 결정된다는 것이다. 즉 모달리티 인코더 임베딩 공간에서 결측 1 초 윈도우의 토큰 단위 복원이 어떤 모양인지는, ECG 와 EEG 처럼 서로 다른 신호 계열 사이에서는 크게 다르지만, 같은 1D 생체전기 계열 내부의 EEG Fpz-Cz 와 EEG Pz-Oz 사이에서는 대체로 공유된다. §Ablation 의 A1 (버킷 수 $K \in \{1, 3, M\}$) 은 이 가정을 직접 검증한다.

#### Availability-Aware Reconstruction Loss

PhysioME [@PhysioME] 의 missing-modality reconstruction loss $\mathcal{L}_{\text{missing recon}}$ 는 drop modality 가 적용된 모달리티 집합 $\mathcal{M} \setminus \tilde{\mathcal{M}}$ 에 대해 균등 평균을 취한다. 이는 모든 모달리티 결측 사례를 동등하게 취급한다는 것을 의미하며, 결과적으로 학습 신호는 모달리티 간 차이 없이 분배된다. 그러나 임상 배포 환경에서 모달리티 결측은 균등하지 않다 — 항응고 치료 중인 환자의 ABP 결측은 특정 환자군에서 매우 높은 빈도로 발생하는 반면, ECG 결측은 모니터링 표준이 잘 잡힌 환경에서 드물게 일어난다. 임상 결측 모달리티 모델의 유용성은 *실제 배포에서 결측되는* 모달리티에 대한 복원 정확도에 달려 있으므로, 균등 평균은 모델 표현 용량을 임상적으로 가장 필요한 곳으로 집중시키지 못한다.

본 연구는 배포 코호트에서 모달리티 $m$ 의 경험적 결측률 $\pi^{(m)} \in [0, 1]$ 로 손실을 재가중한다. 가용성 가중치 $w^{(m)}$ 는 평균 정규화로 정의된다.

$$
w^{(m)} = \frac{\pi^{(m)}}{\bar{\pi}},\qquad \bar{\pi} = \frac{1}{|\mathcal{M}|}\sum_{m \in \mathcal{M}} \pi^{(m)},\qquad \sum_{m \in \mathcal{M}} w^{(m)} = |\mathcal{M}|.
$$

PhysioME 의 missing-modality reconstruction loss 는 다음과 같이 일반화된다.

$$
\mathcal{L}_{\text{miss recon}}^{\text{avail}} = \frac{1}{|\mathcal{M}\setminus\tilde{\mathcal{M}}|}\sum_{m \in \mathcal{M}\setminus\tilde{\mathcal{M}}} w^{(m)} \cdot \frac{1}{N}\sum_{i=1}^{N} \bigl\|g_i^{(m)} - e_i^{(m)}\bigr\|_2^2.
$$

가중 정규화는 $\sum_m w^{(m)} = |\mathcal{M}|$ 를 보장하므로, $\pi$ 가 균등 ($\pi^{(m)} = c$ for all $m$) 인 한계 상황에서 $w^{(m)} \equiv 1$ 이 되어 PhysioME 의 균등 평균 손실로 환원된다. 즉 본 손실은 PhysioME 의 *strict generalization* 이다.

가용성 사전확률 $\pi$ 의 추정은 두 단계로 이루어진다. (a) Sleep-EDFX 와 VitalDB 의 학습 분할에서 모달리티 단위 결측률을 직접 집계, (b) 외부 코호트에 대해서는 동일한 절차로 사이트별 $\pi$ 를 별도 추정. 본 연구는 $\pi$ 를 학습 시점이 아닌 설정 시점의 벡터로 노출하여, 배포 사이트가 자기 코호트의 결측률로 재가중하기 위해 모델을 재사전학습할 필요가 없도록 한다 — 동일 가중치가 사이트마다 다른 $\pi$ 로 미세 조정만으로 사용 가능하다.

§Ablation 의 A2 (uniform vs. cohort-empirical vs. inverted $w$) 는 본 가중의 효과를 직접 정량화하며, 본 연구의 main figure (Figure 2) 의 핵심 비교가 된다.

#### Presence Embedding

PhysioME [@PhysioME] 와 본 분야의 다중 모달리티 결측 학습 [@MultiMAE; @CroSSL; @MaskMentor] 은 결측 모달리티의 토큰 위치를 단일 학습 가능 mask token $\mu \in \mathbb{R}^D$ 으로 치환한다. 이 단일 mask 표현은 두 가지 정보를 모델에 전달하지 않는다. 첫째, *어떤* 모달리티가 결측되었는지 — multimodal encoder 는 위치 인덱스를 통해 간접적으로 추론해야 하며, 이는 학습 가능한 모달리티 인코딩 $mt^{(m)}$ 가 일부 보완하지만 mask 자리 자체에는 모달리티 정체성이 부재하다. 둘째, *왜* 결측되었는지 — 임상에서 동일한 모달리티가 결측된 경우라도 그 사유가 (a) 환자 금기, (b) 센서 드롭아웃, (c) 일시적 모션 아티팩트, (d) 미상 가운데 무엇인지에 따라, 동시 결측되는 다른 모달리티의 경향과 복원에 활용 가능한 사전지식이 매우 다르다. 단일 mask token 은 이 두 신호를 모두 평탄화한다.

본 연구는 결측 위치에 단일 $\mu$ 대신 *presence embedding* 을 주입한다. 이는 (i) 공유 base mask $\mu_0 \in \mathbb{R}^D$, (ii) 모달리티 type embedding $\rho^{(m)} \in \mathbb{R}^D$ (§Hetero-Bucket Restoration Decoder 에서 도입한 것과 동일한 임베딩, 차원만 디코더 입력 대신 multimodal encoder 입력에 맞게 projection), (iii) 결측 사유 임베딩 $\kappa^{(c)} \in \mathbb{R}^D$ (cause $c \in \mathcal{C}$, $|\mathcal{C}| = 4$: `contraindicated`, `sensor_dropout`, `motion_artifact`, `unknown`) 의 가산으로 정의된다.

$$
\mu_{\text{pres}}^{(m, c)} = \mu_0 + \rho^{(m)} + \kappa^{(c)}.
$$

가산 형식의 동기는 PhysioME-Hetero 의 모달리티 인코딩 $mt^{(m)}$ 가 이미 가산 구조이며, $\rho$ 와 $\kappa$ 의 contribution 을 attention 단계에서 분해 가능한 형태로 유지하기 위함이다. 결측 사유 $c^{(m)}$ 는 다음 두 가지 경로로 결정된다. (a) 데이터셋 메타데이터 — VitalDB 의 환자별 anti-coagulation 기록은 ABP 결측의 `contraindicated` 라벨을 직접 제공한다; Sleep-EDFX 의 EOG 결측은 기록 연도와 장비 모델로부터 `sensor_dropout` 으로 라벨링된다. (b) 메타데이터가 부재한 경우의 *규칙 기반 자동 검출* — 평탄선 (variance < threshold) 은 `sensor_dropout`, 광대역 고주파 burst (high-frequency power 비율 > threshold) 는 `motion_artifact`, 그 외는 `unknown`. 자동 검출의 false positive 가 학습을 오염시키지 않도록, 검출된 cause 라벨은 전체 학습 샘플의 서브셋에만 적용되며 나머지는 `unknown` 으로 폴백된다.

§Ablation 의 A3 ($\mu_0$ 단독 vs $+\rho$ vs $+\rho + \kappa$) 는 cause embedding $\kappa^{(c)}$ 가 추가 학습 신호로 기능하는지를 직접 검증한다. 가설은 다음과 같다 — (a) `contraindicated` 결측은 영구적이고 환자 단위로 일관되므로 multimodal encoder 가 환자 단위 representation 을 학습하도록 유도; (b) `motion_artifact` 결측은 일시적이고 인접 윈도우에서 복구 가능하므로 시간적 인접 정보 활용을 유도; (c) `sensor_dropout` 은 두 가지의 중간이며, multimodal encoder 가 cause 별로 다른 attention 패턴을 학습할 것으로 예상한다.

---

### Pretraining

#### Inherited Two-Stage Curriculum

PhysioME-Hetero 는 PhysioME [@PhysioME] 의 사전학습 절차를 그대로 계승한다. (1) DP-NeuroNet 의 모달리티별 사전학습으로 modality encoder 를 먼저 안정화한 뒤, (2) 다중 모달리티 인코더와 (Hetero-bucket) 복원 디코더를 결합 학습한다. (1) 단계는 PhysioME 의 결과를 그대로 재사용 가능하므로 본 연구의 새로운 사전학습 비용은 (2) 단계에 한정된다. 이는 §Hetero-bucket 의 파라미터 절감과 결합되어, 본 연구가 도입하는 세 변경의 추가 학습 비용이 PhysioME 대비 명목적임을 의미한다.

#### Objective Function

전체 학습 손실은 PhysioME [@PhysioME] 의 세 항을 그대로 유지하되, missing-modality reconstruction 항을 §Availability-Aware Reconstruction Loss 의 가중 형식으로 대체한다.

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{intra recon}} + \beta\,\mathcal{L}_{\text{miss recon}}^{\text{avail}} + \gamma\,\mathcal{L}_{\text{cross contra}}.
$$

- **Intra-modality reconstruction $\mathcal{L}_{\text{intra recon}}$.** PhysioME 와 동일. random sampling 으로 빠진 토큰을 모달리티 디코더 $f_{\text{modality dec}}^{(m)}$ 가 복원하도록 MSE 로 최적화. drop modality 가 적용되지 않은 모달리티 $\tilde{\mathcal{M}}$ 에 대해서만 정의된다.
- **Availability-aware missing-modality reconstruction $\mathcal{L}_{\text{miss recon}}^{\text{avail}}$.** §Availability-Aware Reconstruction Loss 의 정의. PhysioME 의 균등 평균을 가용성 가중 평균으로 대체한 strict generalization.
- **Cross-modality contrastive $\mathcal{L}_{\text{cross contra}}$.** PhysioME 와 동일. NT-Xent [@NTXent] 기반으로 모달리티 인코더 표현 $\bar{e}^{(m)}$ 과 fusion 표현 $\bar{o}$ 사이의 정합성을 학습한다.

복원 디코더의 backpropagation 은 PhysioME 와 동일하게 stop-gradient 로 multimodal encoder 까지 차단되어, 디코더가 복원 태스크에 집중적으로 학습되도록 유지한다. 하이퍼파라미터 $\alpha, \beta, \gamma$ 의 출발점은 PhysioME 와 동일한 $1.0$ 이며, §Ablation A2 가 $\beta$ 의 민감도를 직접 평가한다.

---

### Inference

추론 시점의 절차는 PhysioME [@PhysioME] 의 2-pass 추론을 그대로 유지하되, mask token 자리에 §Presence Embedding 의 $\mu_{\text{pres}}^{(m, c)}$ 가 들어가고, 복원 디코더는 §Hetero-Bucket Restoration Decoder 의 버킷 디코더 $f_{\text{restoration dec}}^{[b(m)]}$ 가 호출된다. 의사코드는 v00002 에 추가한다. 추론 시 결측 사유 $c^{(m)}$ 는 (a) 임상 메타데이터에서 가용한 경우 직접 사용, (b) 부재 시 §Presence Embedding 의 규칙 기반 자동 검출로 결정한다.

---

## Experiments

### Setup

(작성 예정 — 5-fold subject-group cross-validation, $2^M - 1$ modality subset 평가, ablation A1/A2/A3 의 hyperparameter grid 등 v00002 에서 확정.)

### Baselines

(작성 예정 — PhysioME [@PhysioME] (주 베이스라인), MultiMAE [@MultiMAE], CroSSL [@CroSSL], RobustSsF [@RobustSsF], MaskMentor [@MaskMentor], Reza et al. [@Reza2024] (single-model 군), ContraWR [@ContraWR], SynthSleepNet [@SynthSleepNet] (dedicated-model 군).)

### Evaluation Metrics

(작성 예정 — ACC / AUC / MAV (full-modality 대비 평균 절대 저하) / 디코더 파라미터 수 / ECE.)

### Ablations

#### A1. Bucket Count $K$

버킷 수 $K \in \{1, 3, M\}$ 의 비교. $K = 1$ 은 모든 모달리티가 단일 디코더를 공유 (가장 강한 공유), $K = M$ 은 PhysioME 와 동등 (공유 없음), 큐레이션된 $K = 3$ (1D-electric / hemodynamic / optical) 이 본 연구의 권장 설정. 가설 — 큐레이션된 $K = 3$ 이 task accuracy 를 최소 손실로 유지하면서 디코더 파라미터를 약 절반으로 줄임.

#### A2. Availability Weight $w$

가용성 가중 $w^{(m)}$ 의 변형 비교. **(i) Uniform** ($w^{(m)} \equiv 1$, PhysioME 와 동등), **(ii) Cohort-empirical** (학습 분할에서 직접 추정한 $\pi^{(m)}$ 으로 정규화), **(iii) Inverted** ($w^{(m)} \propto 1 - \pi^{(m)}$, 자주 결측되는 모달리티의 손실을 *낮춤*; sanity check 로 작용). 본 ablation 이 main paper 의 Figure 2 (핵심 해석 figure) 의 비교가 된다.

#### A3. Presence Embedding Components

Presence embedding 의 점진적 구성 — **(i)** $\mu_0$ 단독 (PhysioME 와 동등), **(ii)** $\mu_0 + \rho^{(m)}$ (모달리티 정체성 추가), **(iii)** $\mu_0 + \rho^{(m)} + \kappa^{(c)}$ (cause embedding 추가). 가설 — (ii) 와 (iii) 사이의 차이가 cause embedding 의 효용을 정량화하며, 차이가 큰 결측 시나리오 (예: VitalDB 의 ABP `contraindicated` 결측) 가 cause 신호를 가장 적극적으로 활용함.

### External-Cohort Transfer

(작성 예정 — Sleep-EDFX + VitalDB 사전학습 → 외부 코호트의 leave-one-site-out evaluation; 외부 코호트는 v00002 에서 확정.)

---

## Results

(작성 예정 — v00002 에서 Table 1 (Sleep-EDFX 결측 벤치마크), Table 2 (VitalDB), Table 3 (외부 전이), Table 4 (디코더 파라미터 vs MAV trade-off), Figure 2 (A2 main figure), Figure 3 (calibration), Figure 4 (cause 별 confusion).)

---

## Discussion

(작성 예정)

---

## References

본 드래프트의 인용은 [[../library.bib]] 에 BibTeX 로 관리하며, 각 참고문헌의 PDF 사본은 [[../related_works/papers/README]] 인덱스에서 확인할 수 있다.

본 드래프트에서 사용한 1차 인용 키:

- `@PhysioME` — Lee et al., 2025. *PhysioME.* arXiv:2510.11110. (주 베이스라인)
- `@NeuroNet` — Lee et al., 2024. *NeuroNet.* arXiv:2404.17585.
- `@SynthSleepNet` — Lee et al., 2025. *SynthSleepNet.* arXiv:2502.17481.
- `@MultiMAE` — Bachmann et al., 2022. *MultiMAE.* ECCV.
- `@CroSSL` — Deldari et al., 2024. *CroSSL.* WSDM.
- `@RobustSsF` — Lee & Kim, 2023. *RobustSsF.* ML4MHD Workshop.
- `@MaskMentor` — Zhao et al., 2024. *MaskMentor.* ACM MM.
- `@Reza2024` — Reza et al., 2024. *Robust multimodal w/ parameter-efficient adaptation.* TPAMI.
- `@ContraWR` — Yang et al., 2021. arXiv:2110.15278.
- `@Wu2024Survey` — Wu et al., 2024. arXiv:2409.07825.
- `@VitalDB` — Lee H.-C. et al., 2022. *VitalDB.* Scientific Data.
- `@SleepEDFX` — Kemp et al., 2000. *Sleep-EDFX.* IEEE TBME.
- `@LoRA` — Hu et al., 2022. *LoRA.* ICLR.
- `@NTXent` — Sohn, 2016. *N-pair Loss.* NeurIPS.
- `@BIOT` — Yang et al., 2023. *BIOT.* NeurIPS.
- `@EEGPT` — Wang et al., 2024. *EEGPT.* NeurIPS.
