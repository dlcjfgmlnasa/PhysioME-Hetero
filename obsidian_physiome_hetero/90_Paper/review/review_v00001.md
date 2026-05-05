---
tags: [review, work-log, physiome]
version: v00001
date: 2026-05-05
status: in-progress
---

# Review v00001 — 90_Paper 위키 재정비 작업 로그

> 본 노트는 PhysioME (arXiv:2510.11110) 원본 정리와 후속 논문 작업 환경 구축을 위한 첫 번째 드래프트이다. 작성일: 2026-05-05.

## 1. 작업 개요

PhysioME 원본 논문(arXiv:2510.11110v1)의 핵심 자료를 한국어로 정리하고, 후속 논문 작성을 위한 참고문헌 아카이브와 위키 인덱스를 재구성했다. 기존에 존재하던 PhysioME-Hetero / PhysioME-Original 노트군은 모두 제거하고 단일 한국어 번역본으로 통합했다.

## 2. 위키 구조 변경

### 2.1 삭제

- `90_Paper/PhysioME-Hetero/` (총 13개 노트) — 후속 논문 초안 노트군. 본 드래프트 체계로 대체.
- `90_Paper/PhysioME-Original/` (총 8개 노트) — 영문 원본의 섹션별 분할 노트. 단일 한국어 번역본 [[../PhysioME-Original-KR]] 으로 대체.

### 2.2 신규/갱신

| 파일 | 역할 |
|---|---|
| `90_Paper/Paper_Hub.md` | 위키 허브. baseline 인용·외부 링크·BibTeX·디렉터리 안내만 유지하도록 정리. |
| `90_Paper/PhysioME-Original-KR.md` | arXiv:2510.11110v1 PDF 전문의 한국어 번역. Abstract → Introduction → Methods → Experiments → Results → Discussion → Appendix → References 구조. |
| `90_Paper/related_works/papers/` | 원본 논문의 References 20건 PDF 아카이브. |
| `90_Paper/related_works/papers/README.md` | PDF 아카이브 인덱스. 다운로드 완료 13건 / 미수집 7건 정리. |

## 3. PhysioME 원본 한국어 번역 핵심 정리

### 3.1 한 줄 요약

생리신호의 결측 모달리티에 대한 robust single-model SSL 프레임워크. **DP-NeuroNet** 백본 + **복원 디코더(restoration decoder)** + **대조-마스크 결합 손실** 을 통해, 다양한 결측 시나리오에서 dedicated-model 과 동등하거나 우수한 성능을 달성.

### 3.2 핵심 구성요소

1. **DP-NeuroNet** — NeuroNet (Lee et al., 2024) 을 가중치 공유 dual-path 로 확장. 두 경로에 서로 다른 시간 증강을 적용하여 모달리티별 시간 동역학을 풍부하게 학습.
2. **다중모달 인코더** — ViT 기반. positional + modality encoding 후 random sampling / drop modality 를 통해 결측 시나리오를 학습 시 시뮬레이션.
3. **모달리티 디코더** — 모달리티별 ViT 디코더로 random sampling 으로 빠진 토큰을 복원. SSL 학습 종료 후 폐기.
4. **복원 디코더 (핵심 기여)** — drop modality 가 적용된 모달리티의 토큰을 모달리티 인코더 출력으로 복원. **stop-gradient** 로 격리되어 복원 태스크에 집중. 추론 시 이 디코더가 결측 모달리티 토큰을 예측해 다중모달 인코더에 다시 입력.

### 3.3 손실 구성

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{intra recon}} + \beta\,\mathcal{L}_{\text{missing recon}} + \gamma\,\mathcal{L}_{\text{cross contra}}
$$

- $\mathcal{L}_{\text{intra recon}}$: 관측 모달리티 내부의 token reconstruction (MSE).
- $\mathcal{L}_{\text{missing recon}}$: 복원 디코더가 drop 된 모달리티의 인코더 출력을 복원 (MSE).
- $\mathcal{L}_{\text{cross contra}}$: 모달리티 인코더 표현 $\bar{e}^{(m)}$ 과 fusion 표현 $\bar{o}$ 사이의 NT-Xent 대조.

### 3.4 실험 결과 핵심

- **데이터셋**: Sleep-EDFX (수면 단계 분류, 5클래스), VitalDB (수술 중 5분 내 저혈압 예측).
- **결측 강건성**: full-modality 대비 평균 절대 차이가 baseline 대비 모두 가장 작음. Sleep-EDFX 에서 ACC -3.97% / AUC -2.04, VitalDB 에서 ACC -3.54% / AUC -5.43.
- **Single vs Dedicated**: single-model 인 PhysioME 가 dedicated-model (ContraWR, SynthSleepNet) 과 동등 이상의 성능. AUC 기준 거의 모든 시나리오에서 ContraWR 상회.
- **절제 연구**: DP-NeuroNet > 원본 NeuroNet > CNN > ViT 류. 복원 디코더 stop-gradient 효과 확인. restoration token 전략 > masked / memory token 전략. 복원 디코더 dim 512 / depth 8 최적.

### 3.5 한계

1. 사전 정의된 모달리티 조합 외 일반화 미검증 (entirely new combinations / domain transfer).
2. DP-NeuroNet + 복원 디코더 구조의 학습 비용. 새 모달리티마다 복원 디코더를 추가해야 하는 확장성 이슈.

## 4. 참고문헌 아카이브 현황

원본 논문 References 20건 중 **13건 다운로드 완료**, **7건 미수집(유료 저널)**. 상세 내역은 [[../related_works/papers/README]] 참고.

다운로드 완료된 핵심 baseline (인용 시 자주 사용 예상):

- **NeuroNet** (Lee et al., 2024) — DP-NeuroNet 의 base. arXiv:2404.17585.
- **SynthSleepNet** (Lee et al., 2025) — PhysioME 의 직접 전신. arXiv:2502.17481.
- **MultiMAE** (Bachmann et al., 2022) — 다중모달 MAE 베이스라인. arXiv:2204.01678.
- **CroSSL** (Deldari et al., 2024) — latent masking SSL 베이스라인. arXiv:2307.16847.
- **MaskMentor** (Zhao et al., 2024) — 마스크 자기교육 베이스라인. *유료 (ACM MM 2024)*.
- **ContraWR** (Yang et al., 2021) — sleep stage dedicated-model. arXiv:2110.15278.
- **VitalDB** (Lee H.-C. et al., 2022) — 데이터셋 인용. Scientific Data, OA.
- **Missing modality survey** (Wu et al., 2024) — 분야 정의/포지셔닝. arXiv:2409.07825.
- **Robust multimodal w/ parameter-efficient adaptation** (Reza et al., 2024) — TPAMI, missing-modality 최신 SOTA. arXiv:2310.03986.

미수집 7건은 추후 기관 라이선스로 보충 필요 (Esteva 2019, Iranfar 2021, Kemp 2000, Lee J. & Kim 2023, Littmann 2021, Puckett 2003, Zhao 2024).

## 5. 후속 작업 제안 (TODO)

### 5.1 본 위키에서 확장할 항목

- [ ] **요약 노트**: `90_Paper/related_works/summary/` 에 baseline 별 한 줄 요약 + Method 도식 + 본 논문과의 차이점 정리.
- [ ] **미수집 PDF 보충**: Esteva, Iranfar, Kemp, Lee J., Littmann, Puckett, Zhao 7건을 기관 프록시로 다운로드.
- [ ] **BibTeX 자동화**: `library.bib` 에 위 13건 entry 추가.

### 5.2 후속 논문 방향 (PhysioME 의 한계 → 차별화 포인트)

원본 논문 §Discussion 의 두 한계가 후속 작업의 자연스러운 출발점이다.

1. **이종(heterogeneous) 모달리티 일반화**
   - 원본은 사전 정의된 (EEG×3, ABP/ECG/PPG) 조합만 평가.
   - 새로운 센서 조합 / 도메인 (예: wearable, ICU, OR) 에 대한 zero-shot · few-shot 전이 검증이 필요.
   - 키 아이디어 후보: 모달리티 토큰 공간을 *type-conditioned* 으로 임베딩 → 신규 모달리티 추가 시 복원 디코더를 처음부터 재학습할 필요 없도록 함.

2. **확장성 (scalability)**
   - 모달리티 수 $M$ 마다 복원 디코더 인스턴스가 추가되어 파라미터·학습 시간이 선형 증가.
   - **Hetero-bucket** / *modality-shared restoration* 등의 통합 디코더 설계로 $M$ 에 sub-linear 한 비용을 달성하는 것이 목표.

3. **결측 의미 강화 (presence-aware)**
   - 단순 mask token 대신 *presence embedding* 으로 어떤 모달리티가 어떤 이유로 결측인지 모델에 명시적으로 신호.
   - *availability-aware* 손실 가중으로 임상적으로 자주 결측되는 모달리티 (예: ABP) 의 복원 품질을 우선 보장.

이 세 방향은 5.1 의 요약 작업과 함께 다음 드래프트(`draft_v00002.md`) 에서 method/experiment skeleton 으로 구체화한다.

## 6. 변경 로그 (이번 작업)

- 2026-05-05  
  - **삭제**: `PhysioME-Hetero/`, `PhysioME-Original/` 노트군 (총 21 노트).  
  - **재작성**: `Paper_Hub.md` (broken wikilink 제거, baseline·외부참조 중심으로 단순화).  
  - **신규**: `PhysioME-Original-KR.md` (PDF 전문 한국어 번역).  
  - **신규**: `related_works/papers/{01..19}*.pdf` 13건 + `papers/README.md` (다운로드 인덱스).  
  - **신규**: `draft/draft_v00001.md` (본 문서).

## 7. 참조

- 원본 PDF: `obsidian_physiome_hetero/2510.11110v1.pdf`
- 한국어 번역: [[../PhysioME-Original-KR]]
- 위키 허브: [[../Paper_Hub]]
- 참고문헌 아카이브: [[../related_works/papers/README]]
