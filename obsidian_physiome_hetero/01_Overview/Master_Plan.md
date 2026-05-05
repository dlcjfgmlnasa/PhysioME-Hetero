---
tags: [overview, ssot]
status: locked
locked_until: paper-submission
---

# Master Plan — Single Source of Truth

> 변경 시 이 파일을 **먼저** 갱신하고, 이후 다른 wiki/code/plan 파일을 동기화한다. 직접 변경 없이 코드만 바꾸지 말 것.

## 데이터 결정 (locked, with phased modality expansion)

| 항목 | 결정 | 이유 |
|---|---|---|
| 사전학습 데이터 | **VitalDB only** | SNU Vital Lab 홈, IOH narrative 강함, modality synergy 임상적으로 명료 |
| 모달리티 — 시작 | **ABP / ECG / PPG** (3종, 7 bucket) | JBHI 1-pass 안전선 |
| 모달리티 — Step 2 | **+CO2** (4종, 15 bucket) | 호흡 ↔ 산소화 ↔ IOH/AKI mechanistic linkage. 여전히 JBHI scope |
| 모달리티 — Step 3 (조건부) | **+CVP** (5종, 31 bucket) | 정맥계 → preload → IOH 인과. **Step 2 bucket 분포 결과에 따라 진행 여부 결정** (CVP coverage ≥20% / non-degenerate spread). Venue 변경 트리거 |
| 샘플레이트 | 100 Hz | 원본 PhysioME 합치 |
| Window | 60 s | 원본 PhysioME 합치 |
| 외부 transfer | MIMIC-III Waveform DB (1회) | "단일 dataset" reviewer concern 대응. transfer는 ABP/ECG/PPG 3종으로만 (MIMIC WDB 한계) |
| **off-table** | Sleep-EDFx, MASS, SHHS, MESA, **EEG** | 2026-05-04 / 2026-05-05 명시. EEG는 cortical activity → hemodynamic SSL과 결이 다름, 별도 EEG-FM (LaBraM 등) 직접 비교 필요해 paper scope 폭발 |

## 모델 결정 (locked)

| 항목 | 결정 | 노트 |
|---|---|---|
| Backbone | BFM-derived TransformerEncoder | [[../02_Architecture/Decision_BFM_Transformer_Port]] |
| Position | RoPE only | [[../02_Architecture/Decision_RoPE_Everywhere]] |
| Phase-1 SSL | NeuroNet + TF-C | [[../02_Architecture/Decision_TFC_over_SimCLR]] |
| Phase-2 SSL | Hetero-bucket + availability-aware | [[../02_Architecture/Decision_Hetero_Bucket]] |
| Modality state | 3-state presence embedding | [[../02_Architecture/Decision_Presence_Embedding]] |
| LoRA | Hand-rolled (`models/transformer/lora.py`) | [[../02_Architecture/Decision_HandRolled_LoRA]] |

## 평가 결정 (locked)

| 항목 | 결정 |
|---|---|
| Primary downstream | IOH (MAP<65 sustained ≥1min, 5min horizon) |
| Secondary | AKI (KDIGO), Mortality (MIMIC-III) |
| Ablations | A1 (synth-only baseline) / **A2 (real vs synth missing — 핵심)** / A3 (restoration toggle) |
| Subset 평가 | N≤3: 모든 부분집합 (7개). N≥4: full + each single + 무작위 추출로 최대 10개 (`select_probe_subsets`) |
| Calibration | ECE + reliability diagram (IOH primary) |
| Seed | 최소 3개 (mean ± std) |
| Split | Subject/case-level (leakage 방지) |

## Venue 결정 (modality 확장과 연동)

| 모달 수 | Primary venue | 이유 |
|---|---|---|
| 3 modal (Step 1 그대로 멈춤) | **IEEE JBHI** | "missing-modality robustness for hemodynamic SSL" 깔끔한 scope |
| 4 modal (Step 2 도달) | **IEEE JBHI** (여전히) | CO2 추가는 mechanistic narrative 강화일 뿐 scope 확장 아님 |
| 5 modal (Step 3 도달) | **npj Digital Medicine** | "perioperative monitoring foundation model" 색채 강화. JBHI는 narrow한 single-task paper 선호 |
| 5 modal + dramatic A2 gap | **ICLR 2027** | 분포-shift 정조준 paper로 framing 가능 |

## 진행 시퀀스 (변경 금지)

```
Phase-1 NeuroNet pretraining (per-modality)
     ↓
Phase-2 PhysioME-Hetero pretraining
     ↓
A1 baseline (synth-only) pretraining ── 매칭 compute
     ↓
A3 ablation pretraining (restoration_only_on_complete=false)
     ↓
Downstream eval: IOH 7-subset × 3 seeds
     ↓
A1/A2/A3 비교
     ↓
AKI + Mortality (transfer)
     ↓
Calibration + 실패 케이스 분석
     ↓
JBHI draft
```

## 외부 데이터 의존성 (수동 다운로드 필요)

- **VitalDB raw .vital** — vitaldb.net 로그인 (case 수십 GB)
- **VitalDB clinical_data.csv + lab_data.csv** — AKI 라벨링용
- **MIMIC-III Waveform Matched Subset** — PhysioNet 인증 필요
- **MIMIC-III ICU mortality cohort CSV** — `subject_id` + `hospital_expire_flag`

## 변경 이력

| 일자 | 변경 | 이유 |
|---|---|---|
| 2026-05-04 | Sleep-EDFx 제거, VitalDB-only 결정 | IOH narrative 우선 |
| 2026-05-04 | BFM Transformer 포팅 (이전 결정 reverse) | unified codebase, RoPE quality |
| 2026-05-04 | SimCLR → TF-C | false-negative 해결, biosignal natural |
| 2026-05-05 | peft → 자체 LoRA | 의존성 제거, state-dict 정합성 |
| 2026-05-05 | Freq view zero-pad → freq_proj | structural-zero artifact 제거 |
| 2026-05-05 | Modality 확장 phased plan 추가 (3→4→5) | impact 키우기 + venue 단계적 상향. Step 3은 conditional |
| 2026-05-05 | linear_probing: SVC → LR + sampled subsets (`probe_utils`) | 4+ modal에서 2^N − 1 SVC 폭발 방지 (Step 1 선결조건) |
