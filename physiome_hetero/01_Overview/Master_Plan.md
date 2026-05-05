---
tags: [overview, ssot]
status: locked
locked_until: paper-submission
---

# Master Plan — Single Source of Truth

> 변경 시 이 파일을 **먼저** 갱신하고, 이후 다른 wiki/code/plan 파일을 동기화한다. 직접 변경 없이 코드만 바꾸지 말 것.

## 데이터 결정 (locked)

| 항목 | 결정 | 이유 |
|---|---|---|
| 사전학습 데이터 | **VitalDB only** | SNU Vital Lab 홈, IOH narrative 강함, modality synergy 임상적으로 명료 |
| 모달리티 | **ABP / ECG / PPG** (3종) | VitalDB에서 함께 측정 가능, 생리적 상관 명확 |
| 샘플레이트 | 100 Hz | 원본 PhysioME 합치 |
| Window | 60 s | 원본 PhysioME 합치 |
| 외부 transfer | MIMIC-III Waveform DB (1회) | "단일 dataset" reviewer concern 대응 |
| **off-table** | Sleep-EDFx, MASS, SHHS, MESA, EEG | 2026-05-04 명시적 제거 |

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
| Subset 평가 | 7개 모달 부분집합 모두 (1~3개) |
| Calibration | ECE + reliability diagram (IOH primary) |
| Seed | 최소 3개 (mean ± std) |
| Split | Subject/case-level (leakage 방지) |

## Venue 결정

1. **IEEE JBHI** (1순위) — IF ~7.7, rolling submission
2. **npj Digital Medicine** (대안) — methodology 강하면
3. **ICLR 2027** (stretch) — A2 gap이 dramatic할 경우

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
