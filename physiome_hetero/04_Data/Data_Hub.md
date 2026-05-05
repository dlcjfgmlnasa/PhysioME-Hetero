---
tags: [hub, data]
---

# 04 Data — Hub

> 데이터 파이프라인·전처리·QC.

## 파이프라인 개요

```
VitalDB raw .vital                      MIMIC-III WDB raw
    │                                       │
    ▼                                       ▼
vital_db_ssl.py (SSL)                   mimic3_waveform_ssl.py
vital_db_downstream.py (label)              (transfer-only)
    │                                       │
    │  공유: _quality_checks.py             │
    │        _signal_filters.py             │
    │                                       │
    ▼                                       ▼
hetero npz (ABP/ECG/PPG + mask)        hetero npz
    │
    │
    ├── Phase-1: dp_neuronet/data_loader.py        (single modality)
    └── Phase-2: physiome/hetero_data_loader.py    (bucket-aware)
                  └── BucketBatchSampler
                  └── HeteroVitalDBDataset
                  └── hetero_collate_fn
```

## 핵심 노트

- [[VitalDB_SSL]] — VitalDB SSL/downstream parser, lookahead 제거 결정
- [[Hetero_NPZ_Schema]] — 7-bucket bitmap 스키마, mask 컨벤션
- [[MIMIC3_WDB_Transfer]] — wfdb-streaming 파서, mortality cohort

## 외부 데이터 의존성 (수동 다운로드)

| 데이터 | 출처 | 필요 시점 |
|---|---|---|
| VitalDB raw `.vital` | vitaldb.net (로그인) | Phase-1, Phase-2, IOH downstream |
| VitalDB `clinical_data.csv` | vitaldb.net | AKI 라벨링 |
| VitalDB `lab_data.csv` | vitaldb.net | AKI 라벨링 (Cr 측정값) |
| MIMIC-III Waveform Matched Subset | physionet.org (인증 필요) | External transfer |
| MIMIC-III ICU mortality cohort CSV | (사용자 보유) | Mortality 라벨 (`subject_id` + `hospital_expire_flag`) |

## Quality control

`dataset/data_parser/_quality_checks.py` — 생리학적 QC:
- HR / pulse rate / autocorrelation 체크
- 지원 channel: ECG, ABP, PPG, CVP, CO2, AWP, PAP, ICP

`dataset/data_parser/_signal_filters.py` — 전처리 chain:
- `SIGNAL_CONFIGS` — modality별 전처리 파라미터
- `preprocess_channel(...)` — range → spike → median → notch → bandpass/lowpass
- `segment_quality_score(...)` — 윈도우당 품질 점수
- `resample_to_target(...)` — fs 통일

## Plan 파일

- `.plans/.agent_plan/plan_data.md` — data engineer agent의 진행 가이드 (현재 미완료 6개 task)

## 원본

- 코드: `dataset/data_parser/`, `pretrained/.../data_loader.py`
- BFM에서 surgical pull: `memory/project_physiome_state.md` "Surgical pull from references/Biosignal-Foundation-Model"
