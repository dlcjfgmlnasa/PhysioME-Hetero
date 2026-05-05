# Plan: Data Engineering — PhysioME-Hetero

> 본 파일은 **data-engineer** 에이전트의 작업 가이드입니다. 항상 이 파일을 먼저 읽고 다음 미완료 태스크를 실행하세요.
> Single source of truth: `physiome_hetero/00_MOC.md` 와 그 하위 노트들 (특히 `01_Overview/Master_Plan.md`).

---

## 1. 프로젝트 개요

- **데이터셋**: VitalDB (1차) + MIMIC-III Waveform DB (외부 transfer 1종)
- **모달리티**: ABP / ECG / PPG @ 100 Hz, 60 s 윈도우
- **저장 포맷**: numpy `.npz` (per-record 또는 per-window) + manifest JSON
- **Phase-1 SSL**: per-modality (NeuroNet + TF-C)
- **Phase-2 SSL**: multimodal hetero (ABP/ECG/PPG 부분집합)
- **Downstream**: IOH (primary, MAP<65 sustained ≥1min within horizon) / AKI (KDIGO) / Mortality (MIMIC-III transfer)

---

## 2. 디렉터리 / 컴포넌트 현황

```
dataset/
├── data_parser/
│   ├── _quality_checks.py        # 생리학적 QC: HR/pulse/autocorr (ECG/ABP/PPG/CVP/CO2/AWP/PAP/ICP)
│   ├── _signal_filters.py        # SIGNAL_CONFIGS + preprocess_channel + segment_quality_score + resample_to_target
│   ├── vital_db.py               # 기존 lookahead-coupled SSL parser (구버전, IOH 라벨용)
│   ├── vital_db_ssl.py           # 신규 SSL 파서: lookahead 제거, modality-availability 메타데이터, hetero npz schema
│   ├── vital_db_downstream.py    # downstream용 full-signal 파서 (윈도잉 X)
│   └── mimic3_waveform_ssl.py    # wfdb-streaming MIMIC-III WDB 파서, scan→manifest, parse→npz
└── utils.py

pretrained/
├── physiome/
│   ├── data_loader.py            # 기존 모달리티 페어드 dataloader (PhysioME v1)
│   └── hetero_data_loader.py     # HeteroVitalDBDataset + BucketBatchSampler + hetero_collate_fn (v2)
└── dp_neuronet/
    ├── data_loader.py            # 단일 모달리티 SSL용
    └── augmentation.py           # (TF-C 전환 후 미사용 — 보관용)

downstream/
├── tasks/
│   ├── hypotension.py            # IOH 라벨링 (MAP<65 ≥1min) + HypotensionDataset
│   ├── aki.py                    # KDIGO Cr-based AKI 라벨링 + AKIDataset
│   └── mortality.py              # MIMIC-III hospital_expire_flag + MortalityDataset
└── data_loader.py                # 공통 wrapper
```

### 외부 데이터 의존성 (저장소 외부)
- **VitalDB raw .vital**: vitaldb.net 로그인 후 다운로드
- **VitalDB clinical_data.csv + lab_data.csv**: AKI 라벨링용 (vitaldb.net)
- **MIMIC-III Waveform Matched Subset**: PhysioNet 인증 필요
- **MIMIC-III ICU mortality cohort CSV**: 최소 `subject_id` + `hospital_expire_flag` 컬럼 필요

---

## 3. Hetero NPZ 스키마

각 case/segment별로:
```
arr_0:  ABP signal  shape (T,)  or NaN-filled if absent
arr_1:  ECG signal  shape (T,)  or NaN-filled if absent
arr_2:  PPG signal  shape (T,)  or NaN-filled if absent
mask:   (3,) bool  - True if modality present
case_id, segment_id, sampling_rate, window_sec
```

---

## 4. 데이터 컨트랙트 (다른 에이전트와 공유)

| 인터페이스 | 형태 | 비고 |
|---|---|---|
| Phase-1 SSL batch | `(B, 1, T)` float32 | T = sampling_rate × window_sec = 6000 |
| Phase-2 batch | `(B, M_present, T)` + presence_mask `(B, 3)` | M_present = bucket의 present 개수 (1~3) |
| Hetero bucket id | int ∈ {0..6} | 7-subset enumeration: 001/010/100/011/101/110/111 |
| Downstream signal | `(B, 3, T)` float32, NaN→0 | absent modality는 zero-fill, presence_mask 별도 전달 |
| Downstream label | `(B,)` int64 (binary) | IOH/AKI/Mortality 모두 binary |

---

## 5. 태스크 — 진행 상태

### 완료 (`[x]`)
- [x] **[High]** vital_db_ssl.py 작성 (lookahead 제거, modality availability 메타데이터)
- [x] **[High]** _quality_checks.py + _signal_filters.py 포팅 (BFM에서)
- [x] **[High]** vital_db_downstream.py — full-signal 파서 (no windowing)
- [x] **[Medium]** mimic3_waveform_ssl.py — wfdb-streaming, scan→parse 2-stage CLI
- [x] **[High]** hetero_data_loader.py — Bucket-aware sampler + collate
- [x] **[Medium]** downstream/tasks/{hypotension, aki, mortality}.py 포팅

### 진행 예정 (`[ ]`)

- [ ] **[High]** **VitalDB raw → SSL npz 본 실행 (1차 데이터 생성)**
  - 입력: VitalDB raw .vital 파일 (사용자 경로 입력 필요)
  - 출력: `data/vital_db_ssl/<case_id>.npz` + `manifest.json`
  - 의존성: vital_db_ssl.py, _quality_checks.py, _signal_filters.py
  - 참고: 7개 bucket 분포 통계를 manifest에 기록 (bucket 분포가 균형 안 맞으면 sampler 가중치 조정 필요)

- [ ] **[High]** **Downstream IOH 라벨링 본 실행**
  - 입력: VitalDB raw .vital
  - 출력: `data/downstream/ioh/<case_id>.npz` (signal full + label list)
  - 의존성: vital_db_downstream.py + downstream/tasks/hypotension.py
  - 참고: window=60s × horizon=5min sweep 결과 manifest 기록

- [ ] **[High]** **AKI 라벨링 (clinical_data.csv + lab_data.csv 받은 후)**
  - 입력: VitalDB clinical_data.csv, lab_data.csv, raw .vital
  - 출력: `data/downstream/aki/<case_id>.npz`
  - 의존성: downstream/tasks/aki.py
  - 참고: postop window opend < dt ≤ opend + 7days. Stage 1/2/3 + binary 모두 라벨 보관.

- [ ] **[Medium]** **MIMIC-III Mortality cohort 매칭**
  - 입력: PhysioNet 인증 후 다운받은 ICU mortality cohort CSV + WDB raw
  - 출력: `data/mimic3/mortality/<subject_id>_<rec>.npz`
  - 의존성: mimic3_waveform_ssl.py + downstream/tasks/mortality.py
  - 참고: subject-level 5-fold split index도 함께 저장.

- [ ] **[Medium]** **Hetero bucket 분포 분석 / 가중치 결정**
  - 입력: SSL manifest.json (위 첫 task 결과)
  - 출력: bucket별 sample 수 + class-weight (또는 oversampling factor) JSON
  - 의존성: hetero_data_loader.py의 BucketBatchSampler가 이를 읽도록 옵션 추가
  - 참고: 자연스러운 분포가 너무 unbalanced면 (예: 111 bucket이 95%면) — A2 ablation의 "real-missing" 데이터가 부족해질 수 있음. 그 경우 minor bucket oversampling 정책 결정 필요.

- [ ] **[Low]** **vital_db.py (legacy) 제거 또는 archive**
  - 현재 vital_db_ssl.py + vital_db_downstream.py로 대체됨. legacy import 사용처 확인 후 제거.

---

## 6. Quality Standards

- 새 parser/dataset 작성 시 다음 항목 docstring에 명시:
  - 입력 파일 경로 패턴 / 출력 npz key 스키마 / tensor shape (`(B, C, T)`) / sampling rate / window length
- NaN/Inf 방지: `_signal_filters.preprocess_channel` 거친 후 `np.isfinite().all()` 체크
- 윈도우당 quality score (`segment_quality_score`) + domain QC (`_quality_checks`) 모두 통과한 윈도우만 SSL에 포함
- 메모리: VitalDB는 케이스당 수십 MB, lazy load (mmap 또는 generator) 권장

---

## 7. 다음 액션

`@data-engineer`를 호출할 때 가장 먼저 처리할 태스크는 **VitalDB raw → SSL npz 본 실행** 입니다 (Section 5의 첫 미완료 항목). 사용자에게 raw `.vital` 파일 경로와 출력 디렉터리 경로를 확인하세요.
