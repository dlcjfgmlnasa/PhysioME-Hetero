---
tags: [data, mimic3, transfer]
---

# MIMIC-III WDB External Transfer

## 목적

JBHI reviewer의 "single dataset" concern 대응. VitalDB와 다른 환자 모집단(US ICU vs SNU 수술실)에서 transfer 성능을 1회 측정.

## 데이터셋

- **MIMIC-III Waveform Database — Matched Subset** (PhysioNet, 인증 필요)
- 사용 modality: ABP, ECG, PPG (VitalDB와 호환)
- Resolution mismatch: MIMIC-III WDB는 fs 가변 (보통 125 Hz) → 우리 100 Hz로 resample

## Parser

`dataset/data_parser/mimic3_waveform_ssl.py` (wfdb 사용, streaming):

```bash
# Stage 1: scan
python -m dataset.data_parser.mimic3_waveform_ssl scan \
    --root <wdb_root> \
    --out_manifest data/mimic3/manifest.json

# Stage 2: parse
python -m dataset.data_parser.mimic3_waveform_ssl parse \
    --manifest data/mimic3/manifest.json \
    --out_dir data/mimic3/npz/
```

같은 전처리 chain (`_signal_filters.preprocess_channel`) 사용 → 같은 hetero npz schema 출력.

## Mortality cohort

ICU mortality 라벨 매칭:
- **외부 CSV 필요**: `subject_id` + `hospital_expire_flag` 최소 2개 컬럼
- 코드: `downstream/tasks/mortality.py::MortalityDataset`
- Subject-level split (5-fold CV) — `subject_id` 기준 leakage 방지

## Eval modes

`downstream/run_mortality.py`:

| Mode | 의미 | Split |
|---|---|---|
| `zero_shot` | encoder freeze + linear classifier weights도 0-shot 추론 (혹은 averaged train classifier) | 5-fold CV |
| `linear_probe` | encoder freeze, fc head만 train | 80/20 train/val |

## 원본

- 코드: `dataset/data_parser/mimic3_waveform_ssl.py`, `downstream/tasks/mortality.py`, `downstream/run_mortality.py`
- 결정: `memory/project_physiome_state.md` "Reviewer-disarming experimental design"
