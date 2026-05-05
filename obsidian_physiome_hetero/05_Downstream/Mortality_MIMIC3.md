---
tags: [downstream, mortality, transfer]
---

# Mortality — MIMIC-III External Transfer

## 목적

VitalDB로 학습한 PhysioME-Hetero를 **다른 환자 모집단** (US ICU, MIMIC-III)으로 transfer 평가. JBHI reviewer의 "single dataset" concern을 1회의 외부 transfer table로 정조준.

## 라벨

- **ICU mortality**: `hospital_expire_flag` (binary)
- 외부 cohort CSV 필요: 최소 `subject_id` + `hospital_expire_flag`

## 왜 mortality인가

- IOH/AKI는 VitalDB-specific (수술 환자) — MIMIC ICU는 *수술 후*가 아니므로 직접 매칭 X
- Mortality는 ICU population 모두에 정의되는 universal outcome
- ABP/ECG/PPG가 MIMIC WDB Matched Subset에서도 가용 → modality compatibility 충족

## 데이터 파이프라인

[[MIMIC3_WDB_Transfer]] 참조.

```
MIMIC-III WDB raw  →  mimic3_waveform_ssl.py  →  hetero npz (ABP/ECG/PPG)
                                                       ↓
ICU mortality cohort CSV  ────────────────────────  MortalityDataset
```

## Eval modes

`downstream/run_mortality.py`:

| Mode | 의미 | 학습되는 것 | Split |
|---|---|---|---|
| `zero_shot` | encoder freeze + classifier도 train data 평균 weight | nothing | 5-fold subject-level CV |
| `linear_probe` | encoder freeze + fc head만 새로 train | fc head | 80/20 train/val |

```bash
python -m downstream.run_mortality \
    --ckpt ckpt/physiome_hetero/best.pt \
    --data_root data/mimic3/npz/ \
    --cohort_csv path/to/icu_mortality_cohort.csv \
    --mode linear_probe \
    --output_dir results/mortality/
```

## Metrics

- AUROC (5-fold mean ± std)
- AUPRC

## Transfer-specific 위험

- **Sample rate mismatch**: MIMIC WDB는 보통 125 Hz, 우리는 100 Hz. parser 단계에서 resample.
- **Filter cutoff drift**: 같은 `_signal_filters.preprocess_channel` 사용하지만 환자 모집단이 다르므로 filter가 capture하는 frequency band의 임상적 의미가 다를 수 있음.
- **Modality 분포 다름**: MIMIC ICU는 PPG는 거의 항상 있고 ABP는 invasive line 가진 환자만. → real-missing 분포가 VitalDB와 다름. A2 ablation의 외부 검증 데이터로 가치.

## 원본

- 코드: `downstream/tasks/mortality.py`, `downstream/run_mortality.py`
- Parser: `dataset/data_parser/mimic3_waveform_ssl.py`
- 결정 메모: `memory/project_physiome_state.md` "External transfer"
