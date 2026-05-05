---
tags: [data, vitaldb]
---

# VitalDB SSL Parser

## 두 가지 parser

| 파일 | 용도 | Lookahead 요구 |
|---|---|---|
| `dataset/data_parser/vital_db.py` | **Legacy** — IOH 라벨용 (lookahead 2분 필요) | ✅ |
| `dataset/data_parser/vital_db_ssl.py` | **SSL pretraining** — 라벨 없이 가능한 모든 윈도우 | ❌ |
| `dataset/data_parser/vital_db_downstream.py` | **Downstream** — full-signal (windowing 안 함) | — |

## Lookahead 제거 결정

원본 `vital_db.py::extract_sample`은 *2분 lookahead window가 valid한 윈도우만* 보존 (IOH 라벨링 전제). SSL pretraining에서는 이게 unnecessary discard:

- IOH 라벨이 없는 데이터도 representation 학습에는 유효
- 실제 데이터 양이 줄어 SSL scale이 손상

→ `vital_db_ssl.py`는 lookahead 검사 제거. 동일 윈도우의 modality availability 메타데이터만 기록.

## 전처리 chain (`_signal_filters.preprocess_channel`)

modality별로 적용:

```
range filter → spike filter → median filter → notch filter → bandpass/lowpass
```

각 modality의 cutoff/order/notch freq는 `SIGNAL_CONFIGS` table에서 가져옴.

검증: 윈도우당 NaN/Inf 검사 + `segment_quality_score` 임계값 + `_quality_checks` domain QC (HR / pulse rate / autocorr) 모두 통과한 윈도우만 출력.

## 출력 스키마

→ [[Hetero_NPZ_Schema]] 참조

## CLI 사용

```bash
python -m dataset.data_parser.vital_db_ssl \
    --raw_dir <vital_files_root> \
    --out_dir data/vital_db_ssl/ \
    --window_sec 60 --target_fs 100
```

(인자명은 코드 확인 — 작성 시점 스냅샷)

## 원본

- 코드: `dataset/data_parser/vital_db_ssl.py`, `vital_db.py`, `vital_db_downstream.py`
- 전처리 헬퍼: `dataset/data_parser/_signal_filters.py`, `_quality_checks.py`
- 결정 메모: `memory/project_physiome_state.md` "SSL data parser caveat"
