---
tags: [data, schema]
---

# Hetero NPZ Schema

> Phase-1 / Phase-2 SSL과 downstream에서 공유하는 npz 포맷.

## 구조

각 case (또는 record)당 1개 `.npz` 파일:

```
arr_0:  ABP signal       shape (n_seg, T)  float32  — modality 0
arr_1:  ECG signal       shape (n_seg, T)  float32  — modality 1
arr_2:  PPG signal       shape (n_seg, T)  float32  — modality 2
mask:                    shape (n_seg, 3)  bool     — True if modality present
modal_names:             shape (3,)        unicode  — ['ABP', 'ECG', 'PPG']
subject_modality_set:    shape (3,)        bool     — case-level availability (segment-level의 union)
case_id:                 scalar            unicode
sampling_rate:           scalar            int      — 100
window_sec:              scalar            int      — 60
```

`T = sampling_rate * window_sec = 100 * 60 = 6000`.

## Mask 컨벤션

`mask[s, m]`:
- `True` — segment `s`의 modality `m`이 측정/유효 (값 사용 가능)
- `False` — naturally absent (해당 modality measure 자체가 없음) **또는** quality check 실패

False 슬롯의 signal 값은 **0으로 채워서** 저장 (모델 forward 시 modality가 없는 경우 zero-fill 입력).

## 7개 bucket bitmap

각 segment의 modality bitmap을 binary string으로 표현:

| bitmap | ABP | ECG | PPG | bucket name |
|---|---|---|---|---|
| `001` | ❌ | ❌ | ✅ | PPG only |
| `010` | ❌ | ✅ | ❌ | ECG only |
| `100` | ✅ | ❌ | ❌ | ABP only |
| `011` | ❌ | ✅ | ✅ | ECG+PPG |
| `101` | ✅ | ❌ | ✅ | ABP+PPG |
| `110` | ✅ | ✅ | ❌ | ABP+ECG |
| `111` | ✅ | ✅ | ✅ | complete |

`HeteroVitalDBDataset.segment_bitmap_keys` 가 segment별 bitmap을 보관 → `BucketBatchSampler`가 bucket별로 sample 리스트 분리.

## Subject vs segment mask

- `subject_modality_set` — 환자/case 전체에서 modality가 한 번이라도 측정되었는가 (case-level)
- `mask[s]` — 특정 segment에서 valid한가 (segment-level)

차이 예: 환자가 ABP 라인을 수술 30분 후 분리한 경우 → `subject_modality_set[ABP] = True`, 하지만 후반 segment의 `mask[s, ABP] = False`.

## 사용 예 (Phase-2)

```python
from pretrained.physiome.hetero_data_loader import HeteroVitalDBDataset, BucketBatchSampler, hetero_collate_fn

ds = HeteroVitalDBDataset(npz_paths, eager=False, normalize=True)
sampler = BucketBatchSampler(ds.segment_bitmap_keys, batch_size=64, sampling='uniform', min_bucket_size=64)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=hetero_collate_fn)

for batch in loader:
    data            = batch['data']             # Dict[modal, (B, T)]   present만
    presence_state  = batch['presence_state']   # (B, 3)
    bucket_pattern  = batch['bucket_pattern']   # str e.g. '110'
    ...
```

## 원본

- 코드: `pretrained/physiome/hetero_data_loader.py::HeteroVitalDBDataset`, `MODAL_ORDER`
- 생성: `dataset/data_parser/vital_db_ssl.py`, `mimic3_waveform_ssl.py`
