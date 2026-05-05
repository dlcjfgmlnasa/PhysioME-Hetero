---
tags: [downstream, aki, secondary]
---

# AKI — KDIGO-based Secondary Task

## 임상 정의

**Acute Kidney Injury (AKI)** — KDIGO 기준 (Cr 변화 기반):
- Stage 1: Cr ≥ 1.5 × baseline 또는 ≥ 0.3 mg/dL 증가 (48h 내)
- Stage 2: Cr ≥ 2.0 × baseline
- Stage 3: Cr ≥ 3.0 × baseline 또는 절대값 ≥ 4.0 mg/dL

본 연구는 **postop AKI**: 수술 종료 (`opend`) 시점부터 **opend < dt ≤ opend + 7 days** 윈도우 내 발생.

## 왜 secondary인가

- IOH의 다음 단계 outcome (IOH → AKI 인과 연결 임상적으로 잘 알려짐)
- **외부 CSV 라벨 의존** — 코드 외부 데이터 필요 (`clinical_data.csv` + `lab_data.csv`)
- IOH보다 더 imbalanced — AUPRC가 main metric

## 외부 데이터 의존성

**필수 다운로드** (vitaldb.net):
- `clinical_data.csv` — case별 metadata (`opend`, `subject_id`, baseline Cr 등)
- `lab_data.csv` — 시계열 lab 측정 (`Cr`, 측정 시각)

→ 두 CSV를 case_id로 join한 뒤 KDIGO 룰을 적용해 binary 라벨 + stage 라벨 생성.

## 라벨링 코드

`downstream/tasks/aki.py::AKIDataset`:
- BFM에서 surgical pull
- KDIGO Cr-based labeling
- Stage 1/2/3 + binary (Stage ≥ 1) 모두 보관
- Postop window: `opend < dt ≤ opend + 7 days`

## Eval

`downstream/run_aki.py`:

```bash
python -m downstream.run_aki \
    --ckpt ckpt/physiome_hetero/best.pt \
    --data_root data/downstream/aki/ \
    --clinical_csv path/to/clinical_data.csv \
    --lab_csv path/to/lab_data.csv \
    --output_dir results/aki/ \
    --seeds 42 123 2024
```

## Metrics

- **AUPRC** (main metric — 매우 imbalanced)
- AUROC (보조)
- Sens@Sp90 (clinical operating point)

## 7-subset gradient

AKI는 long-term outcome이라 modality 결손에 대한 robustness가 IOH보다 demanding. 단일 modal에서 성능 떨어짐 정상.

## 결정

- [[../01_Overview/Master_Plan]] — secondary task 위치 결정

## 원본

- 코드: `downstream/tasks/aki.py`, `downstream/run_aki.py`
- 외부 데이터: `clinical_data.csv`, `lab_data.csv` (vitaldb.net)
