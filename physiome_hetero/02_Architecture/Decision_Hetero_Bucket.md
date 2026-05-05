---
tags: [decision, architecture, paper-core]
date: 2026-05-04
status: locked
---

# Decision: Hetero-bucket SSL training

## 결정

학습 데이터부터 ABP/ECG/PPG의 **임의 부분집합 (7 buckets)**을 노출한다. 원본 PhysioME처럼 "complete-only로 학습 + testtime synth drop"이 아닌, **bucket-aware sampling으로 매 batch마다 다른 modality availability**를 본다.

7개 bucket: `001 (PPG only)`, `010 (ECG only)`, `100 (ABP only)`, `011 (ECG+PPG)`, `101 (ABP+PPG)`, `110 (ABP+ECG)`, `111 (all)`.

## Why

원본 PhysioME 학습/평가 setup의 본질적 문제:

```
학습 분포: P(modality_set = {ABP, ECG, PPG}) = 1.0
평가 분포: random uniform drop → e.g., P({ABP}) = 1/7
         → 학습 분포에 없던 입력 → OOD 일반화 실패
```

이건 sampling 문제가 아닌 **distribution shift**. random drop으로 robustness를 측정하면 가짜 robustness — 실제 임상에서는 modality 결손이 환자/수술/장비별로 *구조적으로 결정*되며 (특정 수술방에는 ABP 라인이 없다든가), distribution이 다르다.

**해결**: 학습 시점부터 hetero bucket을 노출하면 모델이 7가지 availability 분포 모두에 대한 marginal posterior를 학습 → testtime 분포가 어떻든 OOD가 아님.

## How to apply

코드 위치: `pretrained/physiome/hetero_data_loader.py`
- `HeteroVitalDBDataset` — 각 segment마다 modality availability 메타데이터 보존
- `BucketBatchSampler` — bitmap별로 bucket 분리, 매 batch는 single bucket
- `hetero_collate_fn` — bucket내 sample들의 텐서 정렬

학습 시 매 batch는 동일 bucket의 샘플들로 구성됨 (혼합 batch는 v1에서 안 함). v2 ablation으로 per-sample modality mask 검토 가능.

## 위험 / 주의

- **Bucket 분포 불균형**: 자연스러운 VitalDB 분포에서 `111` bucket이 압도적이면 (95%+) Ablation A2의 "real-missing" 데이터가 부족해진다. → 첫 SSL 데이터 생성 후 bucket 분포 측정, 필요 시 minor bucket oversampling.
- **Single-modal bucket의 cross-contrastive**: `001/010/100` bucket에서 cross-contrastive loss가 의미 약함 (fusion vs unimodal이 사실상 동일 입력). 학습 신호 약함을 인지하되 코드는 그대로 (원본 PhysioME 합치).

## 검증 (smoke)

`experiments/smoke_test_hetero.py` 통과 (2026-05-04, 2026-05-05 재실행):
- 7 bucket 모두 노출 ✅
- presence states {0=REAL, 2=NATURALLY_ABSENT} 모두 출현 ✅
- restoration loss는 `111` bucket에서만 fire (다른 bucket은 0) ✅
- gradient flow 정상 (presence_state_embed 0.57, dropped_modality_token 14.6)

## 원본

- 결정 메모: `memory/project_physiome_state.md` "Method decision (2026-05-04)"
- 코드: `pretrained/physiome/hetero_data_loader.py`, `models/physiome/model.py`
- Method 노트: [[../90_Paper/PhysioME-Hetero/03. Method - Hetero-bucket]]
