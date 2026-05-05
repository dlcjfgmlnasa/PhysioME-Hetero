---
tags: [hub, downstream]
---

# 05 Downstream — Hub

> Frozen encoder + classifier head 평가, 7-subset modality robustness, Ablation 3종.

## Task 매트릭스

| Task | 데이터 | 라벨 | 메트릭 | 코드 | Wiki |
|---|---|---|---|---|---|
| **IOH** (primary) | VitalDB | MAP<65 sustained ≥1min, 5min horizon | AUROC / AUPRC / Sens@Sp90 | `downstream/run_ioh.py` | [[IOH_Primary]] |
| **AKI** (secondary) | VitalDB + clinical/lab CSV | KDIGO Cr-based, postop 7d | AUROC / AUPRC | `downstream/run_aki.py` | [[AKI_KDIGO]] |
| **Mortality** (transfer) | MIMIC-III WDB | hospital_expire_flag | AUROC, 5-fold CV | `downstream/run_mortality.py` | [[Mortality_MIMIC3]] |

## Ablation (must-haves for JBHI)

| ID | 비교 | 핵심 질문 | Wiki |
|---|---|---|---|
| **A1** | Synth-only PhysioME vs Hetero-train | Hetero-train 자체의 효과 | [[Ablations_A1_A2_A3]] |
| **A2** | real-missing 하위 vs synth-missing 하위 | OOD-attack counter (논문 핵심 figure) | [[Ablations_A1_A2_A3]] |
| **A3** | restoration_only_on_complete: true vs false | Loss decomposition이 restoration 성능을 해치지 않는가 | [[Ablations_A1_A2_A3]] |

## 7-subset modality robustness

모든 downstream 평가는 **7개 modality subset**에서 수행:
- {ABP}, {ECG}, {PPG} — single modal
- {ABP+ECG}, {ABP+PPG}, {ECG+PPG} — pairs
- {ABP+ECG+PPG} — complete

`downstream/run_ioh.py` 등이 이미 7-subset sweep 자동화. 보고는 7행 표 (subset × metric) + 마지막에 평균.

## Calibration (JBHI 가산점)

`downstream/calibration.py`:
- ECE (Expected Calibration Error)
- Reliability diagram
- Saved preds (`--save_preds` flag) 후처리
- Temperature scaling 전/후 비교 권장

→ [[Calibration]]

## 핵심 노트

- [[IOH_Primary]] — primary downstream 정의, 라벨링 (MAP<65 sustained 1min)
- [[AKI_KDIGO]] — KDIGO Cr-based AKI 라벨링
- [[Mortality_MIMIC3]] — MIMIC-III transfer
- [[Ablations_A1_A2_A3]] — Ablation 3종 상세
- [[Calibration]] — ECE, reliability plot, threshold 결정

## 평가 인프라

- `downstream/model.py::PhysioMEClassifier` — frozen encoder + fc head
- `downstream/utils.py::load_pretrained_to_classifier` — hetero ckpt 직접 load + apply_lora
- `downstream/run_*.py` — task별 evaluator (7-subset sweep 자동)
- `downstream/run_ablation_a2.py` — A2 전용 (real-missing vs synth-missing 분리 평가)

## Plan 파일

- `.plans/.agent_plan/plan_eval.md` — estimator agent의 평가 가이드 (현재 미완료 8개 task)

## 원본

- 코드: `downstream/`
- 검증 인프라: `experiments/smoke_test_downstream.py` (7-subset inference 통과)
