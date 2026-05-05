---
tags: [overview]
---

# Project Overview — PhysioME-Hetero

## 한 줄 요약

> Multimodal SSL with synthetic missing-modality drop is an **out-of-distribution generalization problem**. We close the gap by training on the **real heterogeneous availability distribution** itself.

## 무엇을 만드는가

PhysioME (arXiv:2510.11110)의 **후속 논문**. 원본 PhysioME는 *모든 모달리티가 존재하는* 데이터로만 학습한 뒤 testtime에 random drop으로 robustness를 평가한다. 이 setup은 학습/평가 분포가 다른 OOD 문제 — 실제 임상 환경에서 모달리티 결손은 random drop이 아니라 *수술/환자/장비별로 구조적으로 결정*되기 때문이다.

PhysioME-Hetero는 **3가지 변경**으로 이 gap을 닫는다:

1. **Hetero-bucket sampling** — 학습 단계부터 ABP/ECG/PPG의 임의 부분집합(7 buckets)을 노출.
2. **Availability-aware loss decomposition** — restoration / inter-modal / cross-contrastive loss를 *실제로 적용 가능한 batch에만* 적용.
3. **3-state presence embedding** — real-present / synth-dropped / naturally-absent 세 상태를 모델이 구분 가능하게.

## 데이터 / 모달리티 / 평가

| 항목 | 값 |
|---|---|
| 사전학습 데이터 | **VitalDB only** (서울대 vital lab 출신, IOH 임상 narrative 강함) |
| 모달리티 | **ABP / ECG / PPG** (3종, @ 100 Hz, 60s 윈도우) |
| Phase-1 SSL | NeuroNet + **TF-C** (Time-Frequency Consistency) |
| Phase-2 SSL | PhysioME hetero-bucket + availability-aware loss |
| Primary downstream | IOH (MAP<65 sustained ≥1min, 5min horizon) |
| Secondary downstream | AKI (KDIGO), Mortality (MIMIC-III WDB transfer) |
| 타깃 venue | **IEEE JBHI** (1순위) |

## 왜 JBHI인가

- 임상 AI 방법론 + biosignal + ablation-heavy 논문에 적합
- prospective trial 요구 없음 (npj Digital Medicine보다 수월)
- Rolling submission, ~3–4개월 1차 결정
- IF ~7.7

## 핵심 contribution (paper claim 형태)

1. We frame missing-modality robustness as an **OOD generalization** problem and demonstrate the gap on real VitalDB data.
2. We propose **hetero-bucket SSL training** that exposes the model to natural availability distribution, with **availability-aware loss decomposition** so each loss is only applied where the GT exists.
3. We introduce a **3-state presence embedding** (real-present / synth-dropped / naturally-absent) that lets the model condition on availability source.
4. On IOH (primary) and AKI/Mortality (secondary), our method **closes the real-vs-synth missing gap** measured in Ablation A2 — a direct reviewer-disarming experiment.

## 진척도 (2026-05-05 기준)

```
[✓] Code infrastructure (Tasks 1-7, 14-24)
[✓] BFM Transformer port + TF-C + hand-rolled LoRA + freq_proj fix
[✓] Smoke tests passing on CPU
[ ] Phase-1 pretraining (Task 8) — needs GPU
[ ] Downstream eval (Tasks 9-12)
[ ] Paper draft (Task 13)
```

## 원본

- 코드: `C:\Projects\PhysioME\` (전체 레포)
- 핵심 메시지: [[../90_Paper/PhysioME-Hetero/00. 핵심 메시지]]
- Master plan: [[Master_Plan]]
