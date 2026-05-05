---
tags: [downstream, ablation, paper-core]
---

# Ablation 3종 (A1 / A2 / A3) — JBHI Must-Haves

> Reviewer-disarming 실험 디자인. **A2가 본 논문의 핵심 figure**.

## A1 — Hetero-train의 순수 효과

### 비교

| 모델 | 학습 데이터 | 학습 loss |
|---|---|---|
| **Synth-only baseline** | Complete-only | inter_recon + miss_recon (synth) + cross_contra |
| **Hetero (ours)** | Hetero-bucket | availability-aware decomposition |

### 측정

매 7-subset에서 두 모델의 AUROC/AUPRC 비교. **매칭 compute** (epoch×batch 동일)로 method 효과만 분리.

### Implementation

- A1 baseline: `pretrained/physiome/train.py` + `config/vital_db/physiome.yaml` (구버전 trainer)
- Hetero: `pretrained/physiome/train_hetero.py` + `config/vital_db/physiome_hetero.yaml`

### 주의

원본 PhysioME (arXiv:2510.11110)의 published numbers는 *bug-fixed* 코드와 비교 불공정. A1은 우리 bug-fix 후 *re-trained* synth-only로 비교.

→ [[../02_Architecture/Architecture_Hub#Bug-fixes 기록]]

---

## A2 — Real-missing vs Synth-missing OOD gap (**핵심**)

### 비교

같은 ckpt를 두 부분 평가:
- **Real-missing subset**: 환자가 실제로 modality가 없는 case (e.g., ABP 라인 없음)
- **Synth-missing subset**: complete data를 인위적으로 drop한 case

### 가설

원본 PhysioME (synth-only 학습) — real-missing에서 성능 *현저히 저하*. Synthetic drop으로는 못 잡는 OOD 분포 변화 존재.

PhysioME-Hetero (real distribution 노출) — real-missing과 synth-missing의 gap이 *작거나 없음*. **이게 paper의 핵심 figure**.

### Code

`downstream/run_ablation_a2.py`:

```bash
python -m downstream.run_ablation_a2 \
    --ckpt ckpt/physiome_hetero/best.pt \
    --data_root data/downstream/ioh/ \
    --output results/ablation_a2/
```

→ Output: `{real_missing.json, synth_missing.json, gap.json}` per ckpt.

### Stretch

A2 gap이 dramatic할 경우 ICLR 2027 노릴 수 있음.

### 의존성

- 데이터 파이프라인이 **real-missing flag를 보존**해야 함. `vital_db_ssl.py` / `vital_db_downstream.py`의 hetero npz schema에 modality availability 메타데이터가 segment-level로 보존되는지 확인 필요. ([[../04_Data/Hetero_NPZ_Schema]])

---

## A3 — Restoration loss decomposition 효과

### 비교

| Setting | `restoration_only_on_complete` | Loss 구성 |
|---|---|---|
| **Hetero (default)** | `true` | restoration loss는 complete bucket(`111`)의 synth-drop만 |
| **A3 ablation** | `false` | restoration loss는 모든 hetero bucket의 synth-drop |

### 가설

A3 결과가 default와 **거의 비슷**하면 → loss decomposition이 restoration 성능을 해치지 않는다는 증거 (정확한 supervision-target gating 정당화).

A3가 *더 나으면* → 더 많은 bucket에서 restoration을 학습하는 게 유리. default 결정 reverse.

A3가 *더 나쁘면* → naturally-absent slot에 GT 없이 restoration 시도하는 게 noisy. default 결정 정당화.

### Implementation

- Default hetero: `config/vital_db/physiome_hetero.yaml` (`restoration_only_on_complete: true`)
- A3 ablation: `config/vital_db/physiome_hetero_a3.yaml` (`restoration_only_on_complete: false`)
- 동일 trainer (`pretrained/physiome/train_hetero.py`), config만 다름

### 평가

A1과 동일한 방식 — 7-subset × seed 3개 × {AUROC, AUPRC, Sens@Sp90}.

---

## Reporting format

```
| Subset | Synth-only | Hetero (ours) | A3 (no complete-only) |
|---|---|---|---|
| {ABP}        | ... | ... | ... |
| {ECG}        | ... | ... | ... |
| {PPG}        | ... | ... | ... |
| {ABP+ECG}    | ... | ... | ... |
| {ABP+PPG}    | ... | ... | ... |
| {ECG+PPG}    | ... | ... | ... |
| {ABP+ECG+PPG}| ... | ... | ... |
| Average      | ... | ... | ... |
```

A2는 별도 figure (real-missing vs synth-missing gap, 두 모델 모두).

## 원본

- 코드: `downstream/run_ablation_a2.py`, configs `config/vital_db/physiome_hetero{,_a3}.yaml`
- Method 노트: [[../90_Paper/PhysioME-Hetero/07. Ablation 3종]]
- 결정 메모: `memory/project_physiome_state.md` "Reviewer-disarming experimental design"
