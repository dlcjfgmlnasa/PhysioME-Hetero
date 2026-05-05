---
tags: [pretraining, phase2, hetero]
---

# Phase 2 — PhysioME-Hetero (multimodal SSL)

## 목표

3 modality (ABP/ECG/PPG)의 임의 부분집합으로 학습되는 multimodal SSL. **Hetero-bucket sampling + availability-aware loss decomposition + 3-state presence embedding**의 조합.

## Architecture

```
data: Dict[modality_name, Tensor (B, T)]
presence_state: Tensor (B, 3)  ∈ {0=REAL, 1=SYNTH_DROPPED, 2=NATURALLY_ABSENT}

각 modality:
    unimodal backbone (LoRA-wrapped, base frozen)
        from Phase-1 NeuroNetEncoder
        ↓
    + presence_state_embed[state]   ← 3-state
    ↓
    (absent slots → dropped_modality_token)

concat all modalities → multimodal encoder (RoPE TransformerEncoder)
    ↓
    fusion_tokens

Three losses:
    1. inter_recon  — present modality에 대한 inter-modal masked reconstruction
    2. miss_recon   — synth-drop된 modality 복원 (availability gate)
    3. cross_contra — fusion ↔ each present unimodal (NT-Xent)
```

## Forward signature

```python
inter, miss, contra, acc = PhysioME.forward(
    data={...},                      # Dict[modal_name, (B, T)]
    presence_state=...,              # (B, 3)
    mask_ratio=0.8,
    simulate_drop=True,
    restoration_only_on_complete=True,
)
total = inter + miss + contra
```

## 3가지 loss의 적용 조건 (availability-aware)

| Loss | 조건 | Smoke 결과 |
|---|---|---|
| `inter_recon` | present (state ∈ {0, 1}) modality | 모든 bucket에서 fire |
| `miss_recon` | `restoration_only_on_complete=True`이면 complete bucket(`111`)의 synth drop만 / `False`이면 모든 hetero bucket의 synth drop | 111 bucket: 2.5+, 그 외: 0.0 |
| `cross_contra` | (현재) 모든 bucket — 단일 modal에서는 약한 신호 | 모든 bucket에서 fire |

→ A3 ablation toggle: [[../05_Downstream/Ablations_A1_A2_A3]]

## 핵심 설정

| 항목 | 값 |
|---|---|
| Backbone | NeuroNetEncoder (Phase-1 ckpt에서 transfer) — LoRA-wrapped |
| LoRA target | `out_proj` (GQA의 output projection) |
| LoRA rank / alpha | r=4, alpha=16 (rsLoRA, scale=alpha/sqrt(r)) — config로 조정 |
| Multimodal encoder | RoPE TransformerEncoder, separate per-modal MAE decoder + restoration decoder |
| presence_state_embed | `nn.Embedding(3, encoder_embed_dim)` |
| dropped_modality_token | `nn.Parameter(...)` — 학습형, naturally-absent/synth-dropped 슬롯 채움 |
| mask_ratio | 0.8 (default — config로 조정) |

## Bucket sampler

`pretrained/physiome/hetero_data_loader.py`:
- `HeteroVitalDBDataset` — 각 segment마다 modality availability 메타데이터 보존
- `BucketBatchSampler` — 매 batch는 동일 bucket
- `hetero_collate_fn` — present-only modality tensor 묶기

7 bucket: `{001, 010, 100, 011, 101, 110, 111}`.

## 학습 시작 (manual workflow)

```bash
# 1. Phase-1 ckpt 3개 준비 (per modality)
python -m pretrained.dp_neuronet.train --modality ABP --config config/vital_db/dp_neuronet.yaml
python -m pretrained.dp_neuronet.train --modality ECG ...
python -m pretrained.dp_neuronet.train --modality PPG ...

# 2. Phase-2 hetero
python -m pretrained.physiome.train_hetero \
    --config config/vital_db/physiome_hetero.yaml \
    --abp_ckpt ckpt/dp_neuronet/ABP.pt \
    --ecg_ckpt ckpt/dp_neuronet/ECG.pt \
    --ppg_ckpt ckpt/dp_neuronet/PPG.pt
```

## 검증 (smoke)

`experiments/smoke_test_hetero.py`:
- 7 bucket 모두 노출 ✅
- `miss` loss는 `111` bucket에서만 non-zero ✅
- `presence_state_embed` gradient 0.57 ✅
- `dropped_modality_token` gradient 14.61 ✅
- 40 step 모두 finite loss ✅

## Bug-fix 기록 (원본 PhysioME 대비)

[[../02_Architecture/Architecture_Hub#Bug-fixes 기록]] 참조 — split_size 불일치 + decoder `.detach()` 두 가지 수정.

## 결정

- [[../02_Architecture/Decision_Hetero_Bucket]]
- [[../02_Architecture/Decision_Presence_Embedding]]
- [[../02_Architecture/Decision_HandRolled_LoRA]]

## 원본

- 코드: `models/physiome/model.py::PhysioME`
- Trainer: `pretrained/physiome/train_hetero.py`
- Sampler: `pretrained/physiome/hetero_data_loader.py`
- Config: `config/vital_db/physiome_hetero.yaml`, `physiome_hetero_a3.yaml`
- Method 노트: [[../90_Paper/PhysioME-Hetero/03. Method - Hetero-bucket]]
