---
tags: [pretraining, config]
---

# Pretraining Configs

| Config | 사용처 | Phase | 비고 |
|---|---|---|---|
| `config/vital_db/dp_neuronet.yaml` | `pretrained/dp_neuronet/train.py` | Phase-1 | per-modality SSL (TF-C 4 loss). modality argument로 ABP/ECG/PPG 선택 |
| `config/vital_db/physiome.yaml` | `pretrained/physiome/train.py` | Phase-2 | 원본 PhysioME 스타일 (complete-only). **A1 baseline** |
| `config/vital_db/physiome_hetero.yaml` | `pretrained/physiome/train_hetero.py` | Phase-2 | hetero-bucket + availability-aware. **메인 모델** |
| `config/vital_db/physiome_hetero_a3.yaml` | `pretrained/physiome/train_hetero.py` | Phase-2 | hetero-bucket이지만 `restoration_only_on_complete: false`. **A3 ablation** |

## 핵심 hyperparameter (cross-config)

| 파라미터 | 의미 | 일반 값 |
|---|---|---|
| `fs / second` | sample rate / window length | 100 / 60 |
| `time_window / time_step` | frame slice length / step | 3 / 3 |
| `encoder_embed_dim` | unimodal backbone embed | 256 |
| `encoder_heads / encoder_depths` | unimodal | 8 / 4 |
| `mask_ratio` | masking fraction | 0.8 (Phase-2) / 0.5 (Phase-1) |
| `lora_r / lora_alpha / lora_dropout` | LoRA configuration | 4 / 16 / 0.05 (rsLoRA) |
| `temperature` | NT-Xent | 0.01 (Phase-1) / 0.1 (Phase-2) |
| `restoration_only_on_complete` | A3 toggle (Phase-2 hetero only) | `true` (default) / `false` (A3) |

## A1 vs Hetero vs A3 차이

```yaml
# physiome.yaml (A1 baseline) — 원본 PhysioME 분포
data_loader: complete-only
loss:
  inter_recon: always
  miss_recon: synth-drop on every batch
  cross_contra: always

# physiome_hetero.yaml — 메인 hetero
data_loader: bucket-aware
presence_embedding: 3-state
loss:
  inter_recon: availability-aware (present only)
  miss_recon: complete bucket의 synth-drop만   ← restoration_only_on_complete=true
  cross_contra: present만

# physiome_hetero_a3.yaml — A3 ablation
data_loader: bucket-aware (hetero와 동일)
presence_embedding: 3-state (동일)
loss:
  miss_recon: 모든 hetero bucket의 synth-drop  ← restoration_only_on_complete=false
```

## Compute matching (A1 vs hetero)

A1 baseline 비교 정당성: **epoch 수, batch size, optimizer settings 동일**하게 두어 "method 효과만" 분리. config 작성 시 동기화 필수.

## 외부 데이터 경로 (config 외부, runtime arg)

| 인자 | 기본 | 설명 |
|---|---|---|
| `--data_root` | (required) | SSL npz 디렉터리 |
| `--ckpt_dir` | `ckpt/` | 출력 ckpt 경로 |
| `--abp_ckpt / --ecg_ckpt / --ppg_ckpt` | (Phase-2만) | Phase-1 ckpt 경로 3종 |

## 원본

- Config: `config/vital_db/*.yaml`
- Trainer: `pretrained/dp_neuronet/train.py`, `pretrained/physiome/train.py`, `pretrained/physiome/train_hetero.py`
- 실행 가이드: [[../90_Paper/PhysioME-Hetero/10. 실험 실행 가이드]]
