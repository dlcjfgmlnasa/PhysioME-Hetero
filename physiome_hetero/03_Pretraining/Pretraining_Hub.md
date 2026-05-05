---
tags: [hub, pretraining]
---

# 03 Pretraining — Hub

> 2-Phase SSL 커리큘럼·Loss·Config.

## 2-Phase 구조

```
Phase 1: per-modality SSL  ─────  3개 modality 별도 ckpt
    NeuroNet + TF-C
        L_T + L_F + L_TF + recon
                    ↓
Phase 2: multimodal SSL    ─────  hetero PhysioME ckpt
    PhysioME + hetero-bucket
        inter_recon + miss_recon + cross_contra
        (availability-aware 분해)
```

## 핵심 노트

- [[Phase1_NeuroNet_TFC]] — per-modality SSL (TF-C 4 loss + RoPE encoder)
- [[Phase2_PhysioME_Hetero]] — multimodal SSL (3-state presence + bucket sampling + LoRA on out_proj)
- [[Configs]] — 어떤 config 파일이 어떤 실험에 쓰이나

## 학습 entry point

| Phase | 스크립트 | Config |
|---|---|---|
| Phase-1 | `pretrained/dp_neuronet/train.py` | `config/vital_db/dp_neuronet.yaml` |
| Phase-2 (hetero) | `pretrained/physiome/train_hetero.py` | `config/vital_db/physiome_hetero.yaml` |
| Phase-2 (A1 baseline) | `pretrained/physiome/train.py` | `config/vital_db/physiome.yaml` |
| Phase-2 (A3 ablation) | `pretrained/physiome/train_hetero.py` | `config/vital_db/physiome_hetero_a3.yaml` |

## 관련 결정

- [[../02_Architecture/Decision_TFC_over_SimCLR]]
- [[../02_Architecture/Decision_FreqProj_Linear]]
- [[../02_Architecture/Decision_Hetero_Bucket]]
- [[../02_Architecture/Decision_HandRolled_LoRA]]

## 검증 인프라

- `experiments/smoke_test_neuronet_tfc.py` — Phase-1 4 loss 모두 fire / gradient 흐름 / 수렴
- `experiments/smoke_test_hetero.py` — Phase-2 7-bucket / availability-aware loss / presence embedding gradient
- `experiments/smoke_test_downstream.py` — frozen encoder + 7-subset inference

## 원본

- 코드: `pretrained/`
- Plan: `.plans/.agent_plan/plan_model.md`
