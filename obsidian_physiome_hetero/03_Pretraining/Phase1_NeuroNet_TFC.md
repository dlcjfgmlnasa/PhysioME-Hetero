---
tags: [pretraining, phase1, tfc]
---

# Phase 1 — NeuroNet + TF-C (per-modality SSL)

## 목표

각 modality (ABP, ECG, PPG)에 대해 *독립적*으로 self-supervised representation 학습. Phase-2 PhysioME가 transfer하는 universal representation을 만든다.

## Architecture

```
input x  (B, T)  T = fs * second = 100 * 60 = 6000
    │
    ├── _frames_time(x)         (B, F, W)  W = fs * time_window = 100 * 3 = 300
    │       └── frame_backbone (time, ResNet1D)
    │
    └── _frames_freq(frames_t)              |FFT| magnitude (W//2+1 = 151)
            └── freq_proj  (Linear: 151 → 300)        ← learned lift
                    └── frame_backbone_freq (freq, ResNet1D)

각각 → SHARED autoencoder (RoPE TransformerEncoder)
    ├── two random-masked passes per view → latent_v1, latent_v2
    ├── decoder reconstruction (MAE)
    └── CLS token 추출

CLS tokens → 3 projectors → contrastive losses
    L_T  : projector_time(latent_t1[CLS])  vs  projector_time(latent_t2[CLS])
    L_F  : projector_freq(latent_f1[CLS])  vs  projector_freq(latent_f2[CLS])
    L_TF : projector_tfc(latent_t1[CLS])   vs  projector_tfc(latent_f1[CLS])

총 loss = recon + L_T + L_F + L_TF
```

## 핵심 설정

| 항목 | 값 |
|---|---|
| fs / second | 100 / 60 |
| time_window / time_step | 3 / 3 (no overlap, F=20 frames) |
| frame backbone | ResNet1D (`FrameBackBone`) — input_size=300 |
| autoencoder encoder | RoPE TransformerEncoder (RMSNorm + GQA + GLU FFN) |
| temperature | 0.01 (default — sweep 0.05~0.1 권장) |

## Forward signature

```python
recon, l_t, l_f, l_tf = NeuroNet(x, mask_ratio=0.5)
# recon: scalar  — MAE on time + MAE on freq
# l_t  : scalar  — time-domain NT-Xent
# l_f  : scalar  — freq-domain NT-Xent
# l_tf : scalar  — cross-domain NT-Xent (the TF-C signature loss)
```

## Phase-2 transfer surface

`NeuroNet.forward_latent(x, global_tokens=...)` 가 Phase-2가 호출하는 entry point:
- `global_tokens=False` → CLS token만 (B, D)
- `global_tokens=True` → 전체 latent (B, F+1, D)

Phase-2는 다음 sub-module만 inherit:
- `frame_backbone` (시간 도메인만 — freq 경로는 Phase-1 only)
- `autoencoder.patch_embed`
- `autoencoder.encoder`
- `autoencoder.cls_token`

`frame_backbone_freq`, `freq_proj`, 3개 projector는 Phase-1 종료 후 폐기.

## 검증 (smoke)

`experiments/smoke_test_neuronet_tfc.py`:

```
[step0] recon=0.879  L_T=0.903  L_F=1.107  L_TF=4.263
[grads] proj_time=13.92  proj_freq=17.94  proj_tfc=31.01
        frame_t=771.99  frame_f=859.78  encoder=141.48  decoder=1.50
[converge] total: 5.74 → 0.14 over 30 steps
```

모든 module이 gradient를 받고, 30-step 고정 batch overfit으로 loss가 수렴함.

## 위험 / sweep 권장

- **L_TF 초기값이 다른 loss 대비 큼** (4.26 vs 0.36~0.90). 초반 학습에서 L_TF가 gradient를 지배할 수 있음. 만약 tensorboard에서 recon이 안 떨어지면 loss weighting (`alpha * L_TF`) 도입 고려.
- **temperature 0.01 너무 작을 수 있음** — large batch size에서 contrastive logit saturation 가능. Sweep 0.05/0.1 권장.

## 결정

- [[../02_Architecture/Decision_TFC_over_SimCLR]] — SimCLR에서 TF-C로 교체한 이유
- [[../02_Architecture/Decision_FreqProj_Linear]] — freq view를 zero-pad가 아닌 learned lift로 바꾼 이유

## 원본

- 코드: `models/dp_neuronet/model.py::NeuroNet`
- Trainer: `pretrained/dp_neuronet/train.py`
- Method 노트: [[../90_Paper/PhysioME-Hetero/11. Phase-1 TF-C]]
