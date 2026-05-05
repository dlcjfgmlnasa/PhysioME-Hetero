# Plan: Model Architecture — PhysioME-Hetero

> 본 파일은 **model-architect** 에이전트의 작업 가이드입니다. 항상 이 파일을 먼저 읽고 다음 미완료 태스크를 실행하세요.
> Single source of truth: `obsidian_physiome_hetero/00_MOC.md` 와 `obsidian_physiome_hetero/02_Architecture/Architecture_Hub.md`.

---

## 1. 모델 스택 개요

PhysioME-Hetero는 **2-Phase SSL** 구조:

```
[Phase 1] per-modality SSL (NeuroNet + TF-C)
    └── ABP backbone, ECG backbone, PPG backbone (각각 독립)

[Phase 2] multimodal SSL (PhysioME with hetero-bucket)
    └── unimodal backbones (LoRA-tuned) + multimodal encoder/decoders
        ↓
[Downstream] PhysioMEClassifier (frozen encoder + fc head)
```

---

## 2. 디렉터리 / 컴포넌트

```
models/
├── transformer/                  # BFM-derived primitives (uni2ts/Salesforce, Apache 2.0)
│   ├── norm.py                   # RMSNorm, AdaRMSNorm
│   ├── attention.py              # GroupedQueryAttention (GQA), MultiHeadAttention, MultiQueryAttention
│   ├── ffn.py                    # FeedForward, GatedLinearUnitFeedForward (GLU), MoEFeedForward
│   ├── position/                 # AttentionBias, BinaryAttentionBias, RotaryProjection, QueryKeyProjection (RoPE)
│   ├── transformer.py            # TransformerEncoder/Layer (RoPE-aware, d_cond=0 path → plain RMSNorm)
│   └── lora.py                   # LoRALinear (rsLoRA scale), apply_lora helper
├── dp_neuronet/
│   ├── model.py                  # NeuroNet (TF-C: time/freq backbones + 3 projectors), NeuroNetEncoder (Phase-2용)
│   └── resnet1d.py               # ResNet1D backbone (FrameBackBone로 사용 가능)
├── physiome/
│   └── model.py                  # PhysioME 멀티모달 모델 (encoder + MAE decoder + restoration decoder)
├── loss.py                       # 공통 loss helpers (NT-Xent 등)
└── utils.py                      # 보조 유틸 (sincos pos_embed 등 — NeuroNet decoder만 사용)
```

---

## 3. 핵심 아키텍처 결정 (변경 금지, 변경 시 사용자 승인 필수)

### 3.1 Backbone (BFM Transformer 통일)
- 모든 transformer stack (NeuroNet encoder, PhysioME multimodal encoder/decoders)이 `models/transformer/transformer.py`의 `TransformerEncoder` 사용.
- timm `Block` 및 ad-hoc `models/rope.py`는 **삭제됨**.
- `d_cond=0` 모드: PhysioME는 conditioning 없으므로 plain `RMSNorm` 사용 (AdaRMSNorm 비활성).
- Position: RoPE only (via `QueryKeyProjection`). 학습형 1D pos_embed 테이블은 **모두 제거**.

### 3.2 Phase-1 (NeuroNet + TF-C)
- 시간 도메인 백본 + 주파수 도메인 백본 (FFT magnitude를 W로 zero-pad해서 동일 `FrameBackBone` 재사용)
- **Shared autoencoder** (`patch_embed` + transformer encoder + `cls_token` + decoder) — Phase-2로 transfer되는 보편 표현
- 3 projectors: `projector_time`, `projector_freq`, `projector_tfc`
- Forward signature: `forward(x, mask_ratio) -> (recon, L_T, L_F, L_TF)`
- 4 losses: Reconstruction + L_T (time NT-Xent) + L_F (freq NT-Xent) + L_TF (cross-domain alignment)
- SimCLR view-augmentation은 더 이상 사용 안 함 (`pretrained/dp_neuronet/augmentation.py` 미사용 보관)

### 3.3 Phase-2 (PhysioME hetero)
- `PhysioME.forward(...) -> (inter_recon_loss, missing_recon_loss, cross_contra_loss, cross_contra_acc)`
- **3-state presence embedding** (real-present / synth-dropped / real-absent) + `dropped_modality_token` (학습형)
- **Availability-aware loss decomposition**:
  - inter-modal masked recon: present modality에만 적용
  - missing recon (restoration): **complete bucket** (3 modal 모두 present) 에서만 (config: `restoration_only_on_complete: true`, A3 ablation에서 `false`)
  - cross-contrastive: 2개 이상 present일 때만
- Bug-fix 적용 완료:
  - `forward_restoration_decoder`의 split_size 버그 수정 (`dropped_modality_token` padding으로 `num_total_modals * num_backbone_frames` 길이 보장)
  - decoder forward 끝의 `.detach()` 제거 (`.contiguous()`만 유지)

### 3.4 LoRA (peft 의존성 제거)
- `models/transformer/lora.py::LoRALinear` (frozen base + rank-r residual, rsLoRA: `alpha / sqrt(r)`)
- `apply_lora(module, target_attrs=('out_proj',), r, alpha, dropout, rslora=True)`
- Target: GQA의 `out_proj` (이전 `attn.proj` → BFM 포팅으로 이름 변경됨)
- 호출 순서: pretrained weight load → `apply_lora` (base에 weight가 들어간 상태에서 swap)
- State-dict key: `encoder.layers.<i>.self_attn.out_proj.{base.weight, lora_A, lora_B}` (peft prefix 없음)

### 3.5 Downstream
- `downstream/model.py::PhysioMEClassifier` — frozen encoder + fc head
- `downstream/utils.py::load_pretrained_to_classifier` — hetero ckpt 직접 로드, `apply_lora` 호출

---

## 4. 데이터 컨트랙트 (data-engineer와 공유)

| 인터페이스 | 형태 | 비고 |
|---|---|---|
| Phase-1 batch input | `(B, 1, T)` | T=6000 (60s @ 100Hz) |
| Phase-2 batch input | `(B, M_present, T)` + presence_mask `(B, 3)` | M_present ∈ {1,2,3}, bucket-uniform |
| Backbone output | `(B, num_frames, embed_dim)` | num_frames = T / patch_size |
| Multimodal encoder input | `(B, M_total * num_frames, embed_dim)` | absent → `dropped_modality_token` |
| Downstream classifier input | `(B, 3, T)` zero-filled + presence_mask | 학습/평가 모두 동일 인터페이스 |

---

## 5. 태스크 — 진행 상태

### 완료 (`[x]`)
- [x] **[High]** BFM Transformer 포팅 (norm/attention/ffn/position/transformer)
- [x] **[High]** Phase-1 SSL: SimCLR → TF-C 전환
- [x] **[High]** PhysioME 멀티모달 스택 RoPE-everywhere 전환 (timm Block 제거)
- [x] **[High]** 3-state presence embedding + dropped_modality_token 도입
- [x] **[High]** Availability-aware loss decomposition 구현
- [x] **[High]** PhysioME 두 가지 버그 수정 (split_size, decoder .detach())
- [x] **[High]** peft → 자체 LoRA (`models/transformer/lora.py`) 교체
- [x] **[High]** Downstream model wrapper (`PhysioMEClassifier`) 작성
- [x] **[Medium]** Smoke test (`experiments/smoke_test_hetero.py`) 통과 — 7 bucket, presence-state, gradient flow 검증
- [x] **[High]** Linear-probe refactor (`pretrained/physiome/probe_utils.py`) — SVC → LR + 모달 부분집합 샘플링 (Step 1, N≥4 확장 선결조건)

### 진행 예정 (`[ ]`)

- [ ] **[High]** **Phase-1 NeuroNet pretraining 실행 (per-modality)**
  - 입력: `data/vital_db_ssl/*.npz` (data-engineer Task 1 결과)
  - 출력: `ckpt/dp_neuronet/{ABP, ECG, PPG}.pt`
  - 의존성: data-engineer의 SSL 데이터 생성 완료
  - 참고: 4-loss (recon / L_T / L_F / L_TF) 별도 로깅. tensorboard에서 cross-domain alignment(L_TF) 수렴 확인.

- [ ] **[High]** **Phase-2 PhysioME-Hetero pretraining 실행**
  - 입력: Phase-1 ckpt 3개 + hetero SSL 데이터
  - 출력: `ckpt/physiome_hetero/best.pt` (model_state, modality_backbone_param, entire_model_param, hyperparameter, ch_names 포함)
  - 의존성: Phase-1 완료
  - 참고: `config/vital_db/physiome_hetero.yaml` 사용. LoRA r/alpha는 config에서.

- [ ] **[Medium]** **A1 baseline (synth-only PhysioME) pretraining**
  - 입력: hetero 데이터 중 complete bucket만 (또는 기존 `data_loader.py` 사용)
  - 출력: `ckpt/physiome_a1_synthonly/best.pt`
  - 의존성: Phase-1 완료
  - 참고: 매칭 compute (epoch 수, batch size 동일). `train.py` 사용 (구버전 trainer).

- [ ] **[Medium]** **A3 ablation pretraining**
  - 입력: hetero 데이터
  - 출력: `ckpt/physiome_a3_no_complete_only/best.pt`
  - 의존성: A1/Phase-2 baseline 완료
  - 참고: `config/vital_db/physiome_hetero_a3.yaml` 사용 (`restoration_only_on_complete: false`).

- [ ] **[Low]** **Sleep-EDFx 잔재 정리 검증**
  - models/dp_neuronet 코드에 `EEG`/`Sleep-EDF` 하드코딩 잔존 여부 grep
  - 참고: 2026-05-04 삭제 결정 후 cleanup. EEG-specific 가정이 남아 있으면 ABP/ECG/PPG generic으로 변경.

- [ ] **[Low]** **MoE / multi-resolution patch 검토 (out of scope, 보류)**
  - BFM에는 있지만 PhysioME에는 미포팅. 첫 paper에는 안 넣기로 결정.
  - 참고: 향후 follow-up에서 고려.

### Modality 확장 (Step 2 / Step 3 — Master_Plan과 연동)

- [ ] **[High]** **Step 2 — CO2 modality 추가 (4 modal, 15 bucket)**
  - 입력: VitalDB raw (CO2/etCO2 채널), 기존 ABP/ECG/PPG 파이프라인
  - 출력: 4-modal Phase-1 ckpt 4개 + Phase-2 hetero ckpt
  - 의존성: data-engineer의 vital_db_ssl.py CO2 확장 완료, Phase-1 trainer가 CO2 modality argument 받기
  - 참고: 변경 위치: `pretrained/physiome/hetero_data_loader.py::MODAL_ORDER`, `models/physiome/model.py` (새 modality 추가 시 backbone dict 자동 반영), config 4종 (`dp_neuronet.yaml`, `physiome_hetero.yaml`, `physiome_hetero_a3.yaml`, A1용 `physiome.yaml`).

- [ ] **[Medium / Conditional]** **Step 3 — CVP modality 추가 (5 modal, 31 bucket)**
  - **선결조건**: Step 2의 SSL bucket 분포 통계에서 CVP coverage ≥20%이고 자연 분포가 한두 bucket에 압도적으로 쏠리지 않을 것.
  - 입력: CO2 단계 산출물 + VitalDB CVP 채널
  - 출력: 5-modal Phase-1 ckpt 5개 + Phase-2 hetero ckpt
  - 의존성: Step 2 완료. **Master_Plan 의 Venue 결정도 npj 로 갱신** 필수.
  - 참고: bucket 31개 운영 시 `BucketBatchSampler.min_bucket_size` 조정 필요할 수 있음 (long-tail bucket 대비 oversampling factor JSON).

---

## 6. Quality Standards

- 모든 `nn.Module` 은 docstring에 (1) 목적, (2) input shape, (3) output shape 명시
- Tensor shape 주석은 non-trivial한 경우만 (`(B, C, T)`, `(B, S, D)`).
- 수치 안정성: `eps` 파라미터 노출, `torch.clamp` 적절히 사용
- `apply_lora` 호출 후 `model.parameters()` 중 `requires_grad=True`인 것의 비율 로깅 (~0.2% 기대)
- Phase 전환 시 ckpt 호환성 명시: BFM 포팅 이후 Phase-1 ckpt는 이전 버전과 비호환

---

## 7. 다음 액션

`@model-architect`를 호출할 때 가장 먼저 처리할 태스크는 **Phase-1 NeuroNet pretraining 실행** 입니다 (Section 5의 첫 미완료 항목). 단, **data-engineer의 첫 task (SSL npz 생성) 완료가 선행 조건**입니다. 데이터 준비 안 됐으면 먼저 data-engineer를 호출하세요.
