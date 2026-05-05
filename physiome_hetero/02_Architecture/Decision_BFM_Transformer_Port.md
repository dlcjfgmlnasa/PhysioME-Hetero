---
tags: [decision, architecture]
date: 2026-05-04
status: locked
---

# Decision: BFM Transformer 포팅 (timm Block 제거)

## 결정

NeuroNet 인코더와 PhysioME 멀티모달 stack 전체에서 timm `Block`과 ad-hoc `models/rope.py`를 **모두 제거**하고, Biosignal-Foundation-Model 프로젝트의 `module/` (uni2ts/Salesforce, Apache 2.0 derived)을 `models/transformer/` 로 wholesale 포팅한다.

## 무엇이 들어왔나

```
models/transformer/
├── norm.py            — RMSNorm, AdaRMSNorm
├── attention.py       — GroupedQueryAttention (GQA), MHA, MQA
├── ffn.py             — FeedForward, GatedLinearUnitFeedForward (GLU), MoEFeedForward
├── position/
│   ├── attn_bias.py
│   └── attn_projection.py — QueryKeyProjection, RotaryProjection (RoPE)
├── transformer.py     — TransformerEncoder/Layer (RoPE-aware, d_cond=0 path)
└── lora.py            — Hand-rolled LoRA (이건 별개 결정 - [[Decision_HandRolled_LoRA]])
```

핵심 조합: **RMSNorm + GQA + GLU FFN + RoPE**.

## Why (이전 결정 reverse)

2026-05-04 초기 결정에서는 BFM `module/` 포팅을 *명시적으로 거절*했었다 ("paper scope를 벗어난다"). 같은 날 후반부에 reverse한 이유:

1. **사용자의 두 프로젝트 unified codebase 요구** — BFM과 PhysioME 모두 같은 transformer 코드를 쓰면 architecture sweep / debugging이 쉬워짐.
2. **timm Block + sinusoidal pos_embed의 한계** — 학습 시 fixed length, inference에서 다른 길이는 interpolation 필요. PhysioME는 multimodal이라 길이가 modality 수에 비례 (`num_modals * num_frames`) → interpolation 매번 필요.
3. **RoPE의 자연스러운 임의 길이 지원** — `models/rope.py` 라는 ad-hoc stopgap을 작성했었으나 production-grade가 아님. BFM의 `QueryKeyProjection` + `RotaryProjection`은 잘 구현된 코드.
4. **BFM 코드가 이미 production grade** (Salesforce uni2ts 출처) — 직접 작성보다 검증됨.

## d_cond=0 모드 (PhysioME-specific)

BFM의 `TransformerEncoder`는 conditioning vector (`d_cond`)를 받으면 AdaRMSNorm을 사용한다. PhysioME에는 conditioning이 없으므로 `d_cond=0`을 명시 → 일반 RMSNorm path 사용.

`models/transformer/transformer.py`에 이 분기가 추가되어 있음:
```python
if d_cond and d_cond > 0:
    self.norm = AdaRMSNorm(...)
else:
    self.norm = norm_layer(d_model)
```

## 구조적 임팩트

이 포팅으로 함께 결정/변경된 것들:

- [[Decision_RoPE_Everywhere]] — pos_embed 테이블 모두 제거
- [[Decision_HandRolled_LoRA]] — peft 의존성 제거 + LoRA target attr 변경 (`attn.proj` → `out_proj`)
- Phase-1 NeuroNet ckpt와 Phase-2 PhysioME ckpt **둘 다 비호환** — 새로 학습 필요

## 위험 / 주의

- BFM 코드의 일부 unused argument (e.g., `mask_ratio`, `absent_modals`) 가 우리 PhysioME에서 다이얼로그 mismatch를 유발할 수 있음 — diagnostic warning은 무시 (out of scope cleanup).
- BFM에 있는 MoE / multi-resolution patch / shared norm 옵션은 **포팅하지 않음** (paper scope 유지).

## 원본

- 결정 메모: `memory/project_physiome_state.md` "Reversed decision 2026-05-04"
- 코드: `models/transformer/`
- 비교 노트: [[../90_Paper/PhysioME-Hetero/03. Method - Hetero-bucket#section 3.6]]
