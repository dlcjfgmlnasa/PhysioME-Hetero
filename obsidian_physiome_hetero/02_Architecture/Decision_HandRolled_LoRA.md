---
tags: [decision, architecture]
date: 2026-05-05
status: locked
---

# Decision: peft 의존성 제거, LoRA 직접 구현

## 결정

`peft` 패키지 의존성을 제거하고 `models/transformer/lora.py`에 자체 `LoRALinear` + `apply_lora` 헬퍼를 구현한다.

## Why

`peft`는 검증된 라이브러리지만 우리 setup에 세 가지 친화도 문제:

1. **Transitive dep 부담** — `peft`는 `transformers`, `accelerate`, `safetensors`를 끌고 옴. PhysioME-Hetero에서는 셋 다 미사용. requirements 비대화.
2. **Substring-matching `target_modules` 가 silently mis-target** — BFM 포팅 후 LoRA target 이름이 `attn.proj` → `out_proj`로 바뀌었는데, peft는 substring 매칭이라 의도 안 한 layer까지 wrap 가능 (e.g., `q_proj`, `k_proj`, `v_proj` 같은 다른 `_proj` suffix layer들).
3. **State-dict prefix 복잡** — peft는 `base_model.model.<...>.base_layer.weight` 같은 경로를 만듦. 우리 ckpt 포맷 (`modality_backbone_param`, `entire_model_param`)에 직접 transfer 어려움.

## What is implemented

`models/transformer/lora.py` (~95 LoC):

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0, rslora=True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # rsLoRA scaling per Kalajdzievski 2023
        self.scale = alpha / (math.sqrt(r) if rslora else r)

    def forward(self, x):
        update = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scale
        return self.base(x) + update


def apply_lora(module, target_attrs=('out_proj',), r=4, alpha=16, dropout=0.05, rslora=True):
    for p in module.parameters():
        p.requires_grad = False
    n_replaced = 0
    for sub in list(module.modules()):
        for attr in set(target_attrs):
            child = getattr(sub, attr, None)
            if isinstance(child, nn.Linear):
                setattr(sub, attr, LoRALinear(child, r, alpha, dropout, rslora))
                n_replaced += 1
    if n_replaced == 0:
        raise RuntimeError(...)
    return module
```

핵심 특성:
- **rsLoRA scaling** (`alpha / sqrt(r)` per Kalajdzievski 2023) — 더 안정적 학습, large r에서도 magnitude 보존
- **A: gaussian init (std=1/r), B: zero init** — initial residual이 정확히 0 (학습 시작 시 base 동작 유지)
- **Exact attribute matching** (substring 아님) — 이름 매칭 모호성 없음
- **Target attr 명시 권장**: GQA의 `out_proj` 만 (다른 _proj layer들은 wrap 안 됨)

## Call order

`apply_lora`는 **pretrained weight load 후에** 호출한다 — 그래야 base에 weight가 들어간 상태에서 swap. 순서를 바꾸면 LoRA 구조에 random init된 base가 들어감.

```python
# pretrained/physiome/train_hetero.py 패턴
backbone.frame_backbone.load_state_dict(pretrained_model.frame_backbone.state_dict())
backbone.patch_embed.load_state_dict(pretrained_model.autoencoder.patch_embed.state_dict())
backbone.encoder.load_state_dict(pretrained_model.autoencoder.encoder.state_dict())
backbone.cls_token = pretrained_model.autoencoder.cls_token

backbone = apply_lora(backbone, target_attrs=('out_proj',),
    r=self.args.lora_r, alpha=self.args.lora_alpha,
    dropout=self.args.lora_dropout, rslora=True)
```

## State-dict key 예시

```
encoder.layers.0.self_attn.out_proj.base.weight    # frozen base (loaded from Phase-1)
encoder.layers.0.self_attn.out_proj.lora_A         # trainable
encoder.layers.0.self_attn.out_proj.lora_B         # trainable
```

peft 의 `base_model.model.<...>.base_layer.weight` 같은 prefix 없음.

## 검증

`experiments/smoke_test_hetero.py` (PhysioME 측 indirect 검증):
- Trainable param ratio: ~0.2% of backbone (LoRA만 학습) ✅

`experiments/smoke_test_downstream.py`:
- 7-subset inference 모두 성공 ✅
- PhysioME backbone gradient = 0 (frozen) ✅
- fc head gradient flow 정상 ✅

## 원본

- 코드: `models/transformer/lora.py`
- 호출 사이트: `pretrained/physiome/train.py`, `pretrained/physiome/train_hetero.py`, `downstream/utils.py`
- requirements.txt: `peft>=0.13` 라인 제거됨, comment로 hand-rolled 명시
- 결정 메모: `memory/project_physiome_state.md` "peft replaced with a hand-rolled LoRA"
