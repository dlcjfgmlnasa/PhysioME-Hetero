---
tags: [decision, architecture, paper-core]
date: 2026-05-04
status: locked
---

# Decision: 3-state presence embedding

## 결정

각 modality token에 **3-state presence embedding** (`nn.Embedding(3, encoder_embed_dim)`)을 더해 다음 세 상태를 모델이 구분할 수 있게 한다:

| 상태 | 정수 | 의미 |
|---|---|---|
| `PRESENCE_REAL` | 0 | 실제 측정값이 있음 |
| `PRESENCE_SYNTH_DROPPED` | 1 | Real이지만 synthetic drop된 것 (restoration loss target 존재) |
| `PRESENCE_NATURALLY_ABSENT` | 2 | 처음부터 측정 안 됨 (GT 없음) |

## Why

원본 PhysioME는 "modality가 있는가/없는가" 2-state만 구분 (있으면 input, 없으면 mask). Hetero-bucket setup에서는 이게 부족:

- **Synth-dropped**와 **naturally-absent**를 모델이 구분 못 하면 *restoration loss를 어디에 적용해야 할지 모호*.
- Restoration head가 synth-dropped에는 GT가 있는데 (real signal에서 drop했으니), naturally-absent에는 GT가 없음. 이 둘을 같이 처리하면 train signal이 noisy.
- 모델이 "이건 실제 측정값" vs "이건 algorithmically dropped" vs "이건 원래부터 없음" 을 알면, 각 상태에 다르게 행동할 수 있음 (e.g., dropped slot은 restore하려 하고 absent slot은 다른 modality로 fall-back).

## How to apply

코드 위치: `models/physiome/model.py`
- `presence_state_embed: nn.Embedding(3, encoder_embed_dim)` — 학습형
- `forward_encoder`에서 각 modality token에 더해진다 (positional 안 더함, RoPE 사용)
- `presence_state` argument를 dataloader에서 명시적으로 전달
- 기본값 (테스트용): `_default_presence_state(data)` — 모든 present는 0, missing은 2

`dropped_modality_token: nn.Parameter(...)` — naturally-absent / synth-dropped 슬롯의 *값* (어떤 token으로 채울지). presence_embed가 *상태 정보*, dropped_modality_token이 *대체 representation*.

## Loss decomposition과의 결합

3-state는 [[Decision_Hetero_Bucket]]의 availability-aware loss와 짝:

| Loss | 적용 조건 |
|---|---|
| Inter-modal masked recon | present (state ∈ {0, 1}) modality 만 |
| Missing recon (restoration) | synth_dropped (state == 1) — config: `restoration_only_on_complete=true`이면 complete bucket의 synth drop만 |
| Cross-contrastive | 2개 이상 present일 때만 |

## 검증 (smoke)

`experiments/smoke_test_hetero.py`에서:
- presence_state_embed gradient norm 0.5662 ✅ (학습됨)
- dropped_modality_token gradient norm 14.61 ✅ (학습됨)
- presence states {0, 2}만 출현 — synth_dropped (1)은 hetero loader에서 안 만들고, drop은 model 내부에서 수행하므로 state 1은 forward 내부에서만 transient

## 원본

- 코드: `models/physiome/model.py` (PRESENCE_REAL/SYNTH_DROPPED/NATURALLY_ABSENT 상수)
- Method 노트: [[../90_Paper/PhysioME-Hetero/05. Method - Presence embedding]]
