---
tags: [decision, architecture]
date: 2026-05-05
status: locked
---

# Decision: TF-C 주파수 view를 학습된 linear lift로 변경

## 결정

`NeuroNet._frames_freq` 의 **zero-padding hack** (|FFT| magnitude를 W//2+1 → W로 0으로 채워 늘림)을 제거하고, 학습된 `nn.Linear(W//2+1, W)` projection (`self.freq_proj`)으로 교체한다.

```python
# 이전 (hack)
mag = torch.fft.rfft(frames_time, dim=-1).abs()        # (B, F, W//2+1)
mag = F.pad(mag, (0, pad))                              # (B, F, W) — pad 절반 가까이 구조적 0

# 현재 (learned lift)
mag = torch.fft.rfft(frames_time, dim=-1).abs()        # (B, F, W//2+1)
return self.freq_proj(mag)                             # (B, F, W)
```

## Why

`FrameBackBone`은 `fs=100, time_window=3` 기준으로 AvgPool kernel size (16, 11, 6)이 하드코딩 — 입력 길이를 바꾸면 shape error. zero-padding은 architecture 재사용을 위한 hack이었으나:

1. **모델 입력의 절반 가까이가 구조적 0**. W=300, W//2+1=151 → 149개 (≈50%)가 항상 0.
2. **Conv 필터들이 trivial 패턴 학습 capacity 낭비** — "151 이후는 무조건 0"을 학습.
3. **Receptive field 오염** — stride=2 conv 두 번 거치면 zero-pad 영역과 진짜 spectral 영역이 convolved되며 정보 흐름이 zero에 dilute.
4. **TF-C 원논문은 zero-padding을 안 함** — 별도 architecture로 freq encoder 운영. zero-padding은 우리 implementation의 hack이지 method가 아님.

## What changed

**Added** in `NeuroNet.__init__`:
```python
window = int(time_window * fs)
self.freq_proj = nn.Linear(window // 2 + 1, window)
```

**Modified** `_frames_freq`:
```python
def _frames_freq(self, frames_time):
    mag = torch.fft.rfft(frames_time, dim=-1).abs()    # (B, F, W//2+1)
    return self.freq_proj(mag)                          # (B, F, W) -- learned
```

**Drive-by cleanup**:
- `import torch.nn.functional as F` 제거 (더 이상 `F.pad` 호출자 없음)
- 주석 fix: time/freq 도메인 contrastive loss는 "SimCLR"이 아니라 "NT-Xent" (loss form 명칭 vs framework 명칭)

## 효과 (smoke 비교)

| 지표 | zero-pad (이전) | freq_proj (현재) | 의미 |
|---|---|---|---|
| L_F init | 0.364 | 1.107 | 3× 증가 — zero 영역의 trivial 패턴 사라져 contrastive task가 진짜 난이도 |
| frame_f gradient | 649.92 | 859.78 | +32% — backbone이 의미있는 freq filter 학습 |
| proj_freq gradient | 9.91 | 17.94 | +80% — projector가 더 많은 일을 함 |
| Total convergence | 5.80 → 0.15 | 5.74 → 0.14 | 동등 — 수렴 보장됨 |

## 비용

- 추가 파라미터: W × (W//2+1) = 300 × 151 ≈ **45k weights** (per-modality 하나라 negligible)
- Forward 비용: 추가 matmul 한 번 — 미미

## 호환성

**이전 NeuroNet ckpt와 비호환** — 새 `freq_proj` 파라미터가 추가되어 state_dict load 실패. 어차피 Phase-1 pretraining이 아직 실행 안 되었으므로 영향 없음.

## 원본

- 코드: `models/dp_neuronet/model.py::NeuroNet._frames_freq`, `freq_proj`
- 검증: `experiments/smoke_test_neuronet_tfc.py`
- Commit: `083d434` (2026-05-05)
