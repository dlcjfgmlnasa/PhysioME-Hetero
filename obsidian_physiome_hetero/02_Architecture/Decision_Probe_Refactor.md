---
tags: [decision, architecture, infra]
date: 2026-05-05
status: locked
---

# Decision: Linear-probe SVC → LR + sampled subsets

## 결정

`pretrained/physiome/{train,train_hetero}.py::linear_probing` 의 **per-epoch SVC enumeration over all 2^N − 1 modality subsets**를 다음 두 변경으로 교체:

1. **Subset sampling** (`probe_utils.select_probe_subsets`) — 항상 포함하는 must-include (full set + 각 single-modal 부분집합) 위에 random 추출로 최대 `max_subsets=10` 까지 채움. N≤3 에서는 7 subset 전부 (back-compat).
2. **Classifier swap** (`probe_utils.run_probe`) — `SVC` → `sklearn.linear_model.LogisticRegression(max_iter=1000, n_jobs=-1)`.

공통 헬퍼는 `pretrained/physiome/probe_utils.py` 에 추출 → `train.py` (A1 baseline) 와 `train_hetero.py` 둘 다 사용.

## Why

원본 메모리에 명시된 한계:
> linear_probing enumerates all 2^N − 1 modality subsets via itertools.combinations and trains an SVC for each every epoch — O(2^N) cost, slow at N=3+, will not scale to more modalities.

Modality 확장 phased plan (3 → 4 → 5)이 결정되면서 이 병목을 먼저 해소해야 함:

| N | subset 수 | 기존 SVC 부담 | 갱신 후 (max_subsets=10) |
|---|---|---|---|
| 3 | 7 | 7 SVC | 7 LR (변화 없음) |
| 4 | 15 | 15 SVC | 10 LR |
| 5 | 31 | **31 SVC** (감당 불가) | 10 LR |

Speedup multiplier (대략):
- SVC O(n²~n³) → LR O(n·d): probe-scale (n≈10³, d=128)에서 fit 시간 ~10× 감소
- Subset count 31 → 10: 추가 3.1× 감소
- N=5 에서 종합 ~30× 더 빠른 per-epoch probe

## How to apply (호출자 입장)

```python
# train_hetero.py 에서
def linear_probing(self, epoch, val_loader, eval_loader):
    self.model.eval()
    subsets = select_probe_subsets(
        self.ch_names,
        max_subsets=int(getattr(self.args, 'probe_max_subsets', 10)),
        seed=epoch,                     # epoch을 seed로 → 재현성 + 매 epoch 다른 추출
    )
    train_fn = lambda s: self._latent_vector(s, val_loader)
    eval_fn = lambda s: self._latent_vector(s, eval_loader)
    mean_acc, mean_mf1, _ = run_probe(
        subsets, train_fn, eval_fn,
        log_prefix=f'[Epoch {epoch:03d}]',
    )
    self.model.train()
    return mean_acc, mean_mf1
```

`probe_max_subsets` 는 config에서 조정 가능 (default 10).

## Must-include 보장

`select_probe_subsets` 는 다음 부분집합을 *항상* 결과에 포함:
- 전체 (`tuple(ch_names)`) — paper의 main 결과 row
- 각 single-modal `(ch,)` × N개 — modality importance / lower-bound 측정

→ "꼭 봐야 하는 corner case는 sampling으로 빠지지 않는다" 보장.

## 검증 (smoke)

`experiments/smoke_test_probe.py`:

```
[ok] N=3 -> 7 subsets (full enumeration)
[ok] N=4 -> capped at 10 of 15 with corner-case coverage
[ok] N=5 capped at 10 / full + singles guaranteed / seed-deterministic
[ok] run_probe synthetic -> acc=1.000 mf1=1.000
[smoke] PASSED -- probe_utils ready for N >= 4 modal expansion
```

## Trade-offs

- LR이 SVC보다 **decision boundary expressiveness 낮음** — kernel SVC가 더 잘 맞는 경우 metric이 약간 떨어질 수 있음. 다만 probe는 *encoder representation의 quality 비교*가 목적이지 absolute SOTA 목적 아님 → linear separability로 충분.
- Sampling이 **per-epoch 결과 variance 증가**시킬 수 있음 — seed=epoch 으로 재현성 확보, 보고 시 마지막 K epoch 평균.
- Must-include 항목 수가 많은 모달에서 비대 — N=10이면 must-include만 11개. 그 경우 `max_subsets`를 자동 조정하거나 must-include를 줄여야 함 (현재는 N≤9까지 안전).

## 원본

- 코드: `pretrained/physiome/probe_utils.py`, 호출 사이트: `pretrained/physiome/train.py`, `train_hetero.py`
- 검증: `experiments/smoke_test_probe.py`
- 결정 메모: `memory/project_physiome_state.md` "Probe efficiency" extension lever
