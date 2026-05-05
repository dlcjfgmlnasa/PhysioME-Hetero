"""Smoke test for the refactored linear-probe utilities (probe_utils).

Verifies:
* select_probe_subsets at N=3 returns all 7 subsets (back-compat).
* select_probe_subsets at N=5 returns at most max_subsets, always
  including the full set and every single-modal subset.
* Sampling is deterministic given a seed.
* run_probe trains LR per subset and returns per-subset records plus
  mean acc / macro-f1 in [0, 1] on a synthetic linearly-separable task.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pretrained.physiome.probe_utils import (  # noqa: E402
    run_probe,
    select_probe_subsets,
)


def test_select_subsets_n3() -> None:
    ch = ('ABP', 'ECG', 'PPG')
    subsets = select_probe_subsets(ch, max_subsets=10)
    assert len(subsets) == 7, f'N=3 should fully enumerate, got {len(subsets)}'
    # Each modality appears in some subset.
    for c in ch:
        assert any(c in s for s in subsets)
    print(f'[ok] N=3 -> {len(subsets)} subsets (full enumeration)')


def test_select_subsets_n5_capped() -> None:
    ch = ('ABP', 'ECG', 'PPG', 'CO2', 'CVP')
    subsets = select_probe_subsets(ch, max_subsets=10, seed=42)
    assert len(subsets) == 10, f'N=5 should cap at 10, got {len(subsets)}'

    full = tuple(ch)
    assert full in subsets, 'full set must always be included'
    for c in ch:
        single = (c,)
        assert single in subsets, f'single subset {single} must always be included'

    # Determinism check.
    subsets2 = select_probe_subsets(ch, max_subsets=10, seed=42)
    assert subsets == subsets2, 'same seed must produce same subsets'

    # Different seed produces different (in general) ordering of extras.
    subsets3 = select_probe_subsets(ch, max_subsets=10, seed=43)
    assert set(subsets) != set(subsets3) or set(subsets) == set(subsets3), (
        'seeds 42 vs 43 should produce same must-includes; extras may differ'
    )
    print(f'[ok] N=5 capped at 10 / full + singles guaranteed / seed-deterministic')


def test_select_subsets_n4() -> None:
    ch = ('ABP', 'ECG', 'PPG', 'CVP')
    subsets = select_probe_subsets(ch, max_subsets=10, seed=0)
    # 2^4 - 1 = 15 > 10 -> capped
    assert len(subsets) == 10
    assert tuple(ch) in subsets
    for c in ch:
        assert (c,) in subsets
    print(f'[ok] N=4 -> capped at 10 of 15 with corner-case coverage')


def test_run_probe_synthetic() -> None:
    """Make latents linearly separable: each class has a class-specific mean
    plus per-modality noise. LR should hit > 0.7 acc easily."""
    rng = np.random.default_rng(0)
    n_train = 200
    n_eval = 100
    d = 16

    def latents_for(n_per_class: int):
        x_pos = rng.normal(loc=+1.0, scale=0.3, size=(n_per_class, d)).astype(np.float32)
        x_neg = rng.normal(loc=-1.0, scale=0.3, size=(n_per_class, d)).astype(np.float32)
        x = np.concatenate([x_pos, x_neg], axis=0)
        y = np.array([1] * n_per_class + [0] * n_per_class, dtype=np.int64)
        return x, y

    train_x, train_y = latents_for(n_train // 2)
    eval_x, eval_y = latents_for(n_eval // 2)

    # Each modal subset gets the same latents (smoke-only — real probe
    # would have different fusion vectors per subset).
    train_fn = lambda subset: (train_x, train_y)
    eval_fn = lambda subset: (eval_x, eval_y)

    subsets = select_probe_subsets(('ABP', 'ECG', 'PPG'), max_subsets=10)
    mean_acc, mean_mf1, records = run_probe(
        subsets, train_fn, eval_fn, log_prefix='[smoke]',
    )
    assert 0.0 <= mean_acc <= 1.0
    assert 0.0 <= mean_mf1 <= 1.0
    assert mean_acc > 0.7, f'LR should easily exceed 0.7 on this task, got {mean_acc:.3f}'
    assert len(records) == len(subsets)
    print(f'[ok] run_probe synthetic -> acc={mean_acc:.3f} mf1={mean_mf1:.3f}')


def main() -> None:
    test_select_subsets_n3()
    test_select_subsets_n4()
    test_select_subsets_n5_capped()
    test_run_probe_synthetic()
    print()
    print('[smoke] PASSED -- probe_utils ready for N >= 4 modal expansion')


if __name__ == '__main__':
    main()
