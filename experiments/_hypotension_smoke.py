"""Smoke test for downstream/tasks/hypotension.py label generation.

Constructs synthetic ABP signals with explicit hypotension events and
asserts ``_has_sustained_hypotension`` / ``extract_forecast_samples``
pick them up correctly.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
from downstream.tasks.hypotension import (
    _has_sustained_hypotension, extract_forecast_samples,
)


def _build_case(map_trace_per_10s, sfreq=100):
    """Build an ABP signal whose 10s-window mean equals each value in
    ``map_trace_per_10s``."""
    arr = np.concatenate([np.full(10 * sfreq, m, dtype=np.float32)
                          for m in map_trace_per_10s])
    return {
        'case_id': 'synth',
        'sfreq': sfreq,
        'signals': {'abp': arr, 'ecg': np.zeros_like(arr)},
    }


def main() -> None:
    # _has_sustained_hypotension unit tests
    assert _has_sustained_hypotension([70, 60, 60, 60, 60, 60, 60, 70], 65, 6)
    assert not _has_sustained_hypotension([60, 60, 60, 70, 60, 60], 65, 6)

    # extract_forecast_samples — simple cases.
    # window_sec=30, horizon_sec=300 → need 30+300 = 330s minimum signal.
    # Build an 800s signal: 330s normal then 60s hypotension then rest normal.
    # First 30s window starts at t=0; future = (30, 330) — fully normal → label 0.
    map_trace = [80] * 33 + [60] * 6 + [80] * 41  # 33+6+41 = 80 windows of 10s = 800s
    case = _build_case(map_trace)
    samples = extract_forecast_samples(
        [case], input_signals=['abp', 'ecg'],
        window_sec=30, stride_sec=30, horizon_sec=300,
    )
    assert samples, 'no samples produced'
    # Label of the first window: future window covers t=30..330s, all MAP=80 → 0.
    assert samples[0].label == 0, f'expected label 0, got {samples[0].label}'

    # Find a window whose horizon includes the 60-window of hypotension.
    # Hypotension occurs at t=330..390s. Window starting at t=60s has horizon
    # t=90..390s, which includes the full hypotensive period → label 1.
    matching = [s for s in samples if abs(s.win_start_sec - 60) < 1e-3]
    assert matching, 'expected window starting at t=60s'
    assert matching[0].label == 1, (
        f'expected label 1 at t=60s, got {matching[0].label}')

    print(f'[hypotension_smoke] {len(samples)} samples produced')
    n_pos = sum(1 for s in samples if s.label == 1)
    print(f'  positives: {n_pos}/{len(samples)}')
    print('[hypotension_smoke] PASSED')


if __name__ == '__main__':
    main()
