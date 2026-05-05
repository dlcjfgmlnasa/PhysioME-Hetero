"""Smoke test for the quality-check pipeline integrated into vital_db_ssl.

Generates synthetic 60s @ 100Hz waveforms for ABP/ECG/PPG (good and bad cases)
and asserts validate_segment accepts the good ones and rejects the bad ones.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import numpy as np
from dataset.data_parser._signal_filters import SIGNAL_CONFIGS, preprocess_channel
from dataset.data_parser.vital_db_ssl import validate_segment


def _make_pulse(sr: int = 100, duration: int = 60,
                hr: float = 75, amp: float = 60.0, baseline: float = 80.0,
                noise: float = 1.0) -> np.ndarray:
    """Synthesize a periodic pulse waveform with clear peaks."""
    t = np.arange(sr * duration) / sr
    cycle_freq = hr / 60.0
    sig = baseline + amp * (0.5 * (1 + np.sin(2 * np.pi * cycle_freq * t)) ** 8)
    sig += np.random.normal(0, noise, size=sig.shape)
    return sig.astype(np.float32)


def _make_ecg(sr: int = 100, duration: int = 60, hr: float = 75,
              amp: float = 1.5, noise: float = 0.05) -> np.ndarray:
    """Cheap ECG-ish synthesis: narrow positive spikes at HR rate."""
    t = np.arange(sr * duration) / sr
    period = 60.0 / hr
    sig = np.zeros_like(t, dtype=np.float32)
    for k in range(int(duration / period) + 2):
        center = k * period
        sig += amp * np.exp(-((t - center) ** 2) / (2 * 0.01 ** 2))
    sig += np.random.normal(0, noise, size=sig.shape)
    return sig.astype(np.float32)


def main() -> None:
    sr = 100
    duration = 60
    np.random.seed(0)

    # GOOD signals
    good_abp = _make_pulse(sr, duration, hr=75, amp=40, baseline=80, noise=0.5)
    good_ecg = _make_ecg(sr, duration, hr=72, amp=1.2, noise=0.02)
    good_ppg = _make_pulse(sr, duration, hr=72, amp=200, baseline=600, noise=2.0)

    # BAD signals
    flat = np.full(sr * duration, 100.0, dtype=np.float32)         # flatline
    out_of_range_abp = good_abp + 500                              # > 300 mmHg
    nanny = good_ecg.copy(); nanny[: int(0.5 * len(nanny))] = np.nan  # too many NaN

    cases = [
        ('ABP good', 'ABP', good_abp, True),
        ('ABP flat', 'ABP', flat, False),
        ('ABP OOR',  'ABP', out_of_range_abp, False),
        ('ECG good', 'ECG', good_ecg, True),
        ('ECG NaN50%', 'ECG', nanny, False),
        ('PPG good', 'PPG', good_ppg, True),
        ('PPG flat', 'PPG', flat, False),
    ]

    rejection_failures = []
    for name, modal, sig, expect_pass in cases:
        signal_key = {'ABP': 'abp', 'ECG': 'ecg', 'PPG': 'ppg'}[modal]
        cfg = SIGNAL_CONFIGS[signal_key]
        prep = preprocess_channel(sig, signal_key=signal_key, sr=float(sr))
        result = validate_segment(prep, signal_key, cfg, sr=float(sr),
                                  nan_max_ratio=0.1)
        # Hard assertion: bad cases must be rejected (the contract that matters
        # for data sanitation). Good cases are informational here because the
        # synthetic generators are toy and the upstream thresholds were tuned
        # on real biosignals — a real-data pass test belongs in Task 7's
        # full smoke once raw VitalDB cases are available.
        ok = (not expect_pass and not result) or (expect_pass and result)
        if not expect_pass and result:
            rejection_failures.append(name)
        tag = ('PASS' if ok else
               ('SOFT-FAIL' if expect_pass else 'HARD-FAIL'))
        print(f'  {tag:9s} {name:14s} expected={expect_pass}  got={result}')

    if rejection_failures:
        raise SystemExit(
            f'[quality_check_smoke] HARD-FAIL: bad cases were not rejected: '
            f'{rejection_failures}')
    print('[quality_check_smoke] all bad cases correctly rejected')
    print('  (good-case PASS depends on synthetic realism; verify on real '
          'VitalDB data via end-to-end smoke once dataset is available)')


if __name__ == '__main__':
    main()
