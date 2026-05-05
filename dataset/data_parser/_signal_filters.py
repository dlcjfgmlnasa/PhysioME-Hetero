"""Per-signal preprocessing pipeline (range → spike → median → notch → filter).

Ported from references/Biosignal-Foundation-Model/data/parser/vitaldb.py
(2026-05-04), keeping the configurations the upstream project arrived at after
empirical tuning.

The full ``SIGNAL_CONFIGS`` table covers 8 signal types — we only use ABP/ECG/PPG
in PhysioME-Hetero v1, but keeping the full table makes future modality
extensions a one-line change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ── SignalConfig ─────────────────────────────────────────────────


@dataclass
class SignalConfig:
    """Per-signal preprocessing parameters."""
    valid_range: Optional[Tuple[float, float]]
    filter_type: str = "none"            # "bandpass" | "lowpass" | "none"
    filter_freq: Optional[Tuple[float, float]] = None
    max_flatline_ratio: float = 0.5
    max_clip_ratio: float = 0.1
    max_high_freq_ratio: float = 2.0
    min_amplitude: float = 0.0
    max_amplitude: float = 0.0
    min_high_freq_ratio: float = 0.0
    notch_freq: Optional[float] = None
    spike_detection: bool = False
    spike_threshold_std: float = 10.0
    median_kernel: int = 0
    quality_window_s: float = 5.0


SIGNAL_CONFIGS: dict[str, SignalConfig] = {
    "ecg": SignalConfig(
        valid_range=(-5.0, 5.0),
        filter_type="bandpass", filter_freq=(0.5, 40.0),
        max_high_freq_ratio=1.0,
        min_amplitude=0.3,
        min_high_freq_ratio=0.05,
        notch_freq=60.0,
        spike_detection=True, spike_threshold_std=10.0,
    ),
    "abp": SignalConfig(
        valid_range=(20.0, 300.0),
        filter_type="lowpass", filter_freq=(0.0, 15.0),
        max_high_freq_ratio=0.5,
        max_flatline_ratio=0.3,
        min_amplitude=10.0,
        spike_detection=True, spike_threshold_std=6.0,
        median_kernel=5,
    ),
    "ppg": SignalConfig(
        valid_range=(0.0, 2000.0),
        filter_type="lowpass", filter_freq=(0.0, 8.0),
        max_high_freq_ratio=0.05,
        max_flatline_ratio=0.3,
        min_amplitude=5.0,
        notch_freq=60.0,
        spike_detection=True, spike_threshold_std=6.0,
        median_kernel=5,
    ),
    "cvp": SignalConfig(
        valid_range=(-5.0, 40.0),
        filter_type="lowpass", filter_freq=(0.0, 10.0),
        max_high_freq_ratio=0.5,
        spike_detection=True, spike_threshold_std=8.0,
    ),
    "co2": SignalConfig(
        valid_range=(0.0, 100.0),
        filter_type="lowpass", filter_freq=(0.0, 5.0),
        max_high_freq_ratio=1.0,
        max_flatline_ratio=0.3,
        min_amplitude=5.0,
        quality_window_s=15.0,
    ),
    "awp": SignalConfig(
        valid_range=(-20.0, 80.0),
        filter_type="lowpass", filter_freq=(0.0, 20.0),
        max_high_freq_ratio=1.0,
        min_amplitude=2.0,
        quality_window_s=15.0,
    ),
    "pap": SignalConfig(
        valid_range=(5.0, 80.0),
        filter_type="lowpass", filter_freq=(0.0, 15.0),
        max_high_freq_ratio=0.5,
        max_flatline_ratio=0.3,
        min_amplitude=5.0,
        spike_detection=True, spike_threshold_std=6.0,
        median_kernel=5,
    ),
    "icp": SignalConfig(
        valid_range=(-10.0, 80.0),
        filter_type="lowpass", filter_freq=(0.0, 10.0),
        max_high_freq_ratio=0.5,
        max_flatline_ratio=0.3,
        min_amplitude=1.0,
        spike_detection=True, spike_threshold_std=8.0,
    ),
}


# ── Filters ──────────────────────────────────────────────────────


def apply_range_check(data: np.ndarray, valid_range: Tuple[float, float]
                      ) -> tuple[np.ndarray, int]:
    """Out-of-range samples → NaN. Returns (out, n_bad)."""
    lo, hi = valid_range
    out = data.copy()
    bad = (out < lo) | (out > hi)
    n_bad = int(bad.sum())
    if n_bad > 0:
        out[bad] = np.nan
    return out, n_bad


def apply_notch_filter(data: np.ndarray, freq: float, sr: float,
                       Q: float = 30.0) -> np.ndarray:
    from scipy.signal import filtfilt, iirnotch
    nyq = sr / 2.0
    if freq >= nyq:
        return data
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, data).astype(data.dtype)


def apply_median_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    from scipy.signal import medfilt
    if kernel_size < 3:
        return data
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(data, kernel_size=kernel_size).astype(data.dtype)


def apply_bandpass(data: np.ndarray, lo: float, hi: float,
                   sr: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= lo:
        return data
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def apply_lowpass(data: np.ndarray, hi: float, sr: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= 0:
        return data
    sos = butter(4, hi / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def apply_filter(data: np.ndarray, cfg: SignalConfig, sr: float) -> np.ndarray:
    if cfg.filter_type == "bandpass" and cfg.filter_freq is not None:
        return apply_bandpass(data, cfg.filter_freq[0], cfg.filter_freq[1], sr)
    if cfg.filter_type == "lowpass" and cfg.filter_freq is not None:
        return apply_lowpass(data, cfg.filter_freq[1], sr)
    return data


def detect_electrocautery(data: np.ndarray, sr: float,
                          threshold_std: float = 10.0,
                          blank_ms: float = 100.0
                          ) -> tuple[np.ndarray, int]:
    """ECG-style spike detection: |Δsignal| > k·MAD → blank ±blank_ms."""
    out = data.copy()
    diff = np.abs(np.diff(out, prepend=out[0]))
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) * 1.4826
    if mad < 1e-10:
        return out, 0
    spike_mask = diff > (med + threshold_std * mad)
    if not spike_mask.any():
        return out, 0
    blank = int(blank_ms / 1000.0 * sr)
    for idx in np.where(spike_mask)[0]:
        s, e = max(0, idx - blank), min(len(out), idx + blank + 1)
        out[s:e] = np.nan
    return out, int(np.isnan(out).sum() - np.isnan(data).sum())


def detect_step_change(data: np.ndarray, sr: float,
                       threshold_std: float = 6.0,
                       blank_ms: float = 300.0
                       ) -> tuple[np.ndarray, int]:
    """ABP/PPG step artifact (sensor dropout / clip) → NaN."""
    return detect_electrocautery(data, sr, threshold_std, blank_ms)


# ── Quality score on a fixed segment ─────────────────────────────


def segment_quality_score(
    segment: np.ndarray,
    max_flatline_ratio: float = 0.5,
    max_clip_ratio: float = 0.1,
    max_high_freq_ratio: float = 2.0,
    min_amplitude: float = 0.0,
    max_amplitude: float = 0.0,
    min_high_freq_ratio: float = 0.0,
) -> dict:
    """flatline + clip + high-freq + amplitude based segment-level quality."""
    n = len(segment)
    if n < 2:
        return {"flatline_ratio": 1.0, "clip_ratio": 1.0,
                "high_freq_ratio": 1.0, "amplitude": 0.0, "pass": False}

    diffs = np.diff(segment)
    flatline_ratio = float(np.sum(np.abs(diffs) < 1e-4)) / max(len(diffs), 1)

    smin, smax = float(segment.min()), float(segment.max())
    if smax - smin < 1e-10:
        clip_ratio = 1.0
    else:
        at_min = np.sum(np.abs(segment - smin) < 1e-8)
        at_max = np.sum(np.abs(segment - smax) < 1e-8)
        clip_ratio = float(at_min + at_max) / n

    sig_energy = float(np.mean(segment ** 2))
    diff_energy = float(np.mean(diffs ** 2))
    high_freq_ratio = (diff_energy / sig_energy) if sig_energy >= 1e-10 else 1.0
    amplitude = smax - smin

    passed = (flatline_ratio < max_flatline_ratio
              and clip_ratio < max_clip_ratio
              and high_freq_ratio < max_high_freq_ratio
              and amplitude >= min_amplitude
              and (max_amplitude <= 0 or amplitude <= max_amplitude)
              and high_freq_ratio >= min_high_freq_ratio)
    return {
        "flatline_ratio": flatline_ratio, "clip_ratio": clip_ratio,
        "high_freq_ratio": high_freq_ratio, "amplitude": amplitude,
        "pass": passed,
    }


def resample_to_target(signal: np.ndarray, orig_sr: float,
                       target_sr: float = 100.0) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    from scipy.signal import resample_poly
    gcd = math.gcd(int(target_sr), int(orig_sr))
    up = int(target_sr) // gcd
    down = int(orig_sr) // gcd
    if signal.ndim == 1:
        return resample_poly(signal, up, down, axis=0).astype(signal.dtype)
    return resample_poly(signal, up, down, axis=1).astype(signal.dtype)


# ── Per-channel pipeline ─────────────────────────────────────────


def preprocess_channel(data: np.ndarray, signal_key: str,
                       sr: float = 100.0,
                       cfg: Optional[SignalConfig] = None) -> np.ndarray:
    """Run the full preprocessing pipeline for one signal channel.

    Order: range check → spike detection → median → notch → filter.
    Out-of-range / spike samples become NaN — downstream code is expected to
    treat NaN as missing (e.g., zero-fill plus mask).
    """
    cfg = cfg or SIGNAL_CONFIGS.get(signal_key)
    if cfg is None:
        return data
    out = data.astype(np.float32, copy=True)
    if cfg.valid_range is not None:
        out, _ = apply_range_check(out, cfg.valid_range)
    if cfg.spike_detection:
        out, _ = detect_electrocautery(out, sr, cfg.spike_threshold_std)
    # NaN-aware filters: fill with channel mean for filtering, then re-NaN.
    nan_mask = np.isnan(out)
    if nan_mask.any():
        mean_val = float(np.nanmean(out)) if not np.isnan(out).all() else 0.0
        out_fill = np.where(nan_mask, mean_val, out)
    else:
        out_fill = out
    if cfg.median_kernel >= 3:
        out_fill = apply_median_filter(out_fill, cfg.median_kernel)
    if cfg.notch_freq is not None:
        out_fill = apply_notch_filter(out_fill, cfg.notch_freq, sr)
    out_fill = apply_filter(out_fill, cfg, sr)
    # Restore NaN where original was NaN (artifact-marked stays artifact-marked).
    if nan_mask.any():
        out_fill = out_fill.astype(np.float32)
        out_fill[nan_mask] = np.nan
    return out_fill.astype(np.float32)
