"""Domain-specific quality checks for biosignals.

Ported from references/Biosignal-Foundation-Model/data/parser/_quality_checks.py
(2026-05-04). Each check inspects physiology-aware features (peak rhythm,
autocorrelation regularity, flatline/clipping ratios) and returns a dict with
a ``pass`` bool plus diagnostic fields.

Used by ``vital_db_ssl.py`` to validate fixed-window segments per modality
before bucket assignment.
"""
from __future__ import annotations

import numpy as np


# ── helpers ──────────────────────────────────────────────────────


def _autocorrelation_peak(
    segment: np.ndarray,
    sr: float,
    min_lag_s: float,
    max_lag_s: float,
) -> float:
    """Maximum normalized autocorrelation within an HR-range lag window.

    Periodic signals show a clear peak (>0.3) at the period lag; random noise
    decays quickly and produces no peak.
    """
    x = segment - np.mean(segment)
    n = len(x)
    autocorr_full = np.correlate(x, x, mode="full")
    zero_lag = autocorr_full[n - 1]
    if zero_lag < 1e-10:
        return 0.0
    autocorr = autocorr_full[n - 1:] / zero_lag

    min_lag = max(1, int(min_lag_s * sr))
    max_lag = min(len(autocorr) - 1, int(max_lag_s * sr))
    if min_lag >= max_lag:
        return 0.0
    return float(np.max(autocorr[min_lag: max_lag + 1]))


# ── ECG ──────────────────────────────────────────────────────────


def ecg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.7,
    min_autocorr: float = 0.10,
) -> dict:
    """ECG QRS-peak based quality check (HR + regularity + autocorr)."""
    from scipy.signal import find_peaks

    fail = {
        "hr": 0.0, "hr_valid": False, "n_peaks": 0,
        "regularity": 1.0, "autocorr_peak": 0.0, "pass": False,
    }
    if len(segment) < int(sr * 2):
        return fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return fail

    min_distance = max(int(sr * 60.0 / max_hr * 0.8), 1)
    peaks, _ = find_peaks(segment, prominence=iqr * 0.5, distance=min_distance)

    n_peaks = len(peaks)
    if n_peaks < 2:
        fail["n_peaks"] = n_peaks
        return fail

    rr_intervals = np.diff(peaks) / sr
    rr_mean = float(np.mean(rr_intervals))
    if rr_mean < 1e-6:
        fail["n_peaks"] = n_peaks
        return fail

    hr = 60.0 / rr_mean
    hr_valid = min_hr <= hr <= max_hr
    rr_std = float(np.std(rr_intervals))
    regularity = rr_std / rr_mean
    autocorr_peak = _autocorrelation_peak(segment, sr,
                                          60.0 / max_hr, 60.0 / min_hr)
    passed = (hr_valid
              and regularity < regularity_threshold
              and autocorr_peak >= min_autocorr)
    return {
        "hr": round(hr, 1), "hr_valid": hr_valid, "n_peaks": n_peaks,
        "regularity": round(regularity, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "pass": passed,
    }


# ── ABP / PPG / PAP — pulse rhythm checks ────────────────────────


def _pulse_quality_check(
    segment: np.ndarray,
    sr: float,
    min_hr: float,
    max_hr: float,
    regularity_threshold: float,
    min_autocorr: float,
    min_distance_ratio: float = 0.8,
) -> dict:
    from scipy.signal import find_peaks

    fail = {
        "hr": 0.0, "n_peaks": 0,
        "regularity": 1.0, "autocorr_peak": 0.0, "pass": False,
    }
    if len(segment) < int(sr * 2):
        return fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return fail

    min_distance = max(int(sr * 60.0 / max_hr * min_distance_ratio), 1)
    peaks, _ = find_peaks(segment, prominence=iqr * 0.5, distance=min_distance)
    if len(peaks) < 2:
        fail["n_peaks"] = len(peaks)
        return fail

    pp = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp))
    if pp_mean < 1e-6:
        fail["n_peaks"] = len(peaks)
        return fail

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr
    pp_std = float(np.std(pp))
    regularity = pp_std / pp_mean
    autocorr_peak = _autocorrelation_peak(segment, sr,
                                          60.0 / max_hr, 60.0 / min_hr)
    passed = (hr_valid
              and regularity < regularity_threshold
              and autocorr_peak >= min_autocorr)
    return {
        "hr": round(hr, 1), "n_peaks": len(peaks),
        "regularity": round(regularity, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "pass": passed,
    }


def abp_quality_check(segment, sr=100.0, min_hr=30.0, max_hr=200.0,
                      regularity_threshold=0.5, min_autocorr=0.10) -> dict:
    return _pulse_quality_check(segment, sr, min_hr, max_hr,
                                regularity_threshold, min_autocorr,
                                min_distance_ratio=1.0)


def ppg_quality_check(segment, sr=100.0, min_hr=30.0, max_hr=200.0,
                      regularity_threshold=0.5, min_autocorr=0.10) -> dict:
    return _pulse_quality_check(segment, sr, min_hr, max_hr,
                                regularity_threshold, min_autocorr,
                                min_distance_ratio=0.8)


def pap_quality_check(segment, sr=100.0, min_hr=30.0, max_hr=200.0,
                      regularity_threshold=0.7, min_autocorr=0.10) -> dict:
    return _pulse_quality_check(segment, sr, min_hr, max_hr,
                                regularity_threshold, min_autocorr,
                                min_distance_ratio=1.0)


# ── CVP / ICP — venous / intracranial pulsation ──────────────────


def _venous_quality_check(
    segment: np.ndarray, sr: float,
    min_hr: float, max_hr: float,
    regularity_threshold: float,
    max_flatline_ratio: float, min_autocorr: float,
) -> dict:
    from scipy.signal import find_peaks

    fail = {
        "hr": 0.0, "n_peaks": 0, "regularity": 1.0,
        "flatline_ratio": 1.0, "autocorr_peak": 0.0, "pass": False,
    }
    if len(segment) < int(sr * 2):
        return fail

    diffs = np.diff(segment)
    flatline_ratio = float(np.sum(np.abs(diffs) < 1e-4)) / max(len(diffs), 1)
    if flatline_ratio >= max_flatline_ratio:
        fail["flatline_ratio"] = round(flatline_ratio, 4)
        return fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.1:
        fail["flatline_ratio"] = round(flatline_ratio, 4)
        return fail

    min_distance = max(int(sr * 60.0 / max_hr * 0.8), 1)
    peaks, _ = find_peaks(segment, prominence=iqr * 0.3, distance=min_distance)
    if len(peaks) < 2:
        fail.update({"flatline_ratio": round(flatline_ratio, 4),
                     "n_peaks": len(peaks)})
        return fail

    pp = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp))
    if pp_mean < 1e-6:
        fail.update({"flatline_ratio": round(flatline_ratio, 4),
                     "n_peaks": len(peaks)})
        return fail

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr
    pp_std = float(np.std(pp))
    regularity = pp_std / pp_mean
    autocorr_peak = _autocorrelation_peak(segment, sr,
                                          60.0 / max_hr, 60.0 / min_hr)
    passed = (hr_valid
              and regularity < regularity_threshold
              and autocorr_peak >= min_autocorr)
    return {
        "hr": round(hr, 1), "n_peaks": len(peaks),
        "regularity": round(regularity, 4),
        "flatline_ratio": round(flatline_ratio, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "pass": passed,
    }


def cvp_quality_check(segment, sr=100.0, min_hr=30.0, max_hr=200.0,
                      regularity_threshold=0.7, max_flatline_ratio=0.3,
                      min_autocorr=0.15) -> dict:
    return _venous_quality_check(segment, sr, min_hr, max_hr,
                                 regularity_threshold,
                                 max_flatline_ratio, min_autocorr)


def icp_quality_check(segment, sr=100.0, min_hr=30.0, max_hr=200.0,
                      regularity_threshold=0.7, max_flatline_ratio=0.3,
                      min_autocorr=0.15) -> dict:
    return _venous_quality_check(segment, sr, min_hr, max_hr,
                                 regularity_threshold,
                                 max_flatline_ratio, min_autocorr)

# ── Respiration-based (CO2 / AWP) ────────────────────────────────


def _respiration_quality_check(segment, sr, min_rr, max_rr,
                               iqr_threshold) -> dict:
    from scipy.signal import find_peaks

    duration_s = len(segment) / sr
    if duration_s < 5.0:
        return {"resp_rate": 0.0, "pass": False}

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < iqr_threshold:
        return {"resp_rate": 0.0, "pass": False}

    min_distance = max(int(sr * 60.0 / max_rr * 0.8), 1)
    peaks, _ = find_peaks(segment, prominence=iqr * 0.5, distance=min_distance)
    if len(peaks) < 2:
        return {"resp_rate": 0.0, "pass": False}

    intervals = np.diff(peaks) / sr
    mean_interval = float(np.mean(intervals))
    if mean_interval < 1e-6:
        return {"resp_rate": 0.0, "pass": False}

    rr = 60.0 / mean_interval
    return {"resp_rate": round(rr, 1), "pass": min_rr <= rr <= max_rr}


def co2_quality_check(segment, sr=100.0, min_rr=4.0, max_rr=40.0) -> dict:
    return _respiration_quality_check(segment, sr, min_rr, max_rr, 0.5)


def awp_quality_check(segment, sr=100.0, min_rr=4.0, max_rr=40.0) -> dict:
    return _respiration_quality_check(segment, sr, min_rr, max_rr, 0.5)


# ── Dispatcher ───────────────────────────────────────────────────


DOMAIN_QUALITY_CHECKS = {
    "ecg": ecg_quality_check,
    "abp": abp_quality_check,
    "ppg": ppg_quality_check,
    "cvp": cvp_quality_check,
    "co2": co2_quality_check,
    "awp": awp_quality_check,
    "pap": pap_quality_check,
    "icp": icp_quality_check,
}


def domain_quality_check(stype_key: str, segment: np.ndarray,
                         sr: float = 100.0) -> dict:
    """Dispatch to the appropriate check; unknown signals always pass."""
    fn = DOMAIN_QUALITY_CHECKS.get(stype_key)
    if fn is None:
        return {"pass": True}
    try:
        return fn(segment, sr)
    except Exception:
        return {"pass": True}
