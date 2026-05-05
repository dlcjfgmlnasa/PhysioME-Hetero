# -*- coding:utf-8 -*-
"""SSL pretraining parser for VitalDB.

Differences vs ``dataset/data_parser/vital_db.py``:
  1. No 2-minute lookahead validation — every valid window is an SSL sample.
  2. No ``is_all_in`` subject-level filter — subjects with any non-empty
     subset of {ABP, ECG, PPG} contribute data.
  3. Each segment carries a per-modality validity mask, so heterogeneous
     bucket sampling is downstream-friendly.
  4. Per-signal preprocessing pipeline (range check → spike detection →
     median → notch → bandpass/lowpass) ported from
     references/Biosignal-Foundation-Model.
  5. Segment validity uses three layered checks:
       (a) NaN ratio threshold,
       (b) generic quality score (flatline / clip / hf / amplitude) per signal,
       (c) physiology-aware domain check (HR + autocorr regularity etc.).

Output (per case, one ``.npz``):
    x: float32 [T, M, sfreq * duration], zero where ``mask[t, m] == False``.
    mask: bool [T, M] — per-segment, per-modality validity.
    modal_names: object [M] — fixed order from ``MODAL_ORDER``.
    subject_modality_set: bool [M] — modalities present anywhere in the recording.
    case_id: str
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import tqdm

from dataset.data_parser._quality_checks import domain_quality_check
from dataset.data_parser._signal_filters import (
    SIGNAL_CONFIGS,
    SignalConfig,
    preprocess_channel,
    segment_quality_score,
)


MODAL_TRACK_NAMES: Dict[str, str] = {
    'ABP': 'SNUADC/ART',
    'ECG': 'SNUADC/ECG_II',
    'PPG': 'SNUADC/PLETH',
    # Step 2 (2026-05-05): 4-modal expansion — CO2 (capnography / etCO2)
    'CO2': 'Primus/CO2',
}
# short-name → key used by SIGNAL_CONFIGS / domain_quality_check
MODAL_TO_SIGNAL_KEY: Dict[str, str] = {
    'ABP': 'abp', 'ECG': 'ecg', 'PPG': 'ppg',
    'CO2': 'co2',
}
MODAL_ORDER = ['ABP', 'ECG', 'PPG', 'CO2']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path',
                        default=os.path.join('..', '..', '..', '..', 'Dataset', 'vitaldb'))
    parser.add_argument('--trg_path',
                        default=os.path.join('..', '..', 'data', 'vitaldb_ssl'))
    parser.add_argument('--sfreq', type=int, default=100)
    parser.add_argument('--duration', type=int, default=60,
                        help='window length in seconds')
    parser.add_argument('--nan_max_ratio', type=float, default=0.1)
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='disable per-channel preprocessing pipeline '
                             '(useful when input is already preprocessed)')
    return parser.parse_args()


def validate_segment(segment: np.ndarray,
                     signal_key: str,
                     cfg: SignalConfig,
                     sr: float,
                     nan_max_ratio: float) -> bool:
    """Three-layer validity check for one fixed-length segment of one channel.

    1) NaN ratio under threshold.
    2) ``segment_quality_score`` with signal-specific thresholds
       (flatline, clip, high-freq, amplitude).
    3) ``domain_quality_check`` (HR / pulse / autocorr regularity).

    NaN values are filled with the segment median for the score-based checks;
    they remain NaN in the returned data (which is later zero-filled by the
    caller).
    """
    if segment.size == 0:
        return False
    nan_ratio = float(np.isnan(segment).mean())
    if nan_ratio > nan_max_ratio:
        return False

    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return False

    if np.isnan(segment).any():
        median_val = float(np.nanmedian(segment))
        seg_filled = np.where(np.isnan(segment), median_val, segment)
    else:
        seg_filled = segment

    score = segment_quality_score(
        seg_filled,
        max_flatline_ratio=cfg.max_flatline_ratio,
        max_clip_ratio=cfg.max_clip_ratio,
        max_high_freq_ratio=cfg.max_high_freq_ratio,
        min_amplitude=cfg.min_amplitude,
        max_amplitude=cfg.max_amplitude,
        min_high_freq_ratio=cfg.min_high_freq_ratio,
    )
    if not score['pass']:
        return False

    domain = domain_quality_check(signal_key, seg_filled, sr=sr)
    if not domain.get('pass', True):
        return False
    return True


def fill_segment(segment: np.ndarray, expected_len: int) -> np.ndarray:
    s = np.asarray(segment, dtype=np.float32).reshape(-1)
    if s.shape[0] < expected_len:
        s = np.pad(s, (0, expected_len - s.shape[0]))
    elif s.shape[0] > expected_len:
        s = s[:expected_len]
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def extract_segments(data_per_modal: Dict[str, np.ndarray],
                     sfreq: int, duration: int,
                     nan_max_ratio: float) -> tuple:
    expected = sfreq * duration
    if not data_per_modal:
        return None, None
    max_len = max(arr.shape[0] for arr in data_per_modal.values())
    num_windows = max_len // expected
    if num_windows == 0:
        return None, None

    xs = np.zeros((num_windows, len(MODAL_ORDER), expected), dtype=np.float32)
    masks = np.zeros((num_windows, len(MODAL_ORDER)), dtype=bool)

    kept = 0
    for w in range(num_windows):
        s, e = w * expected, (w + 1) * expected
        any_valid = False
        win_x = np.zeros((len(MODAL_ORDER), expected), dtype=np.float32)
        win_mask = np.zeros(len(MODAL_ORDER), dtype=bool)
        for i, m in enumerate(MODAL_ORDER):
            if m not in data_per_modal:
                continue
            arr = data_per_modal[m]
            if arr.shape[0] < e:
                continue
            seg = arr[s:e]
            signal_key = MODAL_TO_SIGNAL_KEY[m]
            cfg = SIGNAL_CONFIGS[signal_key]
            if validate_segment(seg, signal_key, cfg, float(sfreq),
                                nan_max_ratio):
                win_x[i] = fill_segment(seg, expected)
                win_mask[i] = True
                any_valid = True
        if any_valid:
            xs[kept] = win_x
            masks[kept] = win_mask
            kept += 1

    if kept == 0:
        return None, None
    return xs[:kept], masks[:kept]


def vitaldb_ssl_converter(src_path: str, trg_path: str,
                          sfreq: int = 100, duration: int = 60,
                          nan_max_ratio: float = 0.1,
                          skip_preprocess: bool = False) -> None:
    import vitaldb  # imported lazily so validate_segment can be used without it
    os.makedirs(trg_path, exist_ok=True)
    paths = sorted(os.listdir(src_path))

    saved, skipped = 0, 0
    bucket_counts: Dict[str, int] = {}

    for fname in tqdm.tqdm(paths, desc='VitalDB-SSL'):
        case_id = os.path.splitext(fname)[0]
        full_path = os.path.join(src_path, fname)
        try:
            present_tracks = vitaldb.vital_trks(full_path)
        except Exception:
            skipped += 1
            continue

        present_modals = [m for m, trk in MODAL_TRACK_NAMES.items()
                          if trk in present_tracks]
        if not present_modals:
            skipped += 1
            continue

        try:
            tracks = [MODAL_TRACK_NAMES[m] for m in present_modals]
            data = vitaldb.vital_recs(full_path, tracks, 1.0 / sfreq)
        except Exception:
            skipped += 1
            continue
        if data is None or data.size == 0:
            skipped += 1
            continue
        if data.ndim == 1:
            data = data[:, None]

        data_per_modal: Dict[str, np.ndarray] = {}
        for i, m in enumerate(present_modals):
            channel = np.asarray(data[:, i], dtype=np.float32)
            if not skip_preprocess:
                channel = preprocess_channel(
                    channel, signal_key=MODAL_TO_SIGNAL_KEY[m], sr=float(sfreq),
                )
            data_per_modal[m] = channel

        xs, masks = extract_segments(data_per_modal, sfreq, duration,
                                     nan_max_ratio)
        if xs is None:
            skipped += 1
            continue

        subject_modality_set = np.array(
            [m in present_modals for m in MODAL_ORDER], dtype=bool,
        )
        np.savez(
            os.path.join(trg_path, case_id + '.npz'),
            x=xs, mask=masks,
            modal_names=np.array(MODAL_ORDER),
            subject_modality_set=subject_modality_set,
            case_id=case_id,
        )

        bucket_key = ''.join('1' if subject_modality_set[i] else '0'
                             for i in range(len(MODAL_ORDER)))
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
        saved += 1

    print(f'[VitalDB-SSL] saved={saved}, skipped={skipped}')
    print(f'[VitalDB-SSL] bucket counts (ABP-ECG-PPG presence bitmap):')
    for k in sorted(bucket_counts.keys()):
        print(f'    {k}: {bucket_counts[k]}')


if __name__ == '__main__':
    args = get_args()
    vitaldb_ssl_converter(
        src_path=args.src_path,
        trg_path=args.trg_path,
        sfreq=args.sfreq,
        duration=args.duration,
        nan_max_ratio=args.nan_max_ratio,
        skip_preprocess=args.skip_preprocess,
    )
