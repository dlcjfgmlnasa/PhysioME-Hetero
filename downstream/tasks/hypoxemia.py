# -*- coding:utf-8 -*-
"""Intra-operative hypoxemia downstream task.

Mirrors the IOH task in ``hypotension.py``: slide fixed-length input windows
over the recording, label each by whether sustained hypoxemia (SpO2 below
threshold for ``sustained_sec``) occurs within the next ``horizon_sec``
seconds. Inputs are ABP/ECG/PPG/CVP — same as IOH; the SpO2 channel saved
by ``vital_db_downstream.py`` is label-only and never fed to the model.

Default config (Lundberg et al. Anesthesiology 2018-style):
    window_sec    = 60 s
    stride_sec    = 60 s
    horizon_sec   = 300 s   (5 min lookahead)
    spo2_threshold = 92.0   (%)
    sustained_sec = 30 s    (≥30 s sustained SpO2 < threshold)
    spo2_win_sec  = 5 s     (sub-window for averaging future SpO2)
    artifact_spo2_range = (50.0, 100.0)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HypoxemiaSample:
    input_signals: Dict[str, np.ndarray]   # {'abp': (T,), ...}
    label: int                             # 0=normal, 1=sustained hypoxemia
    label_value: float                     # min future SpO2 (%) — informational
    case_id: str
    win_start_sec: float
    horizon_sec: float


def _has_sustained_hypoxemia(future_spo2: Sequence[float],
                             threshold: float,
                             min_consecutive: int) -> bool:
    consecutive = 0
    for v in future_spo2:
        if v < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


def _load_case_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load one case-level npz produced by ``vital_db_downstream.py``.
    Requires SpO2 (label channel) to be present.
    """
    with np.load(path, allow_pickle=True) as arr:
        if 'spo2' not in arr.files:
            return None
        signals: Dict[str, np.ndarray] = {}
        for m in ('abp', 'ecg', 'ppg', 'cvp', 'co2', 'awp'):
            if m in arr.files:
                signals[m] = np.asarray(arr[m], dtype=np.float32)
        spo2 = np.asarray(arr['spo2'], dtype=np.float32)
        sfreq = int(arr['sfreq'])
        case_id = str(arr['case_id'])
    if not signals:
        return None
    return {
        'signals': signals,
        'spo2': spo2,
        'sfreq': sfreq,
        'case_id': case_id,
    }


def load_cases(data_dir: str,
               input_signals: Sequence[str],
               min_duration_sec: float = 1200.0,
               max_subjects: Optional[int] = None) -> List[Dict]:
    """Load preprocessed cases that contain SpO2 (label) plus at least one of
    ``input_signals``.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if max_subjects is not None:
        paths = paths[:max_subjects]
    cases: List[Dict] = []
    requested = set(s.lower() for s in input_signals)
    for p in paths:
        loaded = _load_case_npz(p)
        if loaded is None:
            continue
        signals = loaded['signals']
        # Need at least one of the requested input modalities to do inference.
        if not (requested & set(signals.keys())):
            continue
        # Trim everything to the common length so per-modal alignment + label
        # alignment are preserved.
        all_arrays = list(signals.values()) + [loaded['spo2']]
        min_len = min(a.shape[0] for a in all_arrays)
        sfreq = loaded['sfreq']
        if min_len < int(min_duration_sec * sfreq):
            continue
        signals = {k: v[:min_len] for k, v in signals.items()}
        spo2 = loaded['spo2'][:min_len]
        cases.append({
            'case_id': loaded['case_id'],
            'signals': signals,
            'spo2': spo2,
            'sfreq': sfreq,
        })
    return cases


def extract_hypoxemia_samples(
    cases: Sequence[Dict],
    input_signals: Sequence[str],
    window_sec: float = 60.0,
    stride_sec: float = 60.0,
    horizon_sec: float = 300.0,
    spo2_threshold: float = 92.0,
    sustained_sec: float = 30.0,
    spo2_win_sec: float = 5.0,
    artifact_spo2_range: Tuple[float, float] = (50.0, 100.0),
) -> List[HypoxemiaSample]:
    """Slide ``window_sec`` input windows with ``stride_sec`` stride and label
    each by whether SpO2 < ``spo2_threshold`` is sustained for ``sustained_sec``
    within the next ``horizon_sec`` seconds.

    Per sub-window (``spo2_win_sec``) SpO2 is averaged; sub-windows whose mean
    SpO2 falls outside ``artifact_spo2_range`` are discarded as artifacts.
    """
    samples: List[HypoxemiaSample] = []
    for case in cases:
        signals = case['signals']
        spo2 = case['spo2']
        sfreq = case['sfreq']
        n_total = spo2.shape[0]

        win = int(window_sec * sfreq)
        stride = int(stride_sec * sfreq)
        horizon = int(horizon_sec * sfreq)
        spo2_win = int(spo2_win_sec * sfreq)
        min_consecutive = max(1, int(sustained_sec / spo2_win_sec))
        total_needed = win + horizon

        if n_total < total_needed:
            continue

        for start in range(0, n_total - total_needed + 1, stride):
            input_dict: Dict[str, np.ndarray] = {}
            for stype in input_signals:
                key = stype.lower()
                if key in signals:
                    seg = signals[key][start: start + win]
                    # Skip windows where this modality is mostly NaN; for
                    # cases where every requested modality is dropped, we
                    # bail later.
                    if np.isnan(seg).mean() <= 0.1:
                        input_dict[key] = seg
            if not input_dict:
                continue

            future_start = start + win
            future_end = future_start + horizon
            future_spo2 = spo2[future_start:future_end]

            future_vals: List[float] = []
            for j in range(0, len(future_spo2) - spo2_win + 1, spo2_win):
                w = future_spo2[j: j + spo2_win]
                if np.isnan(w).any():
                    continue
                m = float(np.mean(w))
                if m < artifact_spo2_range[0] or m > artifact_spo2_range[1]:
                    continue
                future_vals.append(m)

            if len(future_vals) < max(1, min_consecutive // 2):
                continue

            label = int(_has_sustained_hypoxemia(
                future_vals, spo2_threshold, min_consecutive))
            min_future_spo2 = float(min(future_vals))

            samples.append(HypoxemiaSample(
                input_signals=input_dict, label=label,
                label_value=min_future_spo2,
                case_id=case['case_id'],
                win_start_sec=start / sfreq,
                horizon_sec=horizon_sec,
            ))
    return samples


class HypoxemiaDataset(Dataset):
    """Wrap a list of :class:`HypoxemiaSample` for PyTorch DataLoader use.

    Yields ``(data_dict, label)`` where ``data_dict`` keys are uppercase
    modality names ('ABP', 'ECG', 'PPG', 'CVP') matching ``PhysioME.modal_names``.
    """

    def __init__(self, samples: List[HypoxemiaSample],
                 modal_order: Sequence[str] = ('ABP', 'ECG', 'PPG', 'CVP')):
        self.samples = samples
        self.modal_order = list(modal_order)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data: Dict[str, torch.Tensor] = {}
        for m in self.modal_order:
            key = m.lower()
            if key in s.input_signals:
                data[m] = torch.from_numpy(s.input_signals[key]).float()
        label = torch.tensor(s.label, dtype=torch.long)
        return data, label


__all__ = [
    'HypoxemiaSample',
    'HypoxemiaDataset',
    'load_cases',
    'extract_hypoxemia_samples',
]
