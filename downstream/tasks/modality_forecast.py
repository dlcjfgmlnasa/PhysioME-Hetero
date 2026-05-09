# -*- coding:utf-8 -*-
"""Self-modality forecast probe tasks (CVP / CO2 / AWP).

Mirrors ``downstream/tasks/hypotension.py`` but parameterised so that each
modality can be probed against a clinically-meaningful threshold-crossing
event computed from *itself*. Used by Phase-1 SSL training to monitor whether
the unimodal backbone is learning useful representations for sparse
modalities (CVP/CO2/AWP) where IOH labels are not directly applicable.

Preset tasks (``--task`` keys):
    'cvp'  — venous congestion: future mean CVP > 12 mmHg
    'co2'  — hypercapnia       : future mean EtCO2 > 50 mmHg
    'awp'  — high airway P     : future peak AWP > 30 cmH2O

All three share the IOH-style protocol:
    window_sec    = 60 s
    stride_sec    = 60 s
    horizon_sec   = 300 s    (5-min lookahead)
    sub_window_sec= 10 s     (mean/peak averaged inside this sub-window)
    sustained_sec = 30 s     (≥30 s consecutive sub-windows above threshold)

Public API:
    ForecastTaskConfig        — preset container.
    TASK_PRESETS              — {'cvp': ..., 'co2': ..., 'awp': ...}
    extract_modality_samples  — sweep windows + label by threshold-crossing.
    ModalityForecastDataset   — torch Dataset over the resulting samples.
    load_cases_modality       — case loader (per-modality npz channel).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Presets ──────────────────────────────────────────────────────


@dataclass
class ForecastTaskConfig:
    modal_name: str            # 'CVP' | 'CO2' | 'AWP'
    signal_key: str            # lowercase npz channel key
    threshold: float
    direction: Literal['above', 'below']
    aggregator: Literal['mean', 'peak']     # how the sub-window value is summarised
    sustained_sec: float = 30.0
    artifact_range: Tuple[float, float] = (-1e9, 1e9)
    label_name: str = ''       # human-readable, used in printouts


TASK_PRESETS: Dict[str, ForecastTaskConfig] = {
    'cvp': ForecastTaskConfig(
        modal_name='CVP', signal_key='cvp',
        threshold=12.0, direction='above', aggregator='mean',
        sustained_sec=30.0,
        artifact_range=(-5.0, 40.0),
        label_name='venous_congestion (mean CVP > 12 mmHg)',
    ),
    'co2': ForecastTaskConfig(
        modal_name='CO2', signal_key='co2',
        threshold=50.0, direction='above', aggregator='mean',
        sustained_sec=30.0,
        artifact_range=(0.0, 100.0),
        label_name='hypercapnia (mean EtCO2 > 50 mmHg)',
    ),
    'awp': ForecastTaskConfig(
        modal_name='AWP', signal_key='awp',
        threshold=30.0, direction='above', aggregator='peak',
        sustained_sec=30.0,
        artifact_range=(-20.0, 80.0),
        label_name='high_airway_pressure (peak AWP > 30 cmH2O)',
    ),
}


# ── Sample dataclass ─────────────────────────────────────────────


@dataclass
class ModalityForecastSample:
    input_signals: Dict[str, np.ndarray]   # {modality_name: (T,)}
    label: int                             # 0 = normal, 1 = sustained event
    label_value: float                     # max/min summary used to label
    case_id: str
    win_start_sec: float
    horizon_sec: float


# ── Label logic ──────────────────────────────────────────────────


def _has_sustained(values: Sequence[float], threshold: float,
                   direction: str, min_consecutive: int) -> bool:
    consecutive = 0
    if direction == 'above':
        cmp = lambda v: v > threshold
    else:
        cmp = lambda v: v < threshold
    for v in values:
        if cmp(v):
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


# ── Case loading ─────────────────────────────────────────────────


def _load_case_npz(path: str, signal_key: str) -> Optional[Dict[str, np.ndarray]]:
    """Load one case-level npz, requiring the modality channel to be present."""
    with np.load(path, allow_pickle=True) as arr:
        if signal_key not in arr.files:
            return None
        sig = np.asarray(arr[signal_key], dtype=np.float32)
        sfreq = int(arr['sfreq'])
        case_id = str(arr['case_id'])
    return {
        'signals': {signal_key: sig},
        'sfreq': sfreq,
        'case_id': case_id,
    }


def load_cases_modality(data_dir: str, task: ForecastTaskConfig,
                        min_duration_sec: float = 600.0,
                        max_subjects: Optional[int] = None) -> List[Dict]:
    """Return cases that contain the requested modality channel."""
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if max_subjects is not None:
        paths = paths[:max_subjects]
    cases: List[Dict] = []
    for p in paths:
        loaded = _load_case_npz(p, task.signal_key)
        if loaded is None:
            continue
        sig = loaded['signals'][task.signal_key]
        if sig.shape[0] < int(min_duration_sec * loaded['sfreq']):
            continue
        cases.append({'case_id': loaded['case_id'],
                      'signals': {task.signal_key: sig},
                      'sfreq': loaded['sfreq']})
    return cases


# ── Window / horizon extraction ─────────────────────────────────


def extract_modality_samples(
    cases: Sequence[Dict],
    task: ForecastTaskConfig,
    window_sec: float = 60.0,
    stride_sec: float = 60.0,
    horizon_sec: float = 300.0,
    sub_window_sec: float = 10.0,
) -> List[ModalityForecastSample]:
    """Slide windows + label by threshold-crossing in the future ``horizon_sec``.

    Sub-windows are summarised with ``task.aggregator`` (mean for CVP/CO2,
    peak for AWP — peak airway pressure is the clinical barotrauma metric).
    Sub-windows whose mean falls outside ``task.artifact_range`` are
    discarded before the sustained-event check.
    """
    samples: List[ModalityForecastSample] = []
    for case in cases:
        sfreq = case['sfreq']
        sig = case['signals'][task.signal_key]
        n_total = sig.shape[0]

        win = int(window_sec * sfreq)
        stride = int(stride_sec * sfreq)
        horizon = int(horizon_sec * sfreq)
        sub_win = int(sub_window_sec * sfreq)
        min_consecutive = max(1, int(task.sustained_sec / sub_window_sec))
        total_needed = win + horizon
        if n_total < total_needed:
            continue

        for start in range(0, n_total - total_needed + 1, stride):
            in_win = sig[start: start + win]
            # discard windows with too many NaNs in the input itself
            if np.isnan(in_win).mean() > 0.1:
                continue

            future = sig[start + win: start + win + horizon]
            future_summary: List[float] = []
            for j in range(0, len(future) - sub_win + 1, sub_win):
                w = future[j: j + sub_win]
                if np.isnan(w).any():
                    continue
                v = float(np.mean(w)) if task.aggregator == 'mean' \
                    else float(np.max(w))
                if v < task.artifact_range[0] or v > task.artifact_range[1]:
                    continue
                future_summary.append(v)

            if len(future_summary) < max(1, min_consecutive // 2):
                continue

            label = int(_has_sustained(
                future_summary, task.threshold, task.direction, min_consecutive,
            ))
            extreme = (max(future_summary) if task.direction == 'above'
                       else min(future_summary))

            samples.append(ModalityForecastSample(
                input_signals={task.modal_name: np.nan_to_num(in_win)},
                label=label,
                label_value=float(extreme),
                case_id=case['case_id'],
                win_start_sec=start / sfreq,
                horizon_sec=horizon_sec,
            ))
    return samples


# ── PyTorch Dataset ─────────────────────────────────────────────


class ModalityForecastDataset(Dataset):
    """Yield ``({modal_name: tensor}, label)`` per sample."""

    def __init__(self, samples: List[ModalityForecastSample],
                 modal_order: Sequence[str]):
        self.samples = samples
        self.modal_order = list(modal_order)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        data: Dict[str, torch.Tensor] = {}
        for m in self.modal_order:
            if m in s.input_signals:
                data[m] = torch.from_numpy(s.input_signals[m]).float()
        label = torch.tensor(s.label, dtype=torch.long)
        return data, label


__all__ = [
    'ForecastTaskConfig', 'TASK_PRESETS',
    'ModalityForecastSample', 'ModalityForecastDataset',
    'extract_modality_samples', 'load_cases_modality',
]
