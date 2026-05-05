# -*- coding:utf-8 -*-
"""Intra-operative hypotension (IOH) downstream task.

Adapted from references/Biosignal-Foundation-Model/downstream/acute_event/
hypotension/prepare_data.py (2026-05-04). The label generation logic — sustained
MAP < 65 mmHg for ≥1 minute within a future horizon — is preserved verbatim;
the data-source side is rewritten to read our full-signal VitalDB downstream
npz files (output of ``dataset/data_parser/vital_db_downstream.py``).

Public API:
    ForecastSample        — one (input window, future-MAP label) sample.
    extract_forecast_samples(...) — sweep window/horizon over a list of cases.
    HypotensionDataset    — torch.utils.data.Dataset over ForecastSample list.
    sweep_window_horizon(...) — convenience wrapper for grid evaluation.

Default config:
    window_sec    = 60 s    (matches NeuroNet backbone pretraining window)
    stride_sec    = 60 s    (sliding window stride)
    horizon_sec   = 300 s   (5 min lookahead)
    map_threshold = 65 mmHg
    sustained_sec = 60 s    (≥1 min sustained MAP < threshold)
    map_win_sec   = 10 s    (sub-window for averaging future MAP)
    artifact_map_range = (30, 200)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Sample ───────────────────────────────────────────────────────


@dataclass
class ForecastSample:
    input_signals: Dict[str, np.ndarray]   # {'abp': (T,), ...}
    label: int                             # 0=normal, 1=sustained hypotension
    label_value: float                     # min future MAP (mmHg) — informational
    case_id: str
    win_start_sec: float
    horizon_sec: float


# ── Label logic (verbatim from upstream, mild rename) ────────────


def _has_sustained_hypotension(future_maps: Sequence[float],
                               threshold: float,
                               min_consecutive: int) -> bool:
    consecutive = 0
    for m in future_maps:
        if m < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return True
        else:
            consecutive = 0
    return False


# ── Case loading ─────────────────────────────────────────────────


def _load_case_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load one case-level npz produced by ``vital_db_downstream.py``."""
    with np.load(path, allow_pickle=True) as arr:
        modality_present = np.asarray(arr['modality_present'], dtype=bool)
        # ABP is required for MAP-based labels.
        if not modality_present[0]:
            return None
        signals: Dict[str, np.ndarray] = {}
        for m in ('abp', 'ecg', 'ppg'):
            if m in arr.files:
                signals[m] = np.asarray(arr[m], dtype=np.float32)
        sfreq = int(arr['sfreq'])
        case_id = str(arr['case_id'])
    if 'abp' not in signals:
        return None
    return {
        'signals': signals,
        'sfreq': sfreq,
        'case_id': case_id,
    }


def load_cases(data_dir: str,
               input_signals: Sequence[str],
               min_duration_sec: float = 1200.0,
               max_subjects: Optional[int] = None) -> List[Dict]:
    """Load preprocessed cases that contain every modality in ``input_signals``
    plus ABP (needed for the label).
    """
    required = set(s.lower() for s in input_signals) | {'abp'}
    paths = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if max_subjects is not None:
        paths = paths[:max_subjects]
    cases: List[Dict] = []
    for p in paths:
        loaded = _load_case_npz(p)
        if loaded is None:
            continue
        signals = loaded['signals']
        if not required.issubset(signals.keys()):
            continue
        # Trim to common length so per-modal alignment is preserved.
        min_len = min(s.shape[0] for s in signals.values())
        sfreq = loaded['sfreq']
        if min_len < int(min_duration_sec * sfreq):
            continue
        signals = {k: v[:min_len] for k, v in signals.items()}
        cases.append({'case_id': loaded['case_id'], 'signals': signals,
                      'sfreq': sfreq})
    return cases


# ── Window / horizon extraction ─────────────────────────────────


def extract_forecast_samples(
    cases: Sequence[Dict],
    input_signals: Sequence[str],
    window_sec: float = 60.0,
    stride_sec: float = 60.0,
    horizon_sec: float = 300.0,
    map_threshold: float = 65.0,
    sustained_sec: float = 60.0,
    map_win_sec: float = 10.0,
    artifact_map_range: Tuple[float, float] = (30.0, 200.0),
) -> List[ForecastSample]:
    """Slide ``window_sec`` input windows with ``stride_sec`` stride and label
    each by whether MAP < ``map_threshold`` is sustained for ``sustained_sec``
    within the next ``horizon_sec`` seconds.

    Per-window MAP is averaged inside ``map_win_sec`` (default 10 s).
    Sub-windows whose mean MAP falls outside ``artifact_map_range`` are
    discarded.
    """
    samples: List[ForecastSample] = []
    for case in cases:
        signals = case['signals']
        abp = signals['abp']
        sfreq = case['sfreq']
        n_total = abp.shape[0]

        win = int(window_sec * sfreq)
        stride = int(stride_sec * sfreq)
        horizon = int(horizon_sec * sfreq)
        map_win = int(map_win_sec * sfreq)
        min_consecutive = max(1, int(sustained_sec / map_win_sec))
        total_needed = win + horizon

        if n_total < total_needed:
            continue

        for start in range(0, n_total - total_needed + 1, stride):
            input_dict: Dict[str, np.ndarray] = {}
            for stype in input_signals:
                key = stype.lower()
                if key in signals:
                    input_dict[key] = signals[key][start: start + win]
            if not input_dict:
                continue

            # Discard windows that contain too many NaNs in the input itself.
            if any(np.isnan(v).mean() > 0.1 for v in input_dict.values()):
                continue

            future_start = start + win
            future_end = future_start + horizon
            future_abp = abp[future_start:future_end]

            future_maps: List[float] = []
            for j in range(0, len(future_abp) - map_win + 1, map_win):
                w = future_abp[j: j + map_win]
                if np.isnan(w).any():
                    continue
                m = float(np.mean(w))
                if m < artifact_map_range[0] or m > artifact_map_range[1]:
                    continue
                future_maps.append(m)

            if len(future_maps) < max(1, min_consecutive // 2):
                continue

            label = int(_has_sustained_hypotension(
                future_maps, map_threshold, min_consecutive))
            min_future_map = float(min(future_maps))

            samples.append(ForecastSample(
                input_signals=input_dict, label=label,
                label_value=min_future_map,
                case_id=case['case_id'],
                win_start_sec=start / sfreq,
                horizon_sec=horizon_sec,
            ))
    return samples


# ── PyTorch Dataset ──────────────────────────────────────────────


class HypotensionDataset(Dataset):
    """Wrap a list of :class:`ForecastSample` for PyTorch DataLoader use.

    Yields ``(data_dict, label)`` where ``data_dict`` keys are uppercase
    modality names ('ABP', 'ECG', 'PPG', 'CVP') matching ``PhysioME.modal_names``.
    """

    def __init__(self, samples: List[ForecastSample],
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


# ── Sweep convenience ────────────────────────────────────────────


def sweep_window_horizon(
    cases: Sequence[Dict],
    input_signals: Sequence[str],
    windows_sec: Sequence[float] = (30.0, 60.0, 180.0, 300.0, 600.0),
    horizons_sec: Sequence[float] = (300.0, 600.0, 900.0),
) -> Dict[Tuple[float, float], List[ForecastSample]]:
    """Convenience: build a dict ``{(window_sec, horizon_sec): samples}``
    for a window×horizon grid. Used by downstream/run.py to train one model
    per (window, horizon) combination.
    """
    out: Dict[Tuple[float, float], List[ForecastSample]] = {}
    for w in windows_sec:
        for h in horizons_sec:
            out[(w, h)] = extract_forecast_samples(
                cases, input_signals=input_signals,
                window_sec=w, stride_sec=w, horizon_sec=h,
            )
    return out


__all__ = [
    'ForecastSample',
    'HypotensionDataset',
    'load_cases',
    'extract_forecast_samples',
    'sweep_window_horizon',
]
