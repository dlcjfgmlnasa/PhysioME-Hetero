# -*- coding:utf-8 -*-
"""Shared dev-cohort probing data loader for SSL training scripts.

Both Phase-1 (DP-NeuroNet, unimodal) and Phase-2 (PhysioME-Hetero, multimodal)
training loops monitor learning progress with periodic linear-probing on a
held-out *dev cohort* — the case_ids listed in ``dev_subjects_file``
(produced by ``sample_holdout.py --n_dev``). Using the same cohort across
phases:

  * keeps the dev cohort fully disjoint from the downstream test cohort
    (= holdout), so probing never leaks into reported test metrics;
  * makes Phase-1 vs Phase-2 probe curves directly comparable on the same
    subjects.

Two task families are supported:

  ``load_dev_probe_split``         — IOH forecast (sustained MAP<65 mmHg in
                                    a 5-min lookahead). Useful for probes
                                    whose latents are derived from
                                    ABP/ECG/PPG (the modalities directly
                                    related to the MAP-based label).
  ``load_dev_probe_modality_split`` — self-modality forecast (CVP/CO2/AWP
                                    each with their own clinical threshold-
                                    crossing event). Used by Phase-1 to
                                    monitor backbones for modalities that
                                    have no IOH-equivalent label channel.

Public API:
    load_dev_probe_split(...)            -> IOH samples
    load_dev_probe_modality_split(...)   -> self-modality forecast samples
    PHASE1_PROBE_TASK_FOR_MODAL          -> per-modality dispatch table
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import tqdm

from downstream.tasks.hypotension import (
    ForecastSample,
    _load_case_npz as _load_case_npz_iohlike,
    extract_forecast_samples,
)
from downstream.tasks.modality_forecast import (
    ForecastTaskConfig,
    ModalityForecastSample,
    TASK_PRESETS,
    _load_case_npz as _load_case_npz_modality,
    extract_modality_samples,
)


def _read_dev_cases_iohlike(downstream_dir: str, dev_ids: Set[str],
                            min_duration_sec: float,
                            input_signals: Sequence[str]) -> List[dict]:
    """Direct npz read for the dev cohort, IOH-style (ABP+ECG+PPG channels).

    Bypasses ``hypotension.load_cases``'s ``glob.glob`` over the whole
    downstream_dir so that probe initialisation does not list every file
    on the (often slow) NFS share. Reads only ``<dev_id>.npz`` for the
    50-ish dev subjects.
    """
    required = set(s.lower() for s in input_signals) | {'abp'}
    cases: List[dict] = []
    missing = 0
    ids_sorted = sorted(dev_ids)
    for cid in tqdm.tqdm(ids_sorted, desc='probe dev_iohlike',
                         total=len(ids_sorted), mininterval=1.0):
        p = os.path.join(downstream_dir, f'{cid}.npz')
        if not os.path.isfile(p):
            missing += 1
            continue
        loaded = _load_case_npz_iohlike(p)
        if loaded is None:
            continue
        signals = loaded['signals']
        if not required.issubset(signals.keys()):
            continue
        min_len = min(s.shape[0] for s in signals.values())
        sfreq = loaded['sfreq']
        if min_len < int(min_duration_sec * sfreq):
            continue
        signals = {k: v[:min_len] for k, v in signals.items()}
        cases.append({'case_id': loaded['case_id'], 'signals': signals,
                      'sfreq': sfreq})
    if missing:
        print(f'  [probe_dev_data] {missing}/{len(dev_ids)} dev npz '
              f'not found on disk — proceeding with {len(cases)} cases')
    return cases


def _read_dev_cases_modality(downstream_dir: str, dev_ids: Set[str],
                             task: ForecastTaskConfig,
                             min_duration_sec: float) -> List[dict]:
    """Direct npz read for the dev cohort, single-modality forecast tasks."""
    cases: List[dict] = []
    missing = 0
    ids_sorted = sorted(dev_ids)
    for cid in tqdm.tqdm(ids_sorted, desc=f'probe dev_{task.signal_key}',
                         total=len(ids_sorted), mininterval=1.0):
        p = os.path.join(downstream_dir, f'{cid}.npz')
        if not os.path.isfile(p):
            missing += 1
            continue
        loaded = _load_case_npz_modality(p, task.signal_key)
        if loaded is None:
            continue
        sig = loaded['signals'][task.signal_key]
        if sig.shape[0] < int(min_duration_sec * loaded['sfreq']):
            continue
        cases.append({'case_id': loaded['case_id'],
                      'signals': {task.signal_key: sig},
                      'sfreq': loaded['sfreq']})
    if missing:
        print(f'  [probe_dev_data] {missing}/{len(dev_ids)} dev npz '
              f'not found on disk for task {task.modal_name} — '
              f'proceeding with {len(cases)} cases')
    return cases


# Per-Phase-1 modality, the probe task that is *both* (a) computable from the
# modality channel alone, and (b) clinically meaningful. ABP/ECG/PPG re-use
# the IOH probe (MAP-based, ABP-derived label — but ECG/PPG-only encoders
# can still be probed for that label as a "is this latent informative for
# IOH?" sanity check). CVP/CO2/AWP get their own self-forecast task.
PHASE1_PROBE_TASK_FOR_MODAL: Dict[str, str] = {
    'ABP': 'ioh',
    'ECG': 'ioh',
    'PPG': 'ioh',
    'CVP': 'cvp',
    'CO2': 'co2',
    'AWP': 'awp',
}


def _load_subject_ids(path: str) -> Set[str]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return {str(c) for c in payload['case_ids']}


def load_dev_probe_split(
    downstream_dir: str,
    dev_subjects_file: str,
    *,
    input_signals: Sequence[str] = ('ABP', 'ECG', 'PPG'),
    window_sec: float = 60.0,
    stride_sec: float = 60.0,
    horizon_sec: float = 300.0,
    map_threshold: float = 65.0,
    sustained_sec: float = 60.0,
    train_fraction: float = 0.7,
    min_duration_sec: float = 600.0,
) -> Tuple[List[ForecastSample], List[ForecastSample]]:
    """Build (probe_train, probe_eval) IOH samples from the dev cohort.

    The dev cohort is itself sorted by case_id and split deterministically
    into train_fraction / 1 - train_fraction halves, so the probe metric is
    reproducible across runs and phases.
    """
    if not os.path.isdir(downstream_dir):
        raise FileNotFoundError(
            f'downstream_dir not found: {downstream_dir!r}')
    if not os.path.isfile(dev_subjects_file):
        raise FileNotFoundError(
            f'dev_subjects_file not found: {dev_subjects_file!r}')

    dev_ids = _load_subject_ids(dev_subjects_file)

    dev_cases = _read_dev_cases_iohlike(
        downstream_dir=downstream_dir,
        dev_ids=dev_ids,
        min_duration_sec=min_duration_sec,
        input_signals=input_signals,
    )
    if not dev_cases:
        raise RuntimeError(
            f'No dev cases found in {downstream_dir!r} matching '
            f'{dev_subjects_file!r}. Check that downstream npz cohort '
            'overlaps the dev cohort.')

    dev_cases.sort(key=lambda c: str(c['case_id']))
    n_train = max(1, int(len(dev_cases) * train_fraction))
    train_cases = dev_cases[:n_train]
    eval_cases = dev_cases[n_train:]
    if not eval_cases:
        # tiny dev cohort — fall back to last case as eval to keep shape valid
        eval_cases = dev_cases[-1:]
        train_cases = dev_cases[:-1]

    kw = dict(
        input_signals=list(input_signals),
        window_sec=window_sec,
        stride_sec=stride_sec,
        horizon_sec=horizon_sec,
        map_threshold=map_threshold,
        sustained_sec=sustained_sec,
    )
    train_samples = extract_forecast_samples(train_cases, **kw)
    eval_samples = extract_forecast_samples(eval_cases, **kw)
    return train_samples, eval_samples


def load_dev_probe_modality_split(
    downstream_dir: str,
    dev_subjects_file: str,
    *,
    task_key: str,
    window_sec: float = 60.0,
    stride_sec: float = 60.0,
    horizon_sec: float = 300.0,
    train_fraction: float = 0.7,
    min_duration_sec: float = 600.0,
) -> Tuple[List[ModalityForecastSample], List[ModalityForecastSample]]:
    """Build (probe_train, probe_eval) self-modality forecast samples.

    ``task_key`` selects from ``downstream.tasks.modality_forecast.TASK_PRESETS``
    ('cvp' | 'co2' | 'awp'). Returns ModalityForecastSample lists; the
    Phase-1 trainer wraps these with ``ModalityForecastDataset``.
    """
    if task_key not in TASK_PRESETS:
        raise KeyError(f'unknown probe task: {task_key!r}; '
                       f'options = {sorted(TASK_PRESETS)}')
    task = TASK_PRESETS[task_key]

    if not os.path.isdir(downstream_dir):
        raise FileNotFoundError(
            f'downstream_dir not found: {downstream_dir!r}')
    if not os.path.isfile(dev_subjects_file):
        raise FileNotFoundError(
            f'dev_subjects_file not found: {dev_subjects_file!r}')

    dev_ids = _load_subject_ids(dev_subjects_file)

    dev_cases = _read_dev_cases_modality(
        downstream_dir=downstream_dir,
        dev_ids=dev_ids,
        task=task,
        min_duration_sec=min_duration_sec,
    )
    if not dev_cases:
        raise RuntimeError(
            f'No dev cases found in {downstream_dir!r} matching '
            f'{dev_subjects_file!r} for task {task_key!r}. The dev cohort '
            f'may not contain subjects with the {task.modal_name} channel.'
        )

    dev_cases.sort(key=lambda c: str(c['case_id']))
    n_train = max(1, int(len(dev_cases) * train_fraction))
    train_cases = dev_cases[:n_train]
    eval_cases = dev_cases[n_train:] or dev_cases[-1:]
    if eval_cases is dev_cases[-1:]:
        train_cases = dev_cases[:-1]

    kw = dict(
        task=task,
        window_sec=window_sec,
        stride_sec=stride_sec,
        horizon_sec=horizon_sec,
    )
    train_samples = extract_modality_samples(train_cases, **kw)
    eval_samples = extract_modality_samples(eval_cases, **kw)
    return train_samples, eval_samples


__all__ = [
    'load_dev_probe_split',
    'load_dev_probe_modality_split',
    'PHASE1_PROBE_TASK_FOR_MODAL',
]
