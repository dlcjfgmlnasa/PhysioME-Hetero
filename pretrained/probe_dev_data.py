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

The probe task is IOH (intra-operative hypotension; sustained MAP < 65 mmHg
within a 5-min lookahead). Single task is enough as a learning-progress
signal and keeps probe overhead low.

Public API:
    load_dev_probe_split(...) -> (train_samples, eval_samples)
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Set, Tuple

from downstream.tasks.hypotension import (
    ForecastSample,
    extract_forecast_samples,
    load_cases,
)


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

    all_cases = load_cases(
        data_dir=downstream_dir,
        input_signals=list(input_signals),
        min_duration_sec=min_duration_sec,
    )
    dev_cases = [c for c in all_cases if str(c['case_id']) in dev_ids]
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


__all__ = ['load_dev_probe_split']
