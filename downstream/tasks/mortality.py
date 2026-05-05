# -*- coding:utf-8 -*-
"""ICU mortality downstream task (MIMIC-III Waveform-based).

Adapted from references/Biosignal-Foundation-Model/downstream/outcome/mortality/
prepare_data.py (2026-05-04). Label = ``hospital_expire_flag`` from an ICU
cohort CSV, joined to MIMIC-III Waveform records by ``subject_id``.

Cross-dataset generalization story (matches the upstream paper's framing):
    VitalDB pretraining (Korean OR) → MIMIC-III mortality prediction (US ICU).

External data required (NOT in this repo):
    cohort_csv — columns at minimum: subject_id (int), icustay_id (str),
                 hospital_expire_flag (0/1). May also include: first_careunit,
                 age, gender, icu_intime, icu_outtime.
                 References include the upstream's
                 ``downstream/outcome/mortality/icu_mortality_cohort.csv``.

Wave data: per-record npz produced by ``dataset/data_parser/mimic3_waveform_ssl.py``
(parse → hetero npz with ``x[T,3,sfreq*duration]`` windows).

Public API:
    MortalityCaseLabel
    load_mortality_cohort(cohort_csv) -> Dict[subject_id, MortalityCaseLabel]
    MortalityDataset — windowed dataset over hetero MIMIC-III npz + label dict.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Cohort loader ────────────────────────────────────────────────


@dataclass
class MortalityCaseLabel:
    subject_id: int
    icustay_id: str
    mortality: int                 # 0 or 1
    first_careunit: str = ''
    age: str = ''
    gender: str = ''


def load_mortality_cohort(cohort_csv: str
                          ) -> Dict[int, MortalityCaseLabel]:
    out: Dict[int, MortalityCaseLabel] = {}
    with open(cohort_csv, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if 'subject_id' not in (reader.fieldnames or []):
            raise ValueError(
                f'cohort CSV must contain subject_id; got {reader.fieldnames}')
        for row in reader:
            try:
                sid = int(row['subject_id'])
            except (ValueError, TypeError):
                continue
            mortality = int(row.get('hospital_expire_flag') or 0)
            out[sid] = MortalityCaseLabel(
                subject_id=sid,
                icustay_id=row.get('icustay_id', ''),
                mortality=mortality,
                first_careunit=row.get('first_careunit', ''),
                age=row.get('age', ''),
                gender=row.get('gender', ''),
            )
    return out


# ── Subject-id parsing ──────────────────────────────────────────


_SUBJ_RE = re.compile(r'p0*(\d+)')


def _subject_id_from_record(record_name: str) -> Optional[int]:
    """MIMIC-III record names look like 'p000020-...'; extract integer
    subject id (20)."""
    m = _SUBJ_RE.match(record_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ── Sample extraction ───────────────────────────────────────────


@dataclass
class MortalitySample:
    input_signals: Dict[str, np.ndarray]   # keys are 'abp' / 'ecg' / 'ppg'
    label: int                              # 0/1
    subject_id: int
    record_name: str
    win_index: int


def extract_mortality_samples_from_npz(
    mimic_npz_dir: str,
    cohort: Dict[int, MortalityCaseLabel],
    input_signals: Sequence[str] = ('ABP', 'ECG', 'PPG'),
    max_windows_per_record: Optional[int] = None,
) -> List[MortalitySample]:
    """Iterate hetero MIMIC-III npz files and emit (window, mortality) pairs
    keyed by subject_id from ``cohort``.

    ``mimic_npz_dir`` should be the output of
    ``dataset/data_parser/mimic3_waveform_ssl.py parse``.
    """
    paths = sorted(glob.glob(os.path.join(mimic_npz_dir, '*.npz')))
    out: List[MortalitySample] = []
    modal_idx = {'ABP': 0, 'ECG': 1, 'PPG': 2}
    requested = [m for m in input_signals if m in modal_idx]

    for p in paths:
        record_name = os.path.splitext(os.path.basename(p))[0]
        sid = _subject_id_from_record(record_name)
        if sid is None or sid not in cohort:
            continue
        label = cohort[sid].mortality

        with np.load(p, allow_pickle=True) as arr:
            xs = np.asarray(arr['x'], dtype=np.float32)        # [T, 3, S]
            masks = np.asarray(arr['mask'], dtype=bool)         # [T, 3]

        for t in range(xs.shape[0]):
            if max_windows_per_record is not None and t >= max_windows_per_record:
                break
            row_mask = masks[t]
            window_signals: Dict[str, np.ndarray] = {}
            ok = True
            for m in requested:
                idx = modal_idx[m]
                if not row_mask[idx]:
                    ok = False
                    break
                window_signals[m.lower()] = xs[t, idx]
            if not ok:
                continue
            out.append(MortalitySample(
                input_signals=window_signals, label=label,
                subject_id=sid, record_name=record_name, win_index=t,
            ))
    return out


# ── Dataset ─────────────────────────────────────────────────────


class MortalityDataset(Dataset):
    def __init__(self, samples: List[MortalitySample],
                 modal_order: Sequence[str] = ('ABP', 'ECG', 'PPG')):
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
        return data, torch.tensor(s.label, dtype=torch.long)


__all__ = [
    'MortalityCaseLabel', 'MortalitySample', 'MortalityDataset',
    'load_mortality_cohort', 'extract_mortality_samples_from_npz',
]
