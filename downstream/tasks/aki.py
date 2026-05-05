# -*- coding:utf-8 -*-
"""Postoperative AKI (Acute Kidney Injury) downstream task.

Adapted from references/Biosignal-Foundation-Model/downstream/outcome/aki/
prepare_data.py (2026-05-04). KDIGO Cr-based labeling matches VitalDB official
mbp_aki / xgb_aki examples:

    Stage 1: peak postop Cr ≥ 1.5× preop Cr  OR  Δ ≥ 0.3 mg/dL
    Stage 2: peak postop Cr ≥ 2.0× preop Cr
    Stage 3: peak postop Cr ≥ 3.0× preop Cr  OR  peak ≥ 4.0 mg/dL
    Binary AKI = stage ≥ 1
    Postop window: opend < dt ≤ opend + 7 days

External data required (NOT in this repo):
    clinical_data.csv  — VitalDB official: caseid, preop_cr, opend
    lab_data.csv       — VitalDB official: caseid, dt (s), name, result
                         (creatinine rows are name == 'cr')

Both files are downloadable from https://vitaldb.net (login required).

Public API:
    AKICaseLabel
    load_aki_labels(clinical_csv, lab_csv) -> Dict[case_id, AKICaseLabel]
    AKIDataset                    — windowed dataset over our full-signal npz
                                    + the loaded label dict.

Wave-window logic mirrors HypotensionDataset: slide ``window_sec`` windows from
the *intra-operative* portion of each case (start to ``opend``) so that all
predictors are observed *before* the AKI event window opens.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ── KDIGO label dataclass ────────────────────────────────────────


@dataclass
class AKICaseLabel:
    case_id: int
    preop_cr: float
    peak_postop_cr: float
    abs_increase: float
    ratio: float
    stage: int           # 0/1/2/3
    binary: int          # 1 if stage ≥ 1 else 0


# ── CSV loaders (verbatim from upstream) ─────────────────────────


def _load_preop_and_opend(clinical_csv: str
                          ) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    with open(clinical_csv, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'caseid', 'preop_cr', 'opend'}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f'clinical CSV must have {required}, found {reader.fieldnames}')
        for row in reader:
            try:
                caseid = int(row['caseid'])
                preop_cr = float(row['preop_cr'])
                opend = float(row['opend'])
            except (ValueError, TypeError):
                continue
            if preop_cr <= 0 or preop_cr > 20 or opend <= 0:
                continue
            out[caseid] = (preop_cr, opend)
    return out


def _load_postop_peak_cr(lab_csv: str,
                         case_to_opend: Dict[int, float],
                         max_postop_days: float = 7.0
                         ) -> Dict[int, float]:
    """For each case, peak Cr within (opend, opend + max_postop_days * 86400]."""
    peaks: Dict[int, float] = {}
    horizon = max_postop_days * 86400.0
    with open(lab_csv, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'caseid', 'dt', 'name', 'result'}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f'lab CSV must have {required}, found {reader.fieldnames}')
        for row in reader:
            try:
                caseid = int(row['caseid'])
            except (ValueError, TypeError):
                continue
            if caseid not in case_to_opend:
                continue
            if str(row['name']).strip().lower() != 'cr':
                continue
            try:
                dt = float(row['dt'])
                cr = float(row['result'])
            except (ValueError, TypeError):
                continue
            opend = case_to_opend[caseid]
            if dt <= opend or dt > opend + horizon:
                continue
            if cr <= 0 or cr > 30:
                continue
            if cr > peaks.get(caseid, 0.0):
                peaks[caseid] = cr
    return peaks


def _kdigo_stage(preop_cr: float, peak_cr: float) -> int:
    if peak_cr <= 0:
        return 0
    delta = peak_cr - preop_cr
    ratio = peak_cr / preop_cr if preop_cr > 0 else 0.0
    if ratio >= 3.0 or peak_cr >= 4.0:
        return 3
    if ratio >= 2.0:
        return 2
    if ratio >= 1.5 or delta >= 0.3:
        return 1
    return 0


def load_aki_labels(clinical_csv: str, lab_csv: str,
                    max_postop_days: float = 7.0) -> Dict[int, AKICaseLabel]:
    """Build per-case AKI labels from VitalDB CSVs."""
    preop = _load_preop_and_opend(clinical_csv)
    case_to_opend = {cid: op for cid, (_, op) in preop.items()}
    peaks = _load_postop_peak_cr(lab_csv, case_to_opend, max_postop_days)
    out: Dict[int, AKICaseLabel] = {}
    for cid, (preop_cr, _opend) in preop.items():
        peak = peaks.get(cid, 0.0)
        stage = _kdigo_stage(preop_cr, peak)
        out[cid] = AKICaseLabel(
            case_id=cid, preop_cr=preop_cr, peak_postop_cr=peak,
            abs_increase=peak - preop_cr,
            ratio=(peak / preop_cr) if preop_cr > 0 else 0.0,
            stage=stage, binary=int(stage >= 1),
        )
    return out


# ── Wave window extraction ───────────────────────────────────────


def _vitaldb_case_id_from_name(case_id_str: str) -> Optional[int]:
    """Convert filename-style case id ('0042') to VitalDB integer caseid."""
    digits = ''.join(c for c in case_id_str if c.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


@dataclass
class AKISample:
    input_signals: Dict[str, np.ndarray]
    label: int                      # binary AKI
    stage: int
    case_id: int
    win_start_sec: float


def extract_aki_samples(
    cases: Sequence[Dict],
    aki_labels: Dict[int, AKICaseLabel],
    input_signals: Sequence[str],
    window_sec: float = 600.0,
    stride_sec: float = 300.0,
    use_binary: bool = True,
) -> List[AKISample]:
    """For each case with a known AKI label, slide windows over the available
    intra-operative signal and emit (input, label) pairs.

    All windows of one case share the same case-level AKI label — this is the
    standard "outcome from waveform" framing used by the upstream code.
    """
    samples: List[AKISample] = []
    for case in cases:
        cid = _vitaldb_case_id_from_name(case['case_id'])
        if cid is None or cid not in aki_labels:
            continue
        lbl = aki_labels[cid]
        signals = case['signals']
        sfreq = case['sfreq']
        # Use the longest aligned signal length.
        n = min(s.shape[0] for s in signals.values())
        win = int(window_sec * sfreq)
        stride = int(stride_sec * sfreq)
        if n < win:
            continue
        for start in range(0, n - win + 1, stride):
            win_signals: Dict[str, np.ndarray] = {}
            for stype in input_signals:
                key = stype.lower()
                if key in signals:
                    sig = signals[key][start: start + win]
                    if np.isnan(sig).mean() > 0.1:
                        win_signals = {}
                        break
                    win_signals[key] = sig
            if not win_signals:
                continue
            label = lbl.binary if use_binary else lbl.stage
            samples.append(AKISample(
                input_signals=win_signals, label=label, stage=lbl.stage,
                case_id=cid, win_start_sec=start / sfreq,
            ))
    return samples


# ── Dataset ─────────────────────────────────────────────────────


class AKIDataset(Dataset):
    def __init__(self, samples: List[AKISample],
                 modal_order: Sequence[str] = ('ABP', 'ECG', 'PPG', 'CO2')):
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
    'AKICaseLabel', 'AKISample', 'AKIDataset',
    'load_aki_labels', 'extract_aki_samples',
]
