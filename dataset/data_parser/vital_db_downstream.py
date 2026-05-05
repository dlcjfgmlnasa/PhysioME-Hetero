# -*- coding:utf-8 -*-
"""Full-signal VitalDB parser for downstream tasks.

Unlike ``vital_db_ssl.py`` (which writes fixed-window segments for SSL
pretraining), this parser keeps the full preprocessed waveform per case so that
downstream tasks (IOH, AKI, mortality, ...) can compute per-task labels on the
fly with custom window / horizon settings.

Output (per case, one ``.npz`` in ``trg_path``):
    abp: float32 [N]   — preprocessed ABP at sfreq Hz, NaN where invalid
    ecg: float32 [N]   — preprocessed ECG, ditto
    ppg: float32 [N]   — preprocessed PPG, ditto
    co2: float32 [N]   — preprocessed CO2 (capnography / etCO2), ditto
    modality_present: bool [M]   — which of MODAL_ORDER exist in the recording
    sfreq: int
    case_id: str

Modalities not recorded in the source case are not present in the npz at all
(use ``modality_present`` to check before indexing). Modality keys / order are
shared with ``vital_db_ssl.py`` via ``MODAL_ORDER`` / ``MODAL_TO_SIGNAL_KEY``.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import tqdm

from dataset.data_parser._signal_filters import (
    SIGNAL_CONFIGS,
    preprocess_channel,
)
from dataset.data_parser.vital_db_ssl import (
    MODAL_ORDER,
    MODAL_TO_SIGNAL_KEY,
    MODAL_TRACK_NAMES,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path',
                        default=os.path.join('..', '..', '..', '..', 'Dataset', 'vitaldb'))
    parser.add_argument('--trg_path',
                        default=os.path.join('..', '..', 'data', 'vitaldb_downstream'))
    parser.add_argument('--sfreq', type=int, default=100)
    parser.add_argument('--require_abp', action='store_true',
                        help='only save cases that have ABP (needed for MAP-based labels)')
    parser.add_argument('--skip_preprocess', action='store_true')
    return parser.parse_args()


def vitaldb_downstream_converter(src_path: str, trg_path: str,
                                 sfreq: int = 100,
                                 require_abp: bool = False,
                                 skip_preprocess: bool = False) -> None:
    import vitaldb
    os.makedirs(trg_path, exist_ok=True)
    paths = sorted(os.listdir(src_path))

    saved, skipped = 0, 0
    for fname in tqdm.tqdm(paths, desc='VitalDB-Downstream'):
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
        if require_abp and 'ABP' not in present_modals:
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

        save_dict: Dict[str, np.ndarray] = {}
        for i, m in enumerate(present_modals):
            channel = np.asarray(data[:, i], dtype=np.float32)
            if not skip_preprocess:
                channel = preprocess_channel(
                    channel, signal_key=MODAL_TO_SIGNAL_KEY[m], sr=float(sfreq),
                    cfg=SIGNAL_CONFIGS[MODAL_TO_SIGNAL_KEY[m]],
                )
            save_dict[m.lower()] = channel

        modality_present = np.array(
            [m in present_modals for m in MODAL_ORDER], dtype=bool,
        )
        np.savez(
            os.path.join(trg_path, case_id + '.npz'),
            modality_present=modality_present,
            sfreq=int(sfreq),
            case_id=case_id,
            **save_dict,
        )
        saved += 1

    print(f'[VitalDB-Downstream] saved={saved}, skipped={skipped}')


if __name__ == '__main__':
    args = get_args()
    vitaldb_downstream_converter(
        src_path=args.src_path,
        trg_path=args.trg_path,
        sfreq=args.sfreq,
        require_abp=args.require_abp,
        skip_preprocess=args.skip_preprocess,
    )
