# -*- coding:utf-8 -*-
"""Full-signal VitalDB parser for downstream tasks.

Unlike ``vital_db_ssl.py`` (which writes fixed-window segments for SSL
pretraining), this parser keeps the full preprocessed waveform per case so that
downstream tasks (IOH, AKI, mortality, ...) can compute per-task labels on the
fly with custom window / horizon settings.

Output (per case, one ``.npz`` in ``trg_path``):
    abp: float32 [N]        — preprocessed ABP at sfreq Hz, **0.0 at artifact** samples
    abp_mask: bool [N]      — True where the corresponding ABP sample was rejected
                              (out-of-range / spike) and zero-filled
    ecg / ecg_mask, ppg / ppg_mask, cvp / cvp_mask, co2 / co2_mask,
    awp / awp_mask: same convention for the other modalities  (Step 3, 2026-05-09;
                              mask channel added 2026-05-13 — npz no longer carries NaN)
    spo2: float32 [N]       — pulse oximeter SpO2 (%) from Solar8000 monitor, label
                              channel only (NOT in MODAL_ORDER); piecewise-constant
                              resampling at sfreq Hz, 0.0 at out-of-range samples
    spo2_mask: bool [N]     — True where the SpO2 reading was outside [50, 100] and
                              zero-filled
    modality_present: bool [M]   — which of MODAL_ORDER exist in the recording
    label_present: bool [L]      — which of LABEL_ORDER (currently just SPO2) exist
    sfreq: int
    case_id: str

Modalities not recorded in the source case are not present in the npz at all
(use ``modality_present`` / ``label_present`` to check before indexing). When a
modality is present, both ``<modal>`` and ``<modal>_mask`` are present and have
the same length.

Modality keys / order are shared with ``vital_db_ssl.py`` via ``MODAL_ORDER``
/ ``MODAL_TO_SIGNAL_KEY`` — bumping the v2 set there propagates here for free.
Label channels are intentionally separate so the model never sees them as inputs.
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


# Label-only channels (NOT model inputs). Saved to npz so downstream tasks can
# compute labels without re-reading the raw vital file.
LABEL_TRACK_NAMES: Dict[str, str] = {
    # Bedside monitor numeric SpO2 (%), ~1 Hz native; resampled piecewise-
    # constant to sfreq Hz by vitaldb.vital_recs.
    'SPO2': 'Solar8000/PLETH_SPO2',
}
LABEL_ORDER = ['SPO2']


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

        present_labels = [lbl for lbl, trk in LABEL_TRACK_NAMES.items()
                          if trk in present_tracks]

        try:
            tracks = ([MODAL_TRACK_NAMES[m] for m in present_modals]
                      + [LABEL_TRACK_NAMES[lbl] for lbl in present_labels])
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
                channel, artifact_mask = preprocess_channel(
                    channel, signal_key=MODAL_TO_SIGNAL_KEY[m], sr=float(sfreq),
                    cfg=SIGNAL_CONFIGS[MODAL_TO_SIGNAL_KEY[m]],
                    return_mask=True,
                )
            else:
                # No preprocessing requested — still produce a mask channel so
                # the npz schema stays uniform. Mark non-finite samples and
                # zero them out so the payload itself contains no NaN.
                artifact_mask = ~np.isfinite(channel)
                channel = np.where(artifact_mask, 0.0, channel).astype(np.float32)
            save_dict[m.lower()] = channel
            save_dict[m.lower() + '_mask'] = artifact_mask

        # Label channels: no signal-domain preprocessing (these are monitor-
        # derived numerics like SpO2 at 1 Hz; bandpass/notch would destroy
        # them). Just clip to physiological range and emit a mask channel
        # for the out-of-range samples instead of writing NaN into the payload.
        for j, lbl in enumerate(present_labels):
            channel = np.asarray(data[:, len(present_modals) + j],
                                 dtype=np.float32)
            if lbl == 'SPO2':
                bad = (channel < 50.0) | (channel > 100.0) | ~np.isfinite(channel)
            else:
                bad = ~np.isfinite(channel)
            channel = np.where(bad, 0.0, channel).astype(np.float32)
            save_dict[lbl.lower()] = channel
            save_dict[lbl.lower() + '_mask'] = bad.astype(bool)

        modality_present = np.array(
            [m in present_modals for m in MODAL_ORDER], dtype=bool,
        )
        label_present = np.array(
            [lbl in present_labels for lbl in LABEL_ORDER], dtype=bool,
        )
        np.savez(
            os.path.join(trg_path, case_id + '.npz'),
            modality_present=modality_present,
            label_present=label_present,
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
