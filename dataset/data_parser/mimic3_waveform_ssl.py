# -*- coding:utf-8 -*-
"""MIMIC-III Waveform Matched Subset → hetero npz parser.

Adapted from references/Biosignal-Foundation-Model/data/parser/mimic3_waveform.py
(2026-05-04). Keeps the upstream's wfdb streaming, multi-segment handling,
and channel-name matching, but writes outputs in the hetero schema produced by
``vital_db_ssl.py`` so that downstream code (HeteroVitalDBDataset, BucketBatchSampler)
works without modification.

Output (per record, one ``.npz`` in ``trg_path``):
    x: float32 [T, 3, sfreq * duration]
    mask: bool [T, 3]
    modal_names: object [3]   — fixed ``['ABP', 'ECG', 'PPG']``
    subject_modality_set: bool [3]
    case_id: str — record name (e.g., 'p000020-2183-04-28-17-47')

Usage::

    # Scan ABP-bearing records and cache the manifest.
    python -m dataset.data_parser.mimic3_waveform_ssl scan \
        --max_records 200 --manifest data/mimic3_ssl/manifest.json

    # Parse N records into hetero npz.
    python -m dataset.data_parser.mimic3_waveform_ssl parse \
        --manifest data/mimic3_ssl/manifest.json \
        --trg_path data/mimic3_ssl --n_cases 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

from dataset.data_parser._signal_filters import (
    SIGNAL_CONFIGS,
    preprocess_channel,
    resample_to_target,
)
from dataset.data_parser.vital_db_ssl import (
    MODAL_ORDER,
    MODAL_TO_SIGNAL_KEY,
    extract_segments,
)


# ── MIMIC-III specifics ──────────────────────────────────────────

PN_DB = "mimic3wdb-matched/1.0"
MIMIC3_NATIVE_SR: float = 125.0

# Candidate channel names per modality (first match wins).
MIMIC_CHANNEL_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    'ABP': ('ABP', 'ART'),
    'ECG': ('II', 'I', 'III', 'V', 'aVR', 'aVL', 'aVF', 'MCL1'),
    'PPG': ('PLETH',),
}


@dataclass
class MimicRecordInfo:
    record_name: str
    pn_dir: str
    patient_id: str
    abp_channel: str = ''
    ecg_channel: str = ''
    ppg_channel: str = ''
    n_segments: int = 1


# ── Manifest scanning ────────────────────────────────────────────


def _resolve_channel(sig_names: List[str], modal: str) -> str:
    for cand in MIMIC_CHANNEL_CANDIDATES[modal]:
        if cand in sig_names:
            return cand
    return ''


def scan_records(max_records: int = 0,
                 require_abp: bool = False,
                 verbose: bool = True) -> List[MimicRecordInfo]:
    """Stream the MIMIC-III Matched Subset and collect records that contain
    at least one of {ABP, ECG, PPG}. Returns the list (also serialisable to
    json via ``save_manifest``).
    """
    import wfdb
    if verbose:
        print(f'Scanning MIMIC-III Matched Subset ({PN_DB})...')

    all_records = wfdb.get_record_list(PN_DB)
    patient_dirs: Dict[str, str] = {}
    for rec in all_records:
        parts = rec.split('/')
        if len(parts) >= 2:
            patient_dirs[parts[1]] = f'{PN_DB}/{parts[0]}/{parts[1]}'

    items = list(patient_dirs.items())
    if max_records > 0:
        items = items[:max_records]

    found: List[MimicRecordInfo] = []
    n_errors = 0
    for i, (pid, pn_dir) in enumerate(items):
        try:
            sub = wfdb.get_record_list(pn_dir)
        except Exception:
            n_errors += 1
            continue
        if not sub:
            continue
        wave_records = [r for r in sub if r.startswith(pid) and not r.endswith('n')]
        for rec_name in wave_records:
            try:
                hdr = wfdb.rdheader(rec_name, pn_dir=pn_dir)
                if hasattr(hdr, 'seg_name') and hdr.seg_name:
                    layout = hdr.seg_name[0]
                    if layout and not layout.startswith('~'):
                        layout_hdr = wfdb.rdheader(layout, pn_dir=pn_dir)
                        sig_names = layout_hdr.sig_name or []
                    else:
                        sig_names = []
                else:
                    sig_names = hdr.sig_name or []
                if not sig_names:
                    continue
                abp_ch = _resolve_channel(sig_names, 'ABP')
                ecg_ch = _resolve_channel(sig_names, 'ECG')
                ppg_ch = _resolve_channel(sig_names, 'PPG')
                if require_abp and not abp_ch:
                    continue
                if not (abp_ch or ecg_ch or ppg_ch):
                    continue
                found.append(MimicRecordInfo(
                    record_name=rec_name, pn_dir=pn_dir, patient_id=pid,
                    abp_channel=abp_ch, ecg_channel=ecg_ch, ppg_channel=ppg_ch,
                    n_segments=len(hdr.seg_name) if hasattr(hdr, 'seg_name') else 1,
                ))
            except Exception:
                n_errors += 1
                continue
        if verbose and (i + 1) % 50 == 0:
            print(f'  scanned {i + 1}/{len(items)} patients, '
                  f'found {len(found)} usable records ({n_errors} errors)')
    if verbose:
        print(f'Scan complete: {len(found)} records, {n_errors} errors')
    return found


def save_manifest(records: List[MimicRecordInfo], path: str) -> None:
    payload = [r.__dict__ for r in records]
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_manifest(path: str) -> List[MimicRecordInfo]:
    with open(path) as f:
        items = json.load(f)
    return [MimicRecordInfo(**d) for d in items]


# ── Per-record load + preprocess + window ────────────────────────


def _load_aligned_segment(seg_name: str, pn_dir: str,
                          channel_map: Dict[str, str]
                          ) -> Optional[Tuple[Dict[str, np.ndarray], float]]:
    """Read one aligned multi-channel segment.

    Returns (per-modal arrays at native SR, native_sr) or None on failure.
    The arrays are time-aligned (same length) because they come from a single
    wfdb segment.
    """
    import wfdb
    try:
        seg = wfdb.rdrecord(seg_name, pn_dir=pn_dir)
    except Exception:
        return None
    if seg.p_signal is None or seg.sig_name is None:
        return None
    out: Dict[str, np.ndarray] = {}
    for modal, ch_name in channel_map.items():
        if ch_name and ch_name in seg.sig_name:
            idx = seg.sig_name.index(ch_name)
            out[modal] = seg.p_signal[:, idx].astype(np.float32)
    if not out:
        return None
    native_sr = float(getattr(seg, 'fs', MIMIC3_NATIVE_SR))
    return out, native_sr


def process_record(info: MimicRecordInfo,
                   sfreq: int = 100, duration: int = 60,
                   nan_max_ratio: float = 0.1,
                   skip_preprocess: bool = False,
                   verbose: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Process one MIMIC record into hetero windows.

    Returns (xs[T,3,sfreq*duration], masks[T,3]) or None if no valid windows.
    """
    import wfdb

    channel_map = {
        'ABP': info.abp_channel,
        'ECG': info.ecg_channel,
        'PPG': info.ppg_channel,
    }

    try:
        hdr = wfdb.rdheader(info.record_name, pn_dir=info.pn_dir)
    except Exception:
        return None

    # Walk through segments (single-segment is treated as one-iter loop).
    if hasattr(hdr, 'seg_name') and hdr.seg_name:
        seg_iter = list(zip(hdr.seg_name, hdr.seg_len))
    else:
        seg_iter = [(info.record_name, getattr(hdr, 'sig_len', 0) or 0)]

    all_xs: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    for seg_name, seg_len in seg_iter:
        if seg_name == '~' or seg_name.endswith('_layout') or (seg_len and seg_len <= 0):
            continue
        loaded = _load_aligned_segment(seg_name, info.pn_dir, channel_map)
        if loaded is None:
            continue
        per_modal_native, native_sr = loaded

        per_modal: Dict[str, np.ndarray] = {}
        for modal, raw in per_modal_native.items():
            sig_at_target = (raw if native_sr == sfreq
                             else resample_to_target(raw, native_sr, target_sr=sfreq))
            if skip_preprocess:
                per_modal[modal] = sig_at_target.astype(np.float32)
            else:
                signal_key = MODAL_TO_SIGNAL_KEY[modal]
                per_modal[modal] = preprocess_channel(
                    sig_at_target, signal_key=signal_key, sr=float(sfreq),
                    cfg=SIGNAL_CONFIGS[signal_key],
                )

        xs, masks = extract_segments(per_modal, sfreq, duration, nan_max_ratio)
        if xs is None:
            continue
        all_xs.append(xs)
        all_masks.append(masks)

    if not all_xs:
        return None
    return np.concatenate(all_xs, axis=0), np.concatenate(all_masks, axis=0)


# ── Driver ───────────────────────────────────────────────────────


def parse_records_to_hetero(records: List[MimicRecordInfo],
                            trg_path: str,
                            sfreq: int = 100, duration: int = 60,
                            nan_max_ratio: float = 0.1,
                            n_cases: int = 0,
                            skip_preprocess: bool = False,
                            verbose: bool = True) -> None:
    os.makedirs(trg_path, exist_ok=True)
    if n_cases > 0:
        records = records[:n_cases]
    saved, skipped = 0, 0
    bucket_counts: Dict[str, int] = {}

    iterator = tqdm.tqdm(records, desc='MIMIC-III SSL') if verbose else records
    for info in iterator:
        result = process_record(info, sfreq=sfreq, duration=duration,
                                nan_max_ratio=nan_max_ratio,
                                skip_preprocess=skip_preprocess,
                                verbose=verbose)
        if result is None:
            skipped += 1
            continue
        xs, masks = result
        subject_modality_set = np.array([
            bool(info.abp_channel), bool(info.ecg_channel), bool(info.ppg_channel),
        ], dtype=bool)
        np.savez(
            os.path.join(trg_path, info.record_name + '.npz'),
            x=xs, mask=masks,
            modal_names=np.array(MODAL_ORDER),
            subject_modality_set=subject_modality_set,
            case_id=info.record_name,
        )
        bucket_key = ''.join('1' if subject_modality_set[i] else '0'
                             for i in range(len(MODAL_ORDER)))
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
        saved += 1

    print(f'[MIMIC3-SSL] saved={saved}, skipped={skipped}')
    print(f'[MIMIC3-SSL] bucket counts (ABP-ECG-PPG presence bitmap):')
    for k in sorted(bucket_counts.keys()):
        print(f'    {k}: {bucket_counts[k]}')


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MIMIC-III Waveform → hetero npz')
    sub = parser.add_subparsers(dest='command', required=True)

    sp_scan = sub.add_parser('scan', help='Scan and cache record manifest')
    sp_scan.add_argument('--max_records', type=int, default=0,
                         help='Max patients to scan (0 = all)')
    sp_scan.add_argument('--manifest', type=str, required=True,
                         help='Output manifest json path')
    sp_scan.add_argument('--require_abp', action='store_true')

    sp_parse = sub.add_parser('parse', help='Parse records to hetero npz')
    sp_parse.add_argument('--manifest', type=str, required=True)
    sp_parse.add_argument('--trg_path', type=str, required=True)
    sp_parse.add_argument('--n_cases', type=int, default=0,
                          help='Limit to first N records (0 = all)')
    sp_parse.add_argument('--sfreq', type=int, default=100)
    sp_parse.add_argument('--duration', type=int, default=60)
    sp_parse.add_argument('--nan_max_ratio', type=float, default=0.1)
    sp_parse.add_argument('--skip_preprocess', action='store_true')

    args = parser.parse_args()
    if args.command == 'scan':
        records = scan_records(max_records=args.max_records,
                               require_abp=args.require_abp)
        save_manifest(records, args.manifest)
        print(f'Manifest saved: {args.manifest}')
    elif args.command == 'parse':
        records = load_manifest(args.manifest)
        parse_records_to_hetero(records, trg_path=args.trg_path,
                                sfreq=args.sfreq, duration=args.duration,
                                nan_max_ratio=args.nan_max_ratio,
                                n_cases=args.n_cases,
                                skip_preprocess=args.skip_preprocess)


if __name__ == '__main__':
    main()
