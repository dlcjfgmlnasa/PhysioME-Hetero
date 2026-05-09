# -*- coding:utf-8 -*-
"""SSL pretraining parser for VitalDB (sharded output).

Differences vs ``dataset/data_parser/vital_db.py``:
  1. No 2-minute lookahead validation — every valid window is an SSL sample.
  2. No ``is_all_in`` subject-level filter — subjects with any non-empty
     subset of {ABP, ECG, PPG, CVP} contribute data.
  3. Each segment carries a per-modality validity mask.
  4. Per-signal preprocessing pipeline (range check → spike detection →
     median → notch → bandpass/lowpass) ported from
     references/Biosignal-Foundation-Model.
  5. Segment validity uses three layered checks:
       (a) NaN ratio threshold,
       (b) generic quality score (flatline / clip / hf / amplitude) per signal,
       (c) physiology-aware domain check (HR + autocorr regularity etc.).

Output layout (under ``trg_path``) — sharded for slow network filesystems:

    shard_0000.npz, shard_0001.npz, ...   # ~``segments_per_shard`` segments each
    manifest.json                          # index over all shards

Each shard npz contains:
    x: float32 [N, M, sfreq * duration] — concatenated segments, zero where mask=False.
    mask: bool [N, M] — per-segment, per-modality validity.
    case_ids: object [N] — case id of each segment.
    case_offsets: int64 [K + 1] — segment index boundaries per case.
    case_ids_unique: object [K] — unique case ids in this shard, ordered by offsets.
    subject_modality_set: bool [K, M] — recording-level modality presence per case.
    modal_names: object [M] — fixed order from ``MODAL_ORDER``.

manifest.json schema:
    {
        "version": 1,
        "modal_order": [...],
        "sfreq": int, "duration": int,
        "shards": [
            {"path": "shard_NNNN.npz", "n_segments": N, "n_cases": K,
             "bitmap_keys": ["1111", "1110", ...]},
            ...
        ],
        "total_segments": int,
        "total_cases": int,
        "bucket_counts": {"1111": int, "1110": int, ...}   // segment-level
    }
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import tqdm

from dataset.data_parser._quality_checks import domain_quality_check
from dataset.data_parser._signal_filters import (
    SIGNAL_CONFIGS,
    SignalConfig,
    preprocess_channel,
    segment_quality_score,
)


MODAL_TRACK_NAMES: Dict[str, str] = {
    'ABP': 'SNUADC/ART',
    'ECG': 'SNUADC/ECG_II',
    'PPG': 'SNUADC/PLETH',
    # Step 2 (2026-05-05): 4-modal expansion — CVP (central venous pressure).
    # SNUADC/CVP is the 500Hz raw waveform from the analog port; resampled
    # to 100 Hz alongside the others. Available only in ~25% of cases (CV
    # catheter is invasive; major surgery only) — that sparsity is the
    # paper's main hetero-availability signal.
    'CVP': 'SNUADC/CVP',
    # Step 3 (2026-05-09): 6-modal expansion — respiratory bundle.
    # CO2 (capnography) — gas-exchange marker, available in most anesthesia
    # cases (~80% prevalence); Primus/CO2 is end-tidal CO2 from the GE Primus
    # ventilator/anesthesia workstation, sampled at 62.5Hz natively.
    # AWP (airway pressure) — lung mechanics marker, available in mechanical-
    # ventilation cases (~50% prevalence); Primus/AWP from the same source.
    # Together with CVP these define the v2 "cardio-pulmonary" foundation
    # modality set; respiratory pair complements the cardiovascular ABP/ECG/PPG
    # trio with directly-measured ventilation signals.
    'CO2': 'Primus/CO2',
    'AWP': 'Primus/AWP',
}
# short-name → key used by SIGNAL_CONFIGS / domain_quality_check
MODAL_TO_SIGNAL_KEY: Dict[str, str] = {
    'ABP': 'abp', 'ECG': 'ecg', 'PPG': 'ppg',
    'CVP': 'cvp', 'CO2': 'co2', 'AWP': 'awp',
}
MODAL_ORDER = ['ABP', 'ECG', 'PPG', 'CVP', 'CO2', 'AWP']

MANIFEST_NAME = 'manifest.json'
MANIFEST_VERSION = 1
DEFAULT_SEGMENTS_PER_SHARD = 2048


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path',
                        default=os.path.join('..', '..', '..', '..', 'Dataset', 'vitaldb'))
    parser.add_argument('--trg_path',
                        default=os.path.join('..', '..', 'data', 'vitaldb_ssl'))
    parser.add_argument('--sfreq', type=int, default=100)
    parser.add_argument('--duration', type=int, default=60,
                        help='window length in seconds')
    parser.add_argument('--nan_max_ratio', type=float, default=0.1)
    parser.add_argument('--segments_per_shard', type=int,
                        default=DEFAULT_SEGMENTS_PER_SHARD,
                        help='target number of segments per shard file '
                             '(~96 KB raw / segment at 100Hz × 60s × 4ch float32)')
    parser.add_argument('--no_compress', action='store_true',
                        help='write shards uncompressed (faster CPU, larger files)')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='disable per-channel preprocessing pipeline '
                             '(useful when input is already preprocessed)')
    return parser.parse_args()


def validate_segment(segment: np.ndarray,
                     signal_key: str,
                     cfg: SignalConfig,
                     sr: float,
                     nan_max_ratio: float) -> bool:
    """Three-layer validity check for one fixed-length segment of one channel.

    1) NaN ratio under threshold.
    2) ``segment_quality_score`` with signal-specific thresholds
       (flatline, clip, high-freq, amplitude).
    3) ``domain_quality_check`` (HR / pulse / autocorr regularity).

    NaN values are filled with the segment median for the score-based checks;
    they remain NaN in the returned data (which is later zero-filled by the
    caller).
    """
    if segment.size == 0:
        return False
    nan_ratio = float(np.isnan(segment).mean())
    if nan_ratio > nan_max_ratio:
        return False

    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return False

    if np.isnan(segment).any():
        median_val = float(np.nanmedian(segment))
        seg_filled = np.where(np.isnan(segment), median_val, segment)
    else:
        seg_filled = segment

    score = segment_quality_score(
        seg_filled,
        max_flatline_ratio=cfg.max_flatline_ratio,
        max_clip_ratio=cfg.max_clip_ratio,
        max_high_freq_ratio=cfg.max_high_freq_ratio,
        min_amplitude=cfg.min_amplitude,
        max_amplitude=cfg.max_amplitude,
        min_high_freq_ratio=cfg.min_high_freq_ratio,
    )
    if not score['pass']:
        return False

    domain = domain_quality_check(signal_key, seg_filled, sr=sr)
    if not domain.get('pass', True):
        return False
    return True


def fill_segment(segment: np.ndarray, expected_len: int) -> np.ndarray:
    s = np.asarray(segment, dtype=np.float32).reshape(-1)
    if s.shape[0] < expected_len:
        s = np.pad(s, (0, expected_len - s.shape[0]))
    elif s.shape[0] > expected_len:
        s = s[:expected_len]
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def extract_segments(data_per_modal: Dict[str, np.ndarray],
                     sfreq: int, duration: int,
                     nan_max_ratio: float) -> tuple:
    expected = sfreq * duration
    if not data_per_modal:
        return None, None
    max_len = max(arr.shape[0] for arr in data_per_modal.values())
    num_windows = max_len // expected
    if num_windows == 0:
        return None, None

    xs = np.zeros((num_windows, len(MODAL_ORDER), expected), dtype=np.float32)
    masks = np.zeros((num_windows, len(MODAL_ORDER)), dtype=bool)

    kept = 0
    for w in range(num_windows):
        s, e = w * expected, (w + 1) * expected
        any_valid = False
        win_x = np.zeros((len(MODAL_ORDER), expected), dtype=np.float32)
        win_mask = np.zeros(len(MODAL_ORDER), dtype=bool)
        for i, m in enumerate(MODAL_ORDER):
            if m not in data_per_modal:
                continue
            arr = data_per_modal[m]
            if arr.shape[0] < e:
                continue
            seg = arr[s:e]
            signal_key = MODAL_TO_SIGNAL_KEY[m]
            cfg = SIGNAL_CONFIGS[signal_key]
            if validate_segment(seg, signal_key, cfg, float(sfreq),
                                nan_max_ratio):
                win_x[i] = fill_segment(seg, expected)
                win_mask[i] = True
                any_valid = True
        if any_valid:
            xs[kept] = win_x
            masks[kept] = win_mask
            kept += 1

    if kept == 0:
        return None, None
    return xs[:kept], masks[:kept]


def _bitmap_to_key(bitmap) -> str:
    return ''.join('1' if b else '0' for b in bitmap)


class ShardWriter:
    """Buffer per-case segments and flush them as concatenated shard files.

    Designed to amortise the cost of writing to slow network drives: instead of
    one tiny ``.npz`` per case (thousands of metadata round-trips) we batch
    ~``segments_per_shard`` segments into a single compressed shard.

    Usage::

        writer = ShardWriter(trg_path, MODAL_ORDER, segments_per_shard=2048)
        for case_id, xs, masks, subj_set in cases:
            writer.add_case(case_id, xs, masks, subj_set)
        writer.close(extra_meta={'sfreq': 100, 'duration': 60})

    ``close()`` writes ``manifest.json`` describing every shard, including
    per-segment bitmap keys so that ``HeteroVitalDBDataset`` can build its
    bucket sampler without re-reading any shard payload.
    """

    def __init__(self, trg_path: str, modal_order: List[str],
                 segments_per_shard: int = DEFAULT_SEGMENTS_PER_SHARD,
                 compress: bool = True):
        self.trg_path = trg_path
        self.modal_order = list(modal_order)
        self.segments_per_shard = int(segments_per_shard)
        self.compress = bool(compress)
        os.makedirs(trg_path, exist_ok=True)

        self._buf_xs: List[np.ndarray] = []
        self._buf_masks: List[np.ndarray] = []
        self._buf_case_ids: List[str] = []        # one entry per case in the buffer
        self._buf_case_n_seg: List[int] = []      # segments contributed per case
        self._buf_subject_set: List[np.ndarray] = []
        self._buf_total_segments: int = 0

        self._shards: List[dict] = []             # manifest entries
        self._bucket_counts: Dict[str, int] = {}  # global, segment-level
        self._total_segments: int = 0
        self._total_cases: int = 0

    # ---------------- buffering ----------------
    def add_case(self, case_id: str, xs: np.ndarray, masks: np.ndarray,
                 subject_modality_set: np.ndarray) -> None:
        if xs.shape[0] == 0:
            return
        if xs.shape[1] != len(self.modal_order):
            raise ValueError(
                f'add_case: xs has {xs.shape[1]} modals but writer was built '
                f'for {len(self.modal_order)} ({self.modal_order})'
            )
        self._buf_xs.append(np.ascontiguousarray(xs, dtype=np.float32))
        self._buf_masks.append(np.ascontiguousarray(masks, dtype=bool))
        self._buf_case_ids.append(str(case_id))
        self._buf_case_n_seg.append(int(xs.shape[0]))
        self._buf_subject_set.append(
            np.ascontiguousarray(subject_modality_set, dtype=bool)
        )
        self._buf_total_segments += int(xs.shape[0])
        self._total_cases += 1
        if self._buf_total_segments >= self.segments_per_shard:
            self._flush()

    # ---------------- flush ----------------
    def _flush(self) -> None:
        if self._buf_total_segments == 0:
            return
        shard_idx = len(self._shards)
        shard_name = f'shard_{shard_idx:04d}.npz'
        shard_path = os.path.join(self.trg_path, shard_name)

        xs = np.concatenate(self._buf_xs, axis=0)        # [N, M, T]
        masks = np.concatenate(self._buf_masks, axis=0)  # [N, M]
        case_offsets = np.zeros(len(self._buf_case_n_seg) + 1, dtype=np.int64)
        case_offsets[1:] = np.cumsum(self._buf_case_n_seg, dtype=np.int64)
        case_ids_per_seg = np.empty(xs.shape[0], dtype=object)
        for k, (cid, n) in enumerate(zip(self._buf_case_ids, self._buf_case_n_seg)):
            case_ids_per_seg[case_offsets[k]:case_offsets[k] + n] = cid
        case_ids_unique = np.array(self._buf_case_ids, dtype=object)
        subject_set = np.stack(self._buf_subject_set, axis=0)  # [K, M]

        save_fn = np.savez_compressed if self.compress else np.savez
        save_fn(
            shard_path,
            x=xs, mask=masks,
            case_ids=case_ids_per_seg,
            case_offsets=case_offsets,
            case_ids_unique=case_ids_unique,
            subject_modality_set=subject_set,
            modal_names=np.array(self.modal_order),
        )

        bitmap_keys = [_bitmap_to_key(m) for m in masks]
        for k in bitmap_keys:
            self._bucket_counts[k] = self._bucket_counts.get(k, 0) + 1

        self._shards.append({
            'path': shard_name,
            'n_segments': int(xs.shape[0]),
            'n_cases': int(len(self._buf_case_ids)),
            'bitmap_keys': bitmap_keys,
        })
        self._total_segments += int(xs.shape[0])

        self._buf_xs.clear(); self._buf_masks.clear()
        self._buf_case_ids.clear(); self._buf_case_n_seg.clear()
        self._buf_subject_set.clear()
        self._buf_total_segments = 0

    def close(self, extra_meta: Optional[Dict] = None) -> str:
        """Flush any remaining buffered cases and write manifest. Returns path."""
        self._flush()
        manifest = {
            'version': MANIFEST_VERSION,
            'modal_order': list(self.modal_order),
            'segments_per_shard': self.segments_per_shard,
            'compressed': self.compress,
            'shards': self._shards,
            'total_segments': self._total_segments,
            'total_cases': self._total_cases,
            'bucket_counts': self._bucket_counts,
        }
        if extra_meta:
            for k, v in extra_meta.items():
                manifest[k] = v
        manifest_path = os.path.join(self.trg_path, MANIFEST_NAME)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    # ---------------- introspection ----------------
    @property
    def bucket_counts(self) -> Dict[str, int]:
        return dict(self._bucket_counts)

    @property
    def total_segments(self) -> int:
        return self._total_segments

    @property
    def total_cases(self) -> int:
        return self._total_cases

    @property
    def num_shards(self) -> int:
        return len(self._shards)


def vitaldb_ssl_converter(src_path: str, trg_path: str,
                          sfreq: int = 100, duration: int = 60,
                          nan_max_ratio: float = 0.1,
                          segments_per_shard: int = DEFAULT_SEGMENTS_PER_SHARD,
                          compress: bool = True,
                          skip_preprocess: bool = False) -> None:
    import vitaldb  # imported lazily so validate_segment can be used without it
    paths = sorted(os.listdir(src_path))

    writer = ShardWriter(
        trg_path=trg_path, modal_order=MODAL_ORDER,
        segments_per_shard=segments_per_shard, compress=compress,
    )
    skipped = 0

    for fname in tqdm.tqdm(paths, desc='VitalDB-SSL'):
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

        data_per_modal: Dict[str, np.ndarray] = {}
        for i, m in enumerate(present_modals):
            channel = np.asarray(data[:, i], dtype=np.float32)
            if not skip_preprocess:
                channel = preprocess_channel(
                    channel, signal_key=MODAL_TO_SIGNAL_KEY[m], sr=float(sfreq),
                )
            data_per_modal[m] = channel

        xs, masks = extract_segments(data_per_modal, sfreq, duration,
                                     nan_max_ratio)
        if xs is None:
            skipped += 1
            continue

        subject_modality_set = np.array(
            [m in present_modals for m in MODAL_ORDER], dtype=bool,
        )
        writer.add_case(case_id, xs, masks, subject_modality_set)

    manifest_path = writer.close(extra_meta={
        'sfreq': int(sfreq), 'duration': int(duration),
        'nan_max_ratio': float(nan_max_ratio),
        'preprocess': not skip_preprocess,
    })

    print(f'[VitalDB-SSL] saved={writer.total_cases} cases / '
          f'{writer.total_segments} segments / {writer.num_shards} shards, '
          f'skipped={skipped}')
    print(f'[VitalDB-SSL] manifest: {manifest_path}')
    print(f'[VitalDB-SSL] segment-level bucket counts '
          f'(bitmap = {"".join(MODAL_ORDER)}):')
    for k in sorted(writer.bucket_counts.keys()):
        print(f'    {k}: {writer.bucket_counts[k]}')


if __name__ == '__main__':
    args = get_args()
    vitaldb_ssl_converter(
        src_path=args.src_path,
        trg_path=args.trg_path,
        sfreq=args.sfreq,
        duration=args.duration,
        nan_max_ratio=args.nan_max_ratio,
        segments_per_shard=args.segments_per_shard,
        compress=not args.no_compress,
        skip_preprocess=args.skip_preprocess,
    )
