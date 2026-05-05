# -*- coding:utf-8 -*-
"""Single-modality view over the sharded SSL output for Phase-1 NeuroNet
(TF-C) pretraining.

Reads the same on-disk layout as ``pretrained/physiome/hetero_data_loader.py``:

    <data_dir>/
        manifest.json
        shard_0000.npz
        shard_0001.npz
        ...

Per ``ch_idx`` (modality index in MODAL_ORDER), this loader yields only
segments whose ``mask[:, ch_idx] == True`` — i.e. the modality is actually
recorded and passed QC for that segment. Naturally-absent and QC-failed slots
are filtered out.

Train / val / eval split is **shard-level**: shards are written in case-arrival
order by the parser, so a contiguous shard slice is approximately subject-
disjoint without paying the cost of reading per-shard ``case_ids`` at init
(which would defeat the manifest-only init the shard layout was designed for).
If the manifest later carries ``case_ids`` per shard we can sharpen this to a
strict subject split, but for SSL pretraining shard-level is fine.

Phase-1 is unsupervised — no labels exist in the SSL parser output. Items
return ``(x, dummy_y=0)`` so the existing trainer signature
``for x, _ in loader`` keeps working without changes.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


MANIFEST_NAME: str = 'manifest.json'


def _read_manifest(data_dir: str) -> dict:
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f'No manifest at {manifest_path!r}. Run '
            f'dataset/data_parser/vital_db_ssl.py to populate {data_dir!r}.'
        )
    with open(manifest_path, 'r') as f:
        return json.load(f)


def split_shards(num_shards: int, val_ratio: float, eval_ratio: float,
                 seed: int = 0) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic shard split. Returns (train, val, eval) shard index lists.

    Eval uses the trailing slice (closest to "future" cases in alphabetical
    parser order); val uses the slice just before. Train is everything else.
    Seed unused for now — kept for API parity with random splits.
    """
    if num_shards <= 0:
        raise ValueError('split_shards: num_shards must be positive')
    n_eval = max(1, int(round(num_shards * eval_ratio)))
    n_val = max(1, int(round(num_shards * val_ratio)))
    if n_val + n_eval >= num_shards:
        # Tiny dataset — collapse to a single train/val/eval shard each.
        train = list(range(max(1, num_shards - 2)))
        val = [num_shards - 2] if num_shards >= 2 else train
        eval_ = [num_shards - 1]
        return train, val, eval_
    train_end = num_shards - n_val - n_eval
    train = list(range(train_end))
    val = list(range(train_end, train_end + n_val))
    eval_ = list(range(train_end + n_val, num_shards))
    return train, val, eval_


class ShardSingleModalDataset(Dataset):
    """Yield single-modality segments for one ``ch_idx`` from sharded SSL data.

    Args:
        data_dir: directory containing manifest.json + shard_NNNN.npz.
        ch_idx: index into the manifest's ``modal_order``.
        shard_indices: subset of shards to use (typically a train/val/eval slice
            from :func:`split_shards`). If None, use all shards.
        normalize: z-score the channel within each segment.
        shard_cache_size: lazy LRU window in shards (default 4).
    """

    def __init__(self, data_dir: str, ch_idx: int,
                 shard_indices: Optional[Sequence[int]] = None,
                 normalize: bool = True, shard_cache_size: int = 4):
        self.data_dir = data_dir
        self.ch_idx = int(ch_idx)
        self.normalize = bool(normalize)
        self.shard_cache_size = max(1, int(shard_cache_size))

        manifest = _read_manifest(data_dir)
        self.modal_order: List[str] = list(manifest.get('modal_order', []))
        if not self.modal_order:
            raise RuntimeError(f'Manifest at {data_dir} has empty modal_order.')
        if not (0 <= self.ch_idx < len(self.modal_order)):
            raise ValueError(
                f'ch_idx={self.ch_idx} out of range for '
                f'modal_order={self.modal_order}'
            )
        self.modal_name: str = self.modal_order[self.ch_idx]

        all_shards: List[dict] = manifest['shards']
        if not all_shards:
            raise RuntimeError(f'Manifest at {data_dir} has zero shards.')

        if shard_indices is None:
            shard_indices = list(range(len(all_shards)))
        shard_indices = sorted(set(int(i) for i in shard_indices))
        if not shard_indices:
            raise RuntimeError('shard_indices empty after dedup.')
        for si in shard_indices:
            if not (0 <= si < len(all_shards)):
                raise ValueError(
                    f'shard_indices contains out-of-range {si} '
                    f'(have {len(all_shards)} shards).'
                )
        self._shards: List[dict] = [all_shards[i] for i in shard_indices]
        self._shard_ids: List[int] = list(shard_indices)

        # Filter to segments where this modality is actually present (mask=1).
        # The manifest carries per-segment ``bitmap_keys`` so this is cheap.
        self._segment_index: List[Tuple[int, int]] = []  # (local_shard_pos, seg_idx)
        for li, shard in enumerate(self._shards):
            keys = shard['bitmap_keys']
            n = int(shard['n_segments'])
            if len(keys) != n:
                raise RuntimeError(
                    f'Shard {shard["path"]} reports n_segments={n} but '
                    f'manifest lists {len(keys)} bitmap_keys.'
                )
            for seg, key in enumerate(keys):
                if key[self.ch_idx] == '1':
                    self._segment_index.append((li, seg))

        if not self._segment_index:
            raise RuntimeError(
                f'No segments with modality {self.modal_name!r} '
                f'(ch_idx={self.ch_idx}) across {len(self._shards)} shards.'
            )

        self._lazy_cache: 'OrderedDict[int, Tuple[np.ndarray, np.ndarray]]' = OrderedDict()

    def __len__(self) -> int:
        return len(self._segment_index)

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    def _load_shard(self, local_shard_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._lazy_cache.get(local_shard_pos)
        if cached is not None:
            self._lazy_cache.move_to_end(local_shard_pos)
            return cached
        path = os.path.join(self.data_dir, self._shards[local_shard_pos]['path'])
        with np.load(path, allow_pickle=True) as arr:
            xs = np.asarray(arr['x'], dtype=np.float32)        # [N, M, T]
            masks = np.asarray(arr['mask'], dtype=bool)        # [N, M]
        self._lazy_cache[local_shard_pos] = (xs, masks)
        while len(self._lazy_cache) > self.shard_cache_size:
            self._lazy_cache.popitem(last=False)
        return xs, masks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        local_shard_pos, seg_idx = self._segment_index[idx]
        xs, _masks = self._load_shard(local_shard_pos)
        x = xs[seg_idx, self.ch_idx].copy()  # [T]

        if self.normalize:
            std = float(x.std())
            if std > 1e-8:
                x = (x - x.mean()) / std

        # NeuroNet's TF-C trainer signature is ``for x, _ in loader``.
        # SSL has no labels, so y is a dummy zero tensor.
        return torch.from_numpy(x), torch.tensor(0, dtype=torch.long)


if __name__ == '__main__':
    import tempfile
    from dataset.data_parser.vital_db_ssl import ShardWriter

    MODAL = ['ABP', 'ECG', 'PPG', 'CVP']
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        writer = ShardWriter(tmp, MODAL, segments_per_shard=64, compress=True)
        for ci in range(20):
            n_seg = int(rng.integers(8, 16))
            x = rng.standard_normal((n_seg, len(MODAL), 6000)).astype(np.float32)
            mask = np.zeros((n_seg, len(MODAL)), dtype=bool)
            for s in range(n_seg):
                p = rng.integers(1, 1 << len(MODAL))
                for k in range(len(MODAL)):
                    mask[s, k] = bool((p >> k) & 1)
                if not mask[s].any():
                    mask[s, 0] = True
            for s in range(n_seg):
                for k in range(len(MODAL)):
                    if not mask[s, k]:
                        x[s, k] = 0.0
            writer.add_case(f'case_{ci:03d}', x, mask, mask.any(axis=0))
        writer.close(extra_meta={'sfreq': 100, 'duration': 60})

        train, val, eval_ = split_shards(num_shards=writer.num_shards,
                                         val_ratio=0.2, eval_ratio=0.2)
        print(f'shards: {writer.num_shards}  train={train}  val={val}  eval={eval_}')
        for ch_idx, m in enumerate(MODAL):
            ds = ShardSingleModalDataset(tmp, ch_idx=ch_idx,
                                         shard_indices=train, normalize=True)
            sample, y = ds[0]
            print(f'  {m:>4s}: train segments={len(ds):4d}  '
                  f'sample shape={tuple(sample.shape)}  y={y.item()}')
