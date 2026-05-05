# -*- coding:utf-8 -*-
"""Heterogeneous-availability dataset / dataloader for VitalDB SSL pretraining.

Reads npz files produced by ``dataset/data_parser/vital_db_ssl.py`` and
provides:

* ``HeteroVitalDBDataset`` — flat segment dataset. ``__getitem__`` returns one
  (segment, mask, presence_state) triple regardless of bucket.
* ``BucketBatchSampler`` — yields batches in which every sample shares the same
  segment-level modality-availability bitmap, so a single batch corresponds to
  one bucket. Sampling weights across buckets are configurable.
* ``hetero_collate_fn`` — turns a homogeneous-bucket batch into the
  ``data: dict[str, Tensor]`` form that ``PhysioME.forward`` expects, plus
  ``mask`` and ``presence_state`` tensors.

Presence-state encoding is the contract used by ``PhysioME.forward`` (after the
loss-decomposition refactor in Task 4):

* 0 — real-present  (segment has valid data for this modality)
* 1 — synthetically-dropped  (assigned later inside ``forward``, never here)
* 2 — naturally-absent  (segment did not have this modality)
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


MODAL_ORDER: List[str] = ['ABP', 'ECG', 'PPG', 'CO2']
NUM_MODALS: int = len(MODAL_ORDER)

PRESENCE_REAL: int = 0
PRESENCE_SYNTH_DROPPED: int = 1
PRESENCE_NATURALLY_ABSENT: int = 2


def _bitmap_to_key(bitmap: Sequence[bool]) -> str:
    return ''.join('1' if b else '0' for b in bitmap)


def _key_to_bitmap(key: str) -> np.ndarray:
    return np.array([c == '1' for c in key], dtype=bool)


class HeteroVitalDBDataset(Dataset):
    """Flat segment dataset over npz files produced by ``vital_db_ssl.py``.

    Loads metadata (per-segment masks, file pointers) eagerly so that
    bucket assignment is cheap. Segment payloads are loaded on demand from
    a per-file cache, so RAM usage stays bounded for large pretraining sets.

    Args:
        paths: list of npz file paths (one per case).
        eager: if True, load all segment payloads into memory at init time.
            Recommended only when the dataset comfortably fits in RAM.
        normalize: if True, z-score each modality channel within a segment.
            Done at ``__getitem__`` time. Missing modalities (mask=False) are
            left as zeros.
    """

    def __init__(self, paths: Sequence[str], eager: bool = False,
                 normalize: bool = True):
        self.paths: List[str] = list(paths)
        self.eager = eager
        self.normalize = normalize

        self._segment_index: List[Tuple[int, int]] = []  # (file_idx, seg_idx)
        self._segment_bitmap_keys: List[str] = []
        self._eager_x: Optional[List[np.ndarray]] = [] if eager else None
        self._eager_mask: Optional[List[np.ndarray]] = [] if eager else None
        self._lazy_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._lazy_cache_max = 64  # keep at most this many file handles

        for fi, p in enumerate(self.paths):
            with np.load(p, allow_pickle=True) as arr:
                masks = np.asarray(arr['mask'], dtype=bool)
                if eager:
                    xs = np.asarray(arr['x'], dtype=np.float32)
                    self._eager_x.append(xs)
                    self._eager_mask.append(masks)
            for si in range(masks.shape[0]):
                self._segment_index.append((fi, si))
                self._segment_bitmap_keys.append(_bitmap_to_key(masks[si]))

    def __len__(self) -> int:
        return len(self._segment_index)

    @property
    def segment_bitmap_keys(self) -> List[str]:
        return self._segment_bitmap_keys

    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.eager:
            return self._eager_x[file_idx], self._eager_mask[file_idx]
        if file_idx in self._lazy_cache:
            return self._lazy_cache[file_idx]
        with np.load(self.paths[file_idx], allow_pickle=True) as arr:
            xs = np.asarray(arr['x'], dtype=np.float32)
            masks = np.asarray(arr['mask'], dtype=bool)
        if len(self._lazy_cache) >= self._lazy_cache_max:
            self._lazy_cache.pop(next(iter(self._lazy_cache)))
        self._lazy_cache[file_idx] = (xs, masks)
        return xs, masks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, seg_idx = self._segment_index[idx]
        xs, masks = self._load_file(file_idx)
        x = xs[seg_idx].copy()  # [C, T]
        mask = masks[seg_idx].copy()  # [C]

        if self.normalize:
            for c in range(x.shape[0]):
                if not mask[c]:
                    continue
                ch = x[c]
                std = float(ch.std())
                if std > 1e-8:
                    x[c] = (ch - ch.mean()) / std

        presence_state = np.where(
            mask,
            np.int64(PRESENCE_REAL),
            np.int64(PRESENCE_NATURALLY_ABSENT),
        )

        return {
            'x': torch.from_numpy(x),
            'mask': torch.from_numpy(mask),
            'presence_state': torch.from_numpy(presence_state),
        }


class BucketBatchSampler(Sampler[List[int]]):
    """Yield batches of indices that share the same modality-availability bitmap.

    Args:
        bitmap_keys: list of per-segment bitmap keys (e.g., '111', '101').
        batch_size: per-batch size.
        sampling: 'natural' weights buckets by their segment count, 'uniform'
            gives equal time to every (eligible) bucket.
        min_bucket_size: drop buckets smaller than this (cannot form a batch).
        drop_last: drop final partial batch in each bucket per epoch.
        shuffle: shuffle within bucket each epoch.
        bucket_weights: optional explicit per-bucket weights (overrides
            ``sampling``). Keys are bitmap strings; values are floats.
        seed: rng seed.
    """

    def __init__(self,
                 bitmap_keys: Sequence[str],
                 batch_size: int,
                 sampling: str = 'natural',
                 min_bucket_size: Optional[int] = None,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 bucket_weights: Optional[Dict[str, float]] = None,
                 seed: int = 0):
        if sampling not in ('natural', 'uniform'):
            raise ValueError(f"sampling must be 'natural' or 'uniform', got {sampling!r}")
        self.batch_size = int(batch_size)
        self.sampling = sampling
        self.min_bucket_size = min_bucket_size if min_bucket_size is not None else self.batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        buckets: Dict[str, List[int]] = {}
        for i, key in enumerate(bitmap_keys):
            buckets.setdefault(key, []).append(i)
        self._buckets: Dict[str, List[int]] = {
            k: v for k, v in buckets.items() if len(v) >= self.min_bucket_size
        }
        if not self._buckets:
            raise RuntimeError(
                f'No bucket has at least {self.min_bucket_size} segments; '
                f'available bucket sizes: {sorted([len(v) for v in buckets.values()])}'
            )

        self._bucket_keys: List[str] = sorted(self._buckets.keys())
        sizes = np.array([len(self._buckets[k]) for k in self._bucket_keys],
                         dtype=np.float64)

        if bucket_weights is not None:
            w = np.array([bucket_weights.get(k, 0.0) for k in self._bucket_keys],
                         dtype=np.float64)
        elif sampling == 'natural':
            w = sizes
        else:
            w = np.ones_like(sizes)
        if w.sum() <= 0:
            raise RuntimeError('All bucket weights are zero')
        self._bucket_probs: np.ndarray = w / w.sum()

    @property
    def buckets(self) -> Dict[str, List[int]]:
        return self._buckets

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        total = 0
        for k in self._bucket_keys:
            n = len(self._buckets[k])
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        bucket_iters: Dict[str, List[List[int]]] = {}
        for k, idxs in self._buckets.items():
            order = list(idxs)
            if self.shuffle:
                rng.shuffle(order)
            batches = []
            for s in range(0, len(order), self.batch_size):
                batch = order[s:s + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
            bucket_iters[k] = batches

        remaining = {k: len(v) for k, v in bucket_iters.items()}
        cursor = {k: 0 for k in bucket_iters}

        while sum(remaining.values()) > 0:
            available = [k for k, r in remaining.items() if r > 0]
            if not available:
                break
            probs = np.array([self._bucket_probs[self._bucket_keys.index(k)]
                              for k in available])
            probs = probs / probs.sum()
            chosen = rng.choice(available, p=probs)
            batch = bucket_iters[chosen][cursor[chosen]]
            cursor[chosen] += 1
            remaining[chosen] -= 1
            yield batch


def hetero_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict:
    """Collate a homogeneous-bucket batch into PhysioME's expected dict format.

    Returns a dict containing:
        data: dict[modal_name -> Tensor[B, T]]
            Only modalities present in this bucket are included as keys.
        mask: BoolTensor[B, num_modals]
            Per-sample, per-modality validity. Within a bucket batch this is
            identical across samples; we still return per-sample for safety.
        presence_state: LongTensor[B, num_modals]
            0=real-present, 1=synth-dropped (set later by forward()), 2=naturally-absent.
        bucket_pattern: str
            e.g. '110' means ABP+ECG present, PPG absent.
    """
    xs = torch.stack([b['x'] for b in batch], dim=0)            # [B, C, T]
    masks = torch.stack([b['mask'] for b in batch], dim=0)       # [B, C]
    presence = torch.stack([b['presence_state'] for b in batch], dim=0)  # [B, C]

    bucket_bitmap = masks[0].cpu().numpy().astype(bool)
    bucket_pattern = _bitmap_to_key(bucket_bitmap)
    if not (masks == masks[0:1]).all():
        raise RuntimeError(
            'Batch is not homogeneous in modality presence — '
            'BucketBatchSampler must be used with hetero_collate_fn.'
        )

    data = {}
    for i, m in enumerate(MODAL_ORDER):
        if bucket_bitmap[i]:
            data[m] = xs[:, i, :]

    return {
        'data': data,
        'mask': masks,
        'presence_state': presence,
        'bucket_pattern': bucket_pattern,
    }


def find_ssl_npz_paths(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, '*.npz')))


if __name__ == '__main__':
    # Quick offline sanity check using synthetic npz files.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for case_idx in range(20):
            n_seg = int(rng.integers(5, 15))
            x = rng.standard_normal((n_seg, NUM_MODALS, 6000)).astype(np.float32)
            mask = np.zeros((n_seg, NUM_MODALS), dtype=bool)
            for s in range(n_seg):
                pattern = rng.integers(1, 8)
                for k in range(NUM_MODALS):
                    mask[s, k] = bool((pattern >> k) & 1)
            np.savez(os.path.join(tmp, f'case_{case_idx:03d}.npz'),
                     x=x, mask=mask,
                     modal_names=np.array(MODAL_ORDER),
                     subject_modality_set=mask.any(axis=0),
                     case_id=f'case_{case_idx:03d}')

        paths = find_ssl_npz_paths(tmp)
        ds = HeteroVitalDBDataset(paths, eager=True, normalize=True)
        sampler = BucketBatchSampler(ds.segment_bitmap_keys, batch_size=4,
                                     sampling='uniform', min_bucket_size=4)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=hetero_collate_fn)
        for i, batch in enumerate(loader):
            print(f'batch {i}: bucket={batch["bucket_pattern"]!r} '
                  f'modals={list(batch["data"].keys())} '
                  f'B={batch["mask"].shape[0]}')
            if i >= 4:
                break
