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
from collections import OrderedDict, defaultdict
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, Sampler


MANIFEST_NAME: str = 'manifest.json'
CASE_INDEX_NAME: str = 'case_index.json'


def load_holdout_case_ids(path: Optional[str]) -> Optional[set]:
    """Load a holdout-subjects JSON written by ``sample_holdout``."""
    if not path:
        return None
    with open(path, 'r') as f:
        payload = json.load(f)
    return set(str(c) for c in payload['case_ids'])


def _build_segment_case_ids(data_dir: str,
                            shard_paths: List[str]) -> Optional[List[List[str]]]:
    """Read ``case_index.json`` and expand into per-shard segment->case_id lists.

    Returns ``None`` if the sidecar is absent.
    """
    sidecar = os.path.join(data_dir, CASE_INDEX_NAME)
    if not os.path.isfile(sidecar):
        return None
    with open(sidecar, 'r') as f:
        ci = json.load(f)
    by_path = {s['path']: s for s in ci['shards']}
    out: List[List[str]] = []
    for sp in shard_paths:
        meta = by_path.get(sp)
        if meta is None:
            return None
        case_ids_unique = meta['case_ids_unique']
        case_offsets = meta['case_offsets']
        seg_case_ids = []
        for k, cid in enumerate(case_ids_unique):
            seg_case_ids.extend([cid] * (case_offsets[k + 1] - case_offsets[k]))
        out.append(seg_case_ids)
    return out


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
                 normalize: bool = True, shard_cache_size: int = 4,
                 exclude_case_ids: Optional[set] = None,
                 eager: bool = False, eager_workers: int = 8):
        self.data_dir = data_dir
        self.ch_idx = int(ch_idx)
        self.normalize = bool(normalize)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.eager = bool(eager)
        self.eager_workers = max(1, int(eager_workers))

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

        # Optional case-id holdout: requires case_index.json sidecar.
        seg_case_ids_per_shard: Optional[List[List[str]]] = None
        if exclude_case_ids:
            seg_case_ids_per_shard = _build_segment_case_ids(
                data_dir, [s['path'] for s in self._shards],
            )
            if seg_case_ids_per_shard is None:
                raise FileNotFoundError(
                    f'exclude_case_ids was supplied but no '
                    f'{CASE_INDEX_NAME!r} sidecar was found in {data_dir!r}. '
                    f'Run: python -m dataset.data_parser.build_case_index '
                    f'--data_dir {data_dir!r}'
                )

        # Filter to segments where this modality is actually present (mask=1).
        # The manifest carries per-segment ``bitmap_keys`` so this is cheap.
        self._segment_index: List[Tuple[int, int]] = []  # (local_shard_pos, seg_idx)
        n_excluded = 0
        for li, shard in enumerate(self._shards):
            keys = shard['bitmap_keys']
            n = int(shard['n_segments'])
            if len(keys) != n:
                raise RuntimeError(
                    f'Shard {shard["path"]} reports n_segments={n} but '
                    f'manifest lists {len(keys)} bitmap_keys.'
                )
            for seg, key in enumerate(keys):
                if key[self.ch_idx] != '1':
                    continue
                if seg_case_ids_per_shard is not None:
                    if seg_case_ids_per_shard[li][seg] in exclude_case_ids:
                        n_excluded += 1
                        continue
                self._segment_index.append((li, seg))
        self._n_excluded_segments = n_excluded

        if not self._segment_index:
            raise RuntimeError(
                f'No segments with modality {self.modal_name!r} '
                f'(ch_idx={self.ch_idx}) across {len(self._shards)} shards.'
            )

        self._lazy_cache: 'OrderedDict[int, Tuple[np.ndarray, np.ndarray]]' = OrderedDict()
        # Eager mode: pre-load only the relevant (modality, kept-segment) slices
        # into a single in-memory float32 array, sized
        # ``[len(_segment_index), T]``. Avoids re-reading network shards
        # on every batch -- worth it when shards live on a slow filesystem and
        # the resulting array fits in RAM.
        self._eager_x: Optional[np.ndarray] = None
        if self.eager:
            self._eager_x = self._build_eager_array()

    def _build_eager_array(self) -> np.ndarray:
        """Read each needed shard ONCE in parallel and return a flat float32
        array of shape ``[len(self._segment_index), T]`` where row ``i`` is
        the modality channel data for ``self._segment_index[i]``.
        """
        # Group required (global_idx, seg_idx) by shard so each shard is
        # opened exactly once.
        by_shard: 'defaultdict[int, List[Tuple[int, int]]]' = defaultdict(list)
        for global_i, (shard_pos, seg_idx) in enumerate(self._segment_index):
            by_shard[shard_pos].append((global_i, seg_idx))

        # Probe one shard to learn T (segment length). Cheap relative to
        # the full eager load.
        sample_path = os.path.join(self.data_dir,
                                   self._shards[next(iter(by_shard))]['path'])
        with np.load(sample_path, allow_pickle=True) as arr:
            T = int(arr['x'].shape[-1])

        out = np.empty((len(self._segment_index), T), dtype=np.float32)

        def _load_one(shard_pos: int):
            path = os.path.join(self.data_dir, self._shards[shard_pos]['path'])
            with np.load(path, allow_pickle=True) as arr:
                xs = np.asarray(arr['x'], dtype=np.float32)  # [N, M, T]
            picks = by_shard[shard_pos]
            results: List[Tuple[int, np.ndarray]] = []
            for global_i, seg_idx in picks:
                results.append((global_i, xs[seg_idx, self.ch_idx, :].copy()))
            return results

        if self.eager_workers <= 1:
            it = (_load_one(s) for s in by_shard)
        else:
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(self.eager_workers)
            it = pool.imap_unordered(_load_one, by_shard.keys())

        try:
            for results in tqdm.tqdm(it, total=len(by_shard),
                                     desc=f'eager-load ch={self.ch_idx}'):
                for gi, vec in results:
                    out[gi] = vec
        finally:
            if self.eager_workers > 1:
                pool.close()
                pool.join()

        return out

    def __len__(self) -> int:
        return len(self._segment_index)

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def n_excluded_segments(self) -> int:
        return self._n_excluded_segments

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
        if self._eager_x is not None:
            x = self._eager_x[idx].copy()  # [T]
        else:
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


class ShardSequentialSampler(Sampler[int]):
    """Walk shards one at a time; segments within each shard are shuffled.

    Why: with random segment shuffling across 521 shards (~30 GB on a slow
    network drive), every batch hits a different shard -> cache hit rate ~0%
    -> dataloader pays the full ~10s shard decompression every batch -> GPU
    starves. Walking shards sequentially means each shard is loaded **exactly
    once per epoch** instead of dozens of times. Combined with
    ``persistent_workers=True`` and ``prefetch_factor>=2`` on the DataLoader,
    workers prefetch the next shard while the trainer chews through the
    current one, so most of the network read is hidden behind GPU compute.

    Per-epoch behaviour:
      1. Shuffle the *order* of shards (not segments across shards).
      2. For each shard, shuffle the segments inside it and yield them.

    Cross-epoch determinism: ``set_epoch(epoch)`` updates the rng seed.

    Args:
        dataset: a ``ShardSingleModalDataset`` (uses ``_segment_index`` to
            recover the per-segment shard membership).
        seed: rng seed for shard-order + intra-shard shuffles.
        shuffle_shards: if False, walk shards in their stored order each epoch
            (still shuffles segments inside each shard). Useful for
            reproducibility / debugging.
    """

    def __init__(self, dataset: 'ShardSingleModalDataset',
                 seed: int = 0, shuffle_shards: bool = True):
        if not hasattr(dataset, '_segment_index'):
            raise TypeError(
                'ShardSequentialSampler expects a ShardSingleModalDataset.'
            )
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self._epoch = 0

        # Group dataset indices by shard position.
        by_shard: 'OrderedDict[int, List[int]]' = OrderedDict()
        for global_idx, (shard_pos, _seg) in enumerate(dataset._segment_index):
            by_shard.setdefault(shard_pos, []).append(global_idx)
        self._by_shard = by_shard
        self._n = sum(len(v) for v in by_shard.values())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        shard_keys = list(self._by_shard.keys())
        if self.shuffle_shards:
            rng.shuffle(shard_keys)
        for s in shard_keys:
            indices = list(self._by_shard[s])
            rng.shuffle(indices)
            for idx in indices:
                yield idx


if __name__ == '__main__':
    import tempfile
    from dataset.data_parser.vital_db_ssl import ShardWriter

    MODAL = ['ABP', 'ECG', 'PPG', 'CVP', 'CO2', 'AWP']
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
