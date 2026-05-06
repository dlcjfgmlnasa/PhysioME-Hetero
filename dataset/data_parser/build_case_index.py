# -*- coding:utf-8 -*-
"""Sidecar generator for the sharded SSL output.

Reads each ``shard_NNNN.npz`` once, extracts the per-shard
``case_ids_unique`` + ``case_offsets`` arrays, and writes
``case_index.json`` next to ``manifest.json``. The sidecar lets
downstream loaders (Phase-1 ``ShardSingleModalDataset``, Phase-2
``HeteroVitalDBDataset``) build a segment -> case_id map at init time
without having to open every shard themselves (which on a slow network
filesystem would take ~100 minutes).

Output schema::

    {
      "version": 1,
      "manifest": "manifest.json",
      "all_case_ids": [...],          # union across shards (deduped)
      "shards": [
        {
          "path": "shard_0000.npz",
          "case_ids_unique": [...],   # K case ids in this shard
          "case_offsets": [0, ...],   # length K+1, segment boundaries
          "n_segments": int
        },
        ...
      ]
    }

CLI::

    python -m dataset.data_parser.build_case_index \
        --data_dir /path/to/vitaldb_ssl
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Set

import numpy as np
import tqdm


MANIFEST_NAME = 'manifest.json'
CASE_INDEX_NAME = 'case_index.json'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory containing manifest.json + shard_NNNN.npz')
    parser.add_argument('--out_name', type=str, default=CASE_INDEX_NAME,
                        help='sidecar filename (default: case_index.json)')
    parser.add_argument('--force', action='store_true',
                        help='overwrite existing sidecar')
    return parser.parse_args()


def build_case_index(data_dir: str, out_name: str = CASE_INDEX_NAME,
                     force: bool = False) -> str:
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f'No manifest at {manifest_path!r}')
    out_path = os.path.join(data_dir, out_name)
    if os.path.isfile(out_path) and not force:
        raise FileExistsError(
            f'{out_path!r} exists; pass --force to overwrite.'
        )

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    shards_meta: List[dict] = []
    seen_case_ids: Set[str] = set()

    for shard in tqdm.tqdm(manifest['shards'], desc='build_case_index'):
        path = os.path.join(data_dir, shard['path'])
        with np.load(path, allow_pickle=True) as arr:
            case_ids_unique = [str(c) for c in arr['case_ids_unique']]
            case_offsets = [int(v) for v in arr['case_offsets']]
            n_segments = int(arr['mask'].shape[0])
        if len(case_ids_unique) + 1 != len(case_offsets):
            raise RuntimeError(
                f'Shard {shard["path"]} has {len(case_ids_unique)} cases '
                f'but {len(case_offsets)} offsets (expected K+1).'
            )
        if case_offsets[-1] != n_segments:
            raise RuntimeError(
                f'Shard {shard["path"]} case_offsets[-1]={case_offsets[-1]} '
                f'!= n_segments={n_segments}.'
            )
        seen_case_ids.update(case_ids_unique)
        shards_meta.append({
            'path': shard['path'],
            'n_segments': n_segments,
            'case_ids_unique': case_ids_unique,
            'case_offsets': case_offsets,
        })

    out = {
        'version': 1,
        'manifest': MANIFEST_NAME,
        'all_case_ids': sorted(seen_case_ids),
        'shards': shards_meta,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[build_case_index] wrote {out_path}')
    print(f'                    shards={len(shards_meta)}  '
          f'unique cases={len(seen_case_ids)}')
    return out_path


if __name__ == '__main__':
    args = get_args()
    build_case_index(args.data_dir, out_name=args.out_name, force=args.force)
