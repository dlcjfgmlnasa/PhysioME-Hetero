# -*- coding:utf-8 -*-
"""Deterministic holdout-subject sampler.

Picks N case ids from ``case_index.json`` to serve as a held-out cohort
that is excluded from BOTH Phase-1 and Phase-2 SSL training. The same
list should later drive downstream evaluation, ensuring no subject is
seen in any pretraining stage before testing.

Default ``--n 100`` matches ``holdout_subject_size`` in the legacy
hetero config so the new SSL holdout aligns with the labeled-probe
holdout convention.

Optional bucket stratification (default off) preserves the natural
modality-availability mix in the holdout cohort. With 4-modal VitalDB
this matters most for the CVP-bearing buckets (~25% of segments) — a
plain random 100-of-6354 sample can easily under-represent CVP.

CLI::

    python -m dataset.data_parser.sample_holdout \
        --data_dir /path/to/vitaldb_ssl \
        --n 100 --seed 777
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


CASE_INDEX_NAME = 'case_index.json'
HOLDOUT_NAME = 'holdout_case_ids.json'
DEV_NAME = 'dev_case_ids.json'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n', type=int, default=100,
                        help='number of holdout (downstream test) subjects (default 100)')
    parser.add_argument('--n_dev', type=int, default=0,
                        help='number of dev (probe / model selection) subjects, '
                             'sampled disjoint from holdout. Set 0 to skip.')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--out_name', type=str, default=HOLDOUT_NAME)
    parser.add_argument('--dev_out_name', type=str, default=DEV_NAME)
    parser.add_argument('--stratify_by_cvp', action='store_true',
                        help='ensure holdout/dev have roughly the same CVP '
                             'coverage as the full dataset')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _case_to_dominant_bucket(case_index: dict, manifest: dict) -> Dict[str, str]:
    """Map case_id -> the most common bitmap key among its segments."""
    by_shard_keys = {s['path']: s['bitmap_keys'] for s in manifest['shards']}
    case_buckets: Dict[str, Counter] = defaultdict(Counter)
    for shard in case_index['shards']:
        keys = by_shard_keys[shard['path']]
        case_ids = shard['case_ids_unique']
        offsets = shard['case_offsets']
        for k, cid in enumerate(case_ids):
            s, e = offsets[k], offsets[k + 1]
            for key in keys[s:e]:
                case_buckets[cid][key] += 1
    return {cid: ctr.most_common(1)[0][0] for cid, ctr in case_buckets.items()}


def _stratified_choice(rng: np.random.Generator, pool_cvp: List[str],
                       pool_non: List[str], n: int) -> Tuple[List[str], List[str]]:
    """Pick ``n`` ids from (pool_cvp, pool_non) preserving CVP ratio."""
    total = len(pool_cvp) + len(pool_non)
    cvp_ratio = len(pool_cvp) / max(total, 1)
    n_cvp = max(1, int(round(n * cvp_ratio))) if pool_cvp else 0
    n_cvp = min(n_cvp, len(pool_cvp), n)
    n_non = min(n - n_cvp, len(pool_non))
    chosen_cvp = rng.choice(pool_cvp, size=n_cvp, replace=False).tolist() \
        if n_cvp > 0 else []
    chosen_non = rng.choice(pool_non, size=n_non, replace=False).tolist() \
        if n_non > 0 else []
    return chosen_cvp, chosen_non


def sample_holdout(data_dir: str, n: int, seed: int,
                   n_dev: int = 0,
                   out_name: str = HOLDOUT_NAME,
                   dev_out_name: str = DEV_NAME,
                   stratify_by_cvp: bool = False,
                   force: bool = False) -> Dict[str, str]:
    case_index_path = os.path.join(data_dir, CASE_INDEX_NAME)
    if not os.path.isfile(case_index_path):
        raise FileNotFoundError(
            f'{case_index_path!r} not found. Run build_case_index first.'
        )

    out_path = os.path.join(data_dir, out_name)
    dev_out_path = os.path.join(data_dir, dev_out_name) if n_dev > 0 else None
    for p in (out_path, dev_out_path):
        if p and os.path.isfile(p) and not force:
            raise FileExistsError(f'{p!r} exists; pass --force.')

    with open(case_index_path, 'r') as f:
        case_index = json.load(f)
    all_case_ids: List[str] = case_index['all_case_ids']
    if n + n_dev > len(all_case_ids):
        raise ValueError(
            f'n + n_dev = {n + n_dev} > total cases {len(all_case_ids)}'
        )

    rng = np.random.default_rng(seed)

    if stratify_by_cvp:
        manifest_path = os.path.join(data_dir, case_index['manifest'])
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        case_to_bucket = _case_to_dominant_bucket(case_index, manifest)
        # CVP is the 4th modality (last bit of bitmap key).
        cvp_cases = [c for c in all_case_ids if case_to_bucket.get(c, '0000')[-1] == '1']
        non_cvp_cases = [c for c in all_case_ids if c not in set(cvp_cases)]
        cvp_ratio = len(cvp_cases) / len(all_case_ids)

        ho_cvp, ho_non = _stratified_choice(rng, cvp_cases, non_cvp_cases, n)
        chosen = sorted(ho_cvp + ho_non)
        strat_meta = {
            'method': 'cvp_dominant_bucket',
            'cvp_ratio_full': float(cvp_ratio),
            'n_cvp_holdout': len(ho_cvp),
            'n_non_cvp_holdout': len(ho_non),
        }
        # dev: drawn from remainder (disjoint from holdout)
        if n_dev > 0:
            taken = set(chosen)
            rem_cvp = [c for c in cvp_cases if c not in taken]
            rem_non = [c for c in non_cvp_cases if c not in taken]
            dv_cvp, dv_non = _stratified_choice(rng, rem_cvp, rem_non, n_dev)
            dev_chosen = sorted(dv_cvp + dv_non)
            dev_strat_meta = {
                'method': 'cvp_dominant_bucket',
                'cvp_ratio_full': float(cvp_ratio),
                'n_cvp_dev': len(dv_cvp),
                'n_non_cvp_dev': len(dv_non),
            }
        else:
            dev_chosen, dev_strat_meta = [], None
    else:
        chosen = sorted(
            rng.choice(all_case_ids, size=n, replace=False).tolist()
        )
        strat_meta = {'method': 'uniform_random'}
        if n_dev > 0:
            remainder = [c for c in all_case_ids if c not in set(chosen)]
            dev_chosen = sorted(
                rng.choice(remainder, size=n_dev, replace=False).tolist()
            )
            dev_strat_meta = {'method': 'uniform_random'}
        else:
            dev_chosen, dev_strat_meta = [], None

    # Disjoint guard (paranoid: catches any future code change)
    if dev_chosen and (set(chosen) & set(dev_chosen)):
        raise RuntimeError(
            'Internal bug: holdout and dev cohorts overlap — '
            f'{len(set(chosen) & set(dev_chosen))} shared ids'
        )

    out = {
        'version': 2,
        'role': 'holdout_test',
        'n': len(chosen),
        'seed': int(seed),
        'data_dir_at_creation': os.path.abspath(data_dir),
        'stratification': strat_meta,
        'case_ids': chosen,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[sample_holdout] wrote {out_path}')
    print(f'                 n={len(chosen)}  seed={seed}  '
          f'method={strat_meta["method"]}')

    paths = {'holdout': out_path}

    if dev_chosen:
        dev_out = {
            'version': 2,
            'role': 'dev',
            'n': len(dev_chosen),
            'seed': int(seed),
            'data_dir_at_creation': os.path.abspath(data_dir),
            'stratification': dev_strat_meta,
            'disjoint_from': os.path.basename(out_path),
            'case_ids': dev_chosen,
        }
        with open(dev_out_path, 'w') as f:
            json.dump(dev_out, f, indent=2)
        print(f'[sample_holdout] wrote {dev_out_path}')
        print(f'                 n={len(dev_chosen)}  seed={seed}  '
              f'method={dev_strat_meta["method"]}')
        paths['dev'] = dev_out_path

    return paths


if __name__ == '__main__':
    args = get_args()
    sample_holdout(args.data_dir, n=args.n, seed=args.seed,
                   n_dev=args.n_dev,
                   out_name=args.out_name,
                   dev_out_name=args.dev_out_name,
                   stratify_by_cvp=args.stratify_by_cvp,
                   force=args.force)
