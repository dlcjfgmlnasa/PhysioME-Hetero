# -*- coding:utf-8 -*-
"""Report the actual bucket-availability distribution of the v2 SSL cohort.

Reads only ``manifest.json`` (no shard payload, no GPU, runs in seconds even
on slow NFS) and prints:

  * per-bucket segment count + percentage, sorted by frequency,
  * complete-bucket percentage (= the slice v1 would train on),
  * dominant-bucket(s) percentage (= the slice v1 would never see at correct
    frequency under uniform synthetic drop),
  * v1 vs v2 training cohort size comparison.

Usage:
    python -m experiments.bucket_distribution \
        --data_dir /home/coder/workspace/updown/physiome_hetero_v2/train

Optionally exclude holdout / dev cohorts (matches SSL training reality):
    python -m experiments.bucket_distribution \
        --data_dir <...> \
        --holdout <ssl_dir>/holdout_case_ids.json \
        --dev     <ssl_dir>/dev_case_ids.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Optional, Set


MODAL_ORDER = ['ABP', 'ECG', 'PPG', 'CVP', 'CO2', 'AWP']


def _load_case_ids(path: Optional[str]) -> Set[str]:
    if path is None:
        return set()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return {str(c) for c in payload['case_ids']}


def main(args: argparse.Namespace) -> None:
    manifest_path = os.path.join(args.data_dir, 'manifest.json')
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    shards = manifest['shards']

    excluded = _load_case_ids(args.holdout) | _load_case_ids(args.dev)

    # case_index.json schema (CSR-style, one entry per SHARD):
    #   shards[si] = {
    #       'path': ...,
    #       'n_segments': N_si,
    #       'case_ids_unique': [c0, c1, ..., c_{K-1}],
    #       'case_offsets':    [0, off1, off2, ..., N_si],
    #   }
    # segment local-index ``li`` belongs to case_ids_unique[k] where
    # case_offsets[k] <= li < case_offsets[k+1].
    case_index_path = os.path.join(args.data_dir, 'case_index.json')
    case_index_shards = None
    if excluded:
        if not os.path.isfile(case_index_path):
            raise FileNotFoundError(
                f'--holdout/--dev given but {case_index_path} missing. '
                f'Run: python -m dataset.data_parser.build_case_index '
                f'--data_dir {args.data_dir!r}'
            )
        with open(case_index_path, 'r', encoding='utf-8') as f:
            case_index_shards = json.load(f)['shards']

    def _build_segment_case_lookup(idx_shard: dict, n_segs: int):
        """Return a list ``cids`` of length ``n_segs`` where ``cids[li]`` is
        the case_id of the li-th segment in the shard."""
        uniques = idx_shard['case_ids_unique']
        offsets = idx_shard['case_offsets']
        out = [''] * n_segs
        for k, cid in enumerate(uniques):
            lo, hi = offsets[k], offsets[k + 1]
            for li in range(lo, hi):
                out[li] = str(cid)
        return out

    bucket_counter: Counter = Counter()
    n_total = 0
    n_excluded = 0
    for si, shard in enumerate(shards):
        keys = shard['bitmap_keys']
        seg_cids = None
        if case_index_shards is not None:
            seg_cids = _build_segment_case_lookup(
                case_index_shards[si], len(keys),
            )
        for li, key in enumerate(keys):
            if seg_cids is not None:
                if seg_cids[li] in excluded:
                    n_excluded += 1
                    continue
            bucket_counter[key] += 1
            n_total += 1

    print(f'\nData dir         : {args.data_dir}')
    print(f'Manifest shards  : {len(shards)}')
    if excluded:
        print(f'Excluded segs    : {n_excluded}  '
              f'(holdout/dev cohort, mirrors SSL training)')
    print(f'Total segments   : {n_total}')
    print(f'Active buckets   : {len(bucket_counter)}\n')

    print('  bucket   ', end='')
    for m in MODAL_ORDER:
        print(f'{m:>5}', end='')
    print('     segments      %  cum%')
    print('  ' + '-' * (10 + 5 * len(MODAL_ORDER) + 28))

    cum = 0.0
    complete_key = '1' * len(MODAL_ORDER)
    complete_pct = 0.0
    rows = bucket_counter.most_common()
    for key, cnt in rows:
        pct = 100.0 * cnt / max(1, n_total)
        cum += pct
        if key == complete_key:
            complete_pct = pct
        present = [('  X  ' if c == '1' else '  .  ') for c in key]
        flag = '  <- complete' if key == complete_key else ''
        print(f'  {key}   ', end='')
        for cell in present:
            print(cell, end='')
        print(f'  {cnt:>9d}  {pct:5.2f}  {cum:5.2f}{flag}')

    print()
    print('=== Summary ===')
    print(f'  Complete bucket ({complete_key}) : {complete_pct:5.2f}%')
    print(f'  Non-complete (v1 throws away)  : {100 - complete_pct:5.2f}%')

    top1_key, top1_cnt = rows[0]
    top1_pct = 100.0 * top1_cnt / max(1, n_total)
    print(f'  Dominant bucket  : {top1_key}  '
          f'({top1_pct:5.2f}% of cohort)')
    if top1_key == complete_key:
        print('    -> complete IS the mode (rare). v1 trains on the majority.')
    else:
        print('    -> v1 trains on a MINORITY phenotype; dominant phenotype '
              'enters training only via uniform synthetic drop.')

    # Synthetic drop coverage estimate for the dominant bucket.
    n_real_present = top1_key.count('1')
    drop_prob = 0.2
    # P("a complete-modal case gets exactly this pattern after synth drop") =
    # binomial term: each missing slot in top1 must be dropped (prob=drop_prob);
    # each real slot in top1 must NOT be dropped (prob=1-drop_prob).
    n_missing = len(MODAL_ORDER) - n_real_present
    p_synth = (drop_prob ** n_missing) * ((1 - drop_prob) ** n_real_present)
    print(f'  P(synth drop reproduces {top1_key}) | drop_prob={drop_prob} '
          f'= {100 * p_synth:5.2f}%')
    if p_synth > 0:
        ratio = top1_pct / 100.0 / p_synth
        print(f'  Real / synth frequency ratio for dominant bucket : '
              f'{ratio:5.2f}x  (>1 = v1 undertrains; <1 = v1 overtrains)')

    print()
    print('=== v1 vs v2 training cohort size ===')
    v1_segs = bucket_counter.get(complete_key, 0)
    print(f'  v1 (complete-only)  : {v1_segs:>9d} segments  '
          f'({100*v1_segs/max(1,n_total):5.2f}% of cohort)')
    print(f'  v2 (hetero, all)    : {n_total:>9d} segments  (100.00%)')
    if v1_segs > 0:
        print(f'  v2 / v1 ratio       : {n_total / v1_segs:5.2f}x more '
              f'training data for v2')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='SSL shard dir (contains manifest.json + '
                             'shard_*.npz + optional case_index.json)')
    parser.add_argument('--holdout', default=None,
                        help='holdout_case_ids.json (exclude from count to '
                             'match SSL training reality)')
    parser.add_argument('--dev', default=None,
                        help='dev_case_ids.json (likewise)')
    main(parser.parse_args())
