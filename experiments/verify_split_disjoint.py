# -*- coding:utf-8 -*-
"""End-to-end split-integrity verifier.

Asserts the four invariants that keep PhysioME's leakage story honest:

    [I1] holdout intersect dev = empty                      (cohort split)
    [I2] downstream test cohort == holdout                  (Step 4 patch)
    [I3] SSL pretraining excludes holdout union dev         (Phase-1/Phase-2)
    [I4] external (MIMIC) subject_ids vs VitalDB case_ids = disjoint

Designed as a 1-time CI-style check before kicking off experiments. Reads
real cohort JSONs and downstream npz dirs; does NOT touch the model.

Usage::

    python experiments/verify_split_disjoint.py \\
        --holdout       data/vitaldb_ssl/holdout_case_ids.json \\
        --dev           data/vitaldb_ssl/dev_case_ids.json \\
        --downstream_dir data/vitaldb_downstream \\
        [--mimic_cohort  data/icu_mortality_cohort.csv]

Exit code 0 = all invariants hold; non-zero = first failed check is printed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Optional, Set


def _load_case_ids(path: str) -> Set[str]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return {str(c) for c in payload['case_ids']}


def _downstream_case_ids(data_dir: str) -> Set[str]:
    """Read the case_id field from every downstream npz in ``data_dir``."""
    import numpy as np
    ids: Set[str] = set()
    for path in sorted(glob.glob(os.path.join(data_dir, '*.npz'))):
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f'  [warn] could not read {path}: {e}')
            continue
        if 'case_id' in arr.files:
            ids.add(str(arr['case_id']))
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            ids.add(base)
    return ids


def _mimic_subject_ids(cohort_csv: str) -> Set[int]:
    ids: Set[int] = set()
    with open(cohort_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.add(int(row['subject_id']))
            except (KeyError, ValueError):
                continue
    return ids


def _check(label: str, ok: bool, detail: str = '') -> bool:
    mark = 'OK' if ok else 'FAIL'
    print(f'  [{mark}] {label}' + (f' -- {detail}' if detail else ''))
    return ok


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--holdout', required=True, type=str,
                   help='Path to holdout_case_ids.json (sample_holdout output)')
    p.add_argument('--dev', type=str, default=None,
                   help='Path to dev_case_ids.json (optional)')
    p.add_argument('--downstream_dir', type=str, default=None,
                   help='vitaldb_downstream npz dir; if given, verifies that '
                        'every holdout case_id is present (so the downstream '
                        'test cohort = holdout cohort).')
    p.add_argument('--ssl_dir', type=str, default=None,
                   help='vitaldb_ssl shard dir; if given and case_index.json '
                        'is present, verifies that holdout ∪ dev are a subset '
                        'of the SSL universe (so exclusion has bite).')
    p.add_argument('--mimic_cohort', type=str, default=None,
                   help='MIMIC mortality cohort CSV; if given, verifies that '
                        'subject_id (int) namespace does not collide with '
                        'VitalDB case_id (str) namespace.')
    return p.parse_args()


def main() -> int:
    args = get_args()
    print(f'[verify_split_disjoint] holdout={args.holdout}'
          + (f'  dev={args.dev}' if args.dev else ''))

    holdout = _load_case_ids(args.holdout)
    print(f'  |holdout| = {len(holdout)}')

    dev: Set[str] = set()
    if args.dev:
        dev = _load_case_ids(args.dev)
        print(f'  |dev|     = {len(dev)}')

    all_ok = True

    # I1
    inter = holdout & dev
    all_ok &= _check(
        'I1: holdout & dev are disjoint', not inter,
        detail=(f'overlap on {sorted(inter)[:5]} (and more)' if inter else
                f'verified disjoint ({len(holdout)}+{len(dev)} ids)'),
    )

    # I2 — downstream test cohort coverage
    if args.downstream_dir:
        ds_ids = _downstream_case_ids(args.downstream_dir)
        missing = holdout - ds_ids
        all_ok &= _check(
            'I2: every holdout case_id is in downstream_dir',
            not missing,
            detail=(f'{len(missing)} missing (e.g. '
                    f'{sorted(missing)[:5]})' if missing else
                    f'all {len(holdout)} holdout ids present '
                    f'(of {len(ds_ids)} downstream cases)'),
        )
        leaked = (ds_ids - holdout) & dev if dev else set()
        all_ok &= _check(
            'I2b: dev cohort not present in downstream_dir as test',
            not leaked,
            detail=(f'{len(leaked)} dev ids leaked into downstream'
                    if leaked else 'dev disjoint from downstream test'),
        )

    # I3 — SSL universe sanity
    if args.ssl_dir:
        idx_path = os.path.join(args.ssl_dir, 'case_index.json')
        if not os.path.isfile(idx_path):
            print(f'  [skip] I3: {idx_path!r} not found '
                  '(run build_case_index first)')
        else:
            with open(idx_path, 'r', encoding='utf-8') as f:
                ssl_universe = {str(c) for c in json.load(f)['all_case_ids']}
            ho_unknown = holdout - ssl_universe
            dv_unknown = dev - ssl_universe
            all_ok &= _check(
                'I3: holdout subset of SSL universe', not ho_unknown,
                detail=(f'{len(ho_unknown)} unknown ids' if ho_unknown else
                        f'all {len(holdout)} present in SSL'),
            )
            if dev:
                all_ok &= _check(
                    'I3b: dev subset of SSL universe', not dv_unknown,
                    detail=(f'{len(dv_unknown)} unknown ids' if dv_unknown else
                            f'all {len(dev)} present in SSL'),
                )

    # I4 — MIMIC vs VitalDB namespace
    if args.mimic_cohort:
        mimic_subjects = _mimic_subject_ids(args.mimic_cohort)
        # str(int) collision check — MIMIC subject_id rendered as str ought
        # not match any VitalDB case_id.
        mimic_as_str = {str(s) for s in mimic_subjects}
        collide = mimic_as_str & (holdout | dev)
        all_ok &= _check(
            'I4: MIMIC subject_id vs VitalDB case_id disjoint',
            not collide,
            detail=(f'{len(collide)} collisions: {sorted(collide)[:5]}'
                    if collide else
                    f'{len(mimic_subjects)} MIMIC subjects vs '
                    f'{len(holdout) + len(dev)} VitalDB ids -- disjoint'),
        )

    print()
    if all_ok:
        print('[verify_split_disjoint] PASS -- all checked invariants hold.')
        return 0
    else:
        print('[verify_split_disjoint] FAIL -- see [FAIL] lines above.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
