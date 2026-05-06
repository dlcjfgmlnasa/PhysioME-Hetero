"""End-to-end smoke test for subject-level holdout alignment.

Synthesises a small set of shards via ``ShardWriter``, runs the
``build_case_index`` + ``sample_holdout`` pipeline, then verifies that
both Phase-1 (``ShardSingleModalDataset``) and Phase-2
(``HeteroVitalDBDataset``) loaders honour ``exclude_case_ids`` and
produce reproducible holdouts under fixed seed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch  # noqa: F401  (sets up CUDA env consistent with other smokes)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dataset.data_parser.build_case_index import (  # noqa: E402
    build_case_index,
)
from dataset.data_parser.sample_holdout import sample_holdout  # noqa: E402
from dataset.data_parser.vital_db_ssl import ShardWriter  # noqa: E402
from pretrained.dp_neuronet.hetero_data_loader import (  # noqa: E402
    ShardSingleModalDataset,
    load_holdout_case_ids,
)
from pretrained.physiome.hetero_data_loader import (  # noqa: E402
    HeteroVitalDBDataset,
    MODAL_ORDER,
)


def _synthesise_shards(tmp: str, num_cases: int = 20,
                       segments_per_shard: int = 64) -> None:
    rng = np.random.default_rng(0)
    writer = ShardWriter(tmp, MODAL_ORDER, segments_per_shard=segments_per_shard,
                         compress=True)
    for ci in range(num_cases):
        n_seg = int(rng.integers(8, 16))
        x = rng.standard_normal((n_seg, len(MODAL_ORDER), 6000)).astype(np.float32)
        mask = np.zeros((n_seg, len(MODAL_ORDER)), dtype=bool)
        for s in range(n_seg):
            p = int(rng.integers(1, 1 << len(MODAL_ORDER)))
            for k in range(len(MODAL_ORDER)):
                mask[s, k] = bool((p >> k) & 1)
            if not mask[s].any():
                mask[s, 0] = True
            for k in range(len(MODAL_ORDER)):
                if not mask[s, k]:
                    x[s, k] = 0.0
        writer.add_case(f'case_{ci:03d}', x, mask, mask.any(axis=0))
    writer.close(extra_meta={'sfreq': 100, 'duration': 60})


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _synthesise_shards(tmp, num_cases=20, segments_per_shard=48)

        # 1) build sidecar
        idx_path = build_case_index(tmp, force=True)
        with open(idx_path, 'r') as f:
            ci = json.load(f)
        assert len(ci['all_case_ids']) == 20
        # Per-shard offsets cover all segments contiguously.
        for shard in ci['shards']:
            assert shard['case_offsets'][-1] == shard['n_segments']
            assert len(shard['case_offsets']) == len(shard['case_ids_unique']) + 1
        print(f'[smoke] case_index ok: '
              f'{len(ci["shards"])} shards, '
              f'{len(ci["all_case_ids"])} unique cases')

        # 2) sample holdout (deterministic seed)
        out_a = sample_holdout(tmp, n=5, seed=777,
                               out_name='holdout_a.json', force=True)
        out_b = sample_holdout(tmp, n=5, seed=777,
                               out_name='holdout_b.json', force=True)
        with open(out_a) as f:
            ha = json.load(f)
        with open(out_b) as f:
            hb = json.load(f)
        assert ha['case_ids'] == hb['case_ids'], 'seeded holdout not deterministic'
        # Different seed -> different selection.
        out_c = sample_holdout(tmp, n=5, seed=1234,
                               out_name='holdout_c.json', force=True)
        with open(out_c) as f:
            hc = json.load(f)
        assert ha['case_ids'] != hc['case_ids'], 'different seed gave same selection'
        print(f'[smoke] sample_holdout deterministic / seed-sensitive ok '
              f'(holdout: {ha["case_ids"]})')

        # 3) Phase-1 dataset honours exclusion
        holdout = load_holdout_case_ids(out_a)
        ds_full = ShardSingleModalDataset(tmp, ch_idx=2, normalize=True)
        ds_excl = ShardSingleModalDataset(tmp, ch_idx=2, normalize=True,
                                          exclude_case_ids=holdout)
        assert len(ds_excl) < len(ds_full), \
            f'Phase-1 holdout had no effect ({len(ds_full)} -> {len(ds_excl)})'
        assert ds_excl.n_excluded_segments > 0
        print(f'[smoke] Phase-1 holdout: {len(ds_full)} -> {len(ds_excl)} segments '
              f'({ds_excl.n_excluded_segments} excluded)')

        # 4) Phase-2 dataset honours exclusion
        ds2_full = HeteroVitalDBDataset(tmp, eager=True)
        ds2_excl = HeteroVitalDBDataset(tmp, eager=True,
                                        exclude_case_ids=holdout)
        assert len(ds2_excl) < len(ds2_full), \
            f'Phase-2 holdout had no effect ({len(ds2_full)} -> {len(ds2_excl)})'
        assert ds2_excl.n_excluded_segments > 0
        print(f'[smoke] Phase-2 holdout: {len(ds2_full)} -> {len(ds2_excl)} segments '
              f'({ds2_excl.n_excluded_segments} excluded)')

        # 5) Excluded sets agree across phases (segment count discrepancy
        # only stems from the per-modality mask filter on Phase-1).
        ds_full_p1_full = ShardSingleModalDataset(tmp, ch_idx=2)
        ds_full_p2 = HeteroVitalDBDataset(tmp)
        # Same total cases, same holdout -> same set of excluded *cases*.
        # We do not enforce equal segment counts (Phase-1 is per-modality).
        print(f'[smoke] segment counts: phase1_full={len(ds_full_p1_full)}  '
              f'phase2_full={len(ds_full_p2)}')

        # 6) Missing sidecar -> friendly error when exclusion requested
        bad_dir = tempfile.mkdtemp()
        try:
            _synthesise_shards(bad_dir, num_cases=4, segments_per_shard=24)
            try:
                _ = HeteroVitalDBDataset(bad_dir, exclude_case_ids={'case_001'})
            except FileNotFoundError as e:
                msg = str(e)
                assert 'case_index.json' in msg, msg
                print('[smoke] missing-sidecar error message ok')
            else:
                raise AssertionError('expected FileNotFoundError')
        finally:
            import shutil
            shutil.rmtree(bad_dir, ignore_errors=True)

        print()
        print('[smoke] PASSED -- holdout pipeline intact end-to-end')


if __name__ == '__main__':
    main()
