# Troubleshooting

Symptoms first, then root causes. Most issues are cohort/path/yaml
mismatches or shape inconsistencies after a modality-set change.

## SSL loaders

### `RuntimeError: dataset MODAL_ORDER ... does not match`

`pretrained/physiome/hetero_data_loader.py::HeteroVitalDBDataset` raises
this when the on-disk manifest's `modal_order` differs from the loader's
`MODAL_ORDER` constant.

**Cause**: manifest written by an earlier 4-modal parser, but the
current loader is 6-modal (or vice versa).

**Fix**: re-run `00_parse_data.sh` against the current `MODAL_ORDER`,
or point `BASE` at a different output directory so v1 and v2 shards
stay separate.

### `FileNotFoundError: case_index.json not found`

The SSL shards exist but the per-shard sidecar wasn't built.

**Fix**: `python -m dataset.data_parser.build_case_index --data_dir <ssl_dir>`
(takes ~100 min on slow NFS).

### `holdout_subjects_file` ignored / no segments excluded

Check the smoke test:
```bash
python experiments/smoke_test_holdout.py
```
If it passes but `02_phase1.sh` shows full segment count, the trainer is
probably loading a config that doesn't propagate `holdout_subjects_file`.
Verify:
```bash
grep -n holdout_subjects_file config/vital_db/dp_neuronet.yaml
```
The CLI arg `--holdout_subjects_file` overrides the yaml.

## Cohorts

### `verify_split_disjoint.py` reports I2 missing

> `[FAIL] I2: every holdout case_id is in downstream_dir -- N missing`

**Cause**: `vital_db_downstream.py` filtered some holdout subjects out
(e.g. `min_duration_sec`). Holdout was sampled from the SSL universe,
not the downstream universe.

**Decision**: either
* re-cut holdout from the downstream-eligible subset, or
* accept the warning if N is small (those subjects simply won't appear
  in downstream test → smaller test set, otherwise fine).

The downstream scripts already print a `[warn] N holdout case_ids not
present` line when this happens; they don't fail.

### `holdout and dev cohorts overlap on N case(s)`

Hard fail in `sample_holdout`. Should be impossible because the script
samples dev from `all - holdout`. If this triggers, file an issue —
something corrupted the JSON between cohort generation and verification.

## Phase-1

### Probe AUROC stays at ~0.5 across epochs

**Symptom**: probe rows in `<ckpt>/<MODAL>/logs/train.log` show
`probe_auroc=0.50xx` for all epochs while TF-C loss is decreasing.

**Diagnosis**: backbone collapse. Common causes:
1. `mask_ratio` too high (default 0.8 may be too aggressive for sparse
   modalities — try 0.6 or 0.5 for CVP/CO2/AWP).
2. Learning rate too high (→ NaN gradients → silent collapse).
3. Effective batch too small for the modality (sparse → few segments).

**Fix**: kill the run, halve `train_base_learning_rate`, and re-train.
Don't wait until Phase-2 to discover this.

### `[warn] modality probe disabled: No dev cases found ...`

Dev cohort doesn't include any subject with the modality the trainer is
probing. Phase-1 SSL continues without the probe (this is intentional —
better to keep training than crash).

**Fix**: increase `--n_dev` or use `--stratify_by_cvp` (and/or stratify
by other modalities) to ensure every modality has dev representation.

### `RuntimeError: shard X is shorter than ...`

A shard file is corrupt or partially written.

**Fix**: identify the shard from the traceback, delete it, re-run
`vital_db_ssl.py` (it will skip cases already in surviving shards via
`writer.add_case` idempotence — actually it won't, sharding is one-shot
so you need to redo from a clean dir).

## Phase-2

### `missing Phase-1 ckpt: <path>`

`scripts/03_phase2.sh` checks every Phase-1 ckpt before launching.
Missing ckpt = Phase-1 didn't finish for that modality (or
`PHASE1_PARALLEL` rounds were interrupted).

**Fix**: re-run the missing modality individually:
```bash
CUDA_VISIBLE_DEVICES=0 python -m pretrained.dp_neuronet.train \
    --config_yaml config/vital_db/dp_neuronet.yaml \
    --ch_idx <missing-idx> ...
```

### CUDA OOM during Phase-2

6-modal sequence is 1.5× longer than 4-modal at the multimodal encoder.

**Fixes (in order)**:
1. Lower `train_batch_size` from 128 → 96 → 64.
2. Lower `decoder_recon_depths` from 8 → 6 (small accuracy cost).
3. Set `dataloader_eager: false` if RAM is the bottleneck (you give up
   epoch-time gain for memory headroom).
4. Reduce `num_workers` (each worker holds shard caches in RAM).

### Probe never improves

If Phase-2 best epoch stays at epoch 0 (i.e. `best_score = 0.0` is never
beaten), it means dev probe macro-F1 doesn't exceed 0 — implausible
unless the dev probe is degenerate (single-class on either side).

**Diagnosis**: check probe sample counts in startup log:
```
[probe] dev samples: train=X eval=Y
```
If X or Y is very small (< 50 each), increase `--n_dev` or check
`min_duration_sec` filter in `load_dev_probe_split`.

## Downstream

### Empty test set

> `RuntimeError: Test set is empty after applying holdout filter.`

**Cause**: `--holdout_subjects_file` doesn't match the cohort that
generated `<data_dir>` (e.g. you re-ran `vital_db_downstream.py` with
different filters but kept the old holdout JSON).

**Fix**: re-run `verify_split_disjoint.py` with the same `--downstream_dir`.
If I2 fails → re-cut cohort or re-parse downstream npz.

### `--max_subsets 0` is too slow

6-modal full enumeration = 63 subsets × forward + LR fit. At 5-10 min per
subset, that's 5-10 hours per task.

**Fix**: use `--max_subsets 15` for the main table; reserve `0` for
supplementary or final-camera-ready completeness check.

### MIMIC mortality SGKF "every fold has same class"

`StratifiedGroupKFold` requires positives in every fold. With small
MIMIC mortality cohorts and `n_splits=5`, this can fail if positives are
unevenly distributed across subjects.

**Fix**: lower `n_splits` to 3, or use `--mode linear_probe` with a
deterministic 80/20 split (and a held-out test cohort).

## Pipeline

### `scripts/00_parse_data.sh` re-runs the SSL parser unnecessarily

Set `SKIP_PARSE=1`:
```bash
SKIP_PARSE=1 bash scripts/run_all.sh
```

### A stage failed, want to resume from there

Skip everything earlier:
```bash
SKIP_PARSE=1 SKIP_COHORT=1 SKIP_PHASE1=1 bash scripts/run_all.sh   # resume from Phase-2
```

### Pipeline log shows mixed-up timestamps across rounds

Phase-1 2-GPU parallel writes per-modality `02_phase1_<MODAL>.log` files
plus interleaves into `pipeline.log`. The timestamps reflect when each
modality *started/finished* — not always strict order. Use
`02_phase1_*.log` per-modality for clean per-run timelines.

## Other

### `case_id` mismatches between SSL and downstream

VitalDB returns `case_id` as a numeric value in some loaders and a
string in others. Our code coerces to `str(case_id)` everywhere
(`hypotension.py`, `_load_subject_ids`, etc.) so comparisons are
namespace-stable. If you see `KeyError: 1234` somewhere, you're missing
a `str()` coercion.

### Adding a new MIMIC channel candidate name

Edit `MIMIC_CHANNEL_CANDIDATES` in `mimic3_waveform_ssl.py` and re-run
`scan` then `parse`. The list is first-match-wins, so put the most
specific name first.

### Logger files are empty

Logger uses `mode='w'` so each run truncates. If you see an empty
`<ckpt>/<MODAL>/logs/train.log`, the trainer crashed before the first
`logger.info` call (typically during `_build_logger` setup or in
`__init__`'s probe loader construction). Check stdout of
`02_phase1_<MODAL>.log` for the traceback.

## When in doubt

```bash
python experiments/smoke_test_holdout.py     # cohort path
python experiments/verify_split_disjoint.py  # leakage path
```

Both run in seconds and pass cleanly when the v2 stack is healthy.
