# Cohort

`sample_holdout.py` is the **single source of truth** for every later
split. One invocation produces two disjoint case-id JSONs that drive
SSL exclusion, dev probing, and downstream evaluation.

## Three groups

| Group | Size (default) | Used by | Forbidden from |
|---|---|---|---|
| **SSL train** | ≈ 6,150 | Phase-1 / Phase-2 SSL training | (none — implicit complement) |
| **dev**       | 50  | per-epoch linear probe (Phase-1 + Phase-2) | SSL training; downstream (optional) |
| **holdout (= test)** | 100 | downstream evaluation | SSL training; dev probing |

All three are **disjoint by case_id**. `sample_holdout.py` enforces it
at write time; `verify_split_disjoint.py` re-checks before training.

## Generation

```bash
python -m dataset.data_parser.sample_holdout \
    --data_dir <ssl_dir> \
    --n 100 --n_dev 50 \
    --seed 777 \
    --stratify_by_cvp
```

Outputs (next to the SSL shards):
* `holdout_case_ids.json`
* `dev_case_ids.json`

### `--stratify_by_cvp`

CVP appears in only ~25% of VitalDB cases. Plain random sampling could
under-represent CVP-bearing subjects in either cohort, biasing all
CVP-related metrics. With `--stratify_by_cvp`:

* both cohorts preserve the population CVP ratio,
* dev sampling further excludes already-chosen holdout ids (paranoid
  runtime check raises if overlap detected).

For 6-modal v2 the script still strata by CVP only — adding more
stratification axes (CO2, AWP) was considered but `holdout_n=100` is
small enough that single-axis stratification is sufficient and
multi-axis would over-constrain the sample.

## Verify before training

```bash
python experiments/verify_split_disjoint.py \
    --holdout         <ssl_dir>/holdout_case_ids.json \
    --dev             <ssl_dir>/dev_case_ids.json \
    --downstream_dir  <downstream_dir> \
    --ssl_dir         <ssl_dir> \
    --mimic_cohort    <icu_mortality_cohort.csv>    # optional
```

Asserts four invariants:

| | Invariant | What it catches |
|---|---|---|
| I1 | holdout ∩ dev = ∅ | Cohort generation bug |
| I2 | every holdout id is in `downstream_dir` | Downstream parsing dropped a holdout subject |
| I2b | dev ⊄ downstream | Dev leaked as a downstream test |
| I3 | holdout ∪ dev ⊂ SSL universe | Cohort references unknown subjects |
| I4 | MIMIC subject_id ∩ VitalDB case_id = ∅ | Cross-dataset namespace collision |

I2b is a soft check (won't fail unless a future change starts saving
dev cases under a downstream filename pattern that holdout doesn't match).

## How each stage consumes the JSONs

### Phase-1 SSL (`pretrained/dp_neuronet/train.py`)

```bash
--holdout_subjects_file holdout_case_ids.json   # excluded from training
--probe_subjects_file   dev_case_ids.json       # used for dev probing
--probe_downstream_dir  <downstream_dir>        # source for probe windows
```

* `ShardSingleModalDataset(exclude_case_ids=...)` filters out holdout
  segments at load time.
* Per-modality probe (see [Probing](Probing.md)) uses the dev cohort.

### Phase-2 SSL (`pretrained/physiome/train_hetero.py`)

```yaml
holdout_subjects_file: <path>     # excluded from training
probe_subjects_file:   <path>     # dev cohort used for probing
probe_downstream_dir:  <path>     # source for probe windows
probe_every: 1                    # cadence
```

* `HeteroVitalDBDataset(exclude_case_ids=...)` filters at load time.
* `_build_probe_loaders` uses dev cohort + `HypotensionDataset` for
  multimodal IOH probe.

### Downstream (`downstream/run_*.py`)

```bash
--holdout_subjects_file holdout_case_ids.json   # = test set
--dev_subjects_file     dev_case_ids.json       # excluded from train (recommended)
```

Implementation (sketched):

```python
all_cases.sort(key=lambda c: str(c['case_id']))
test_cases  = [c for c in all_cases if str(c['case_id']) in holdout_ids]
train_cases = [c for c in all_cases
               if str(c['case_id']) not in holdout_ids
               and str(c['case_id']) not in dev_ids]
```

Empty test set raises immediately so silent misalignment is impossible.
A warning is printed when some holdout ids are absent from
`downstream_dir` (e.g. dropped by `min_duration_sec`).

## External validation note (MIMIC-III)

VitalDB's `case_id` is a string ('case_NNNN' style); MIMIC's
`subject_id` is an int. The two id namespaces never collide regardless
of how splits are drawn inside MIMIC. `verify_split_disjoint.py --mimic_cohort`
confirms this with a string-coerced set intersection.

`run_mortality.py` accepts an optional `--mimic_holdout_subjects_file`
(same JSON schema, but the ids are coerced to int to match MIMIC's
subject_id). When absent, falls back to a sorted 80/20 split inside
MIMIC subjects (deterministic; reproducible per `--seed`).

## File schema (`*_case_ids.json`)

```json
{
  "version": 2,
  "role": "holdout_test",
  "n": 100,
  "seed": 777,
  "data_dir_at_creation": "/abs/path/to/ssl_dir",
  "stratification": {
    "method": "cvp_dominant_bucket",
    "cvp_ratio_full": 0.25,
    "n_cvp_holdout": 25,
    "n_non_cvp_holdout": 75
  },
  "case_ids": ["case_0001", "case_0042", ...]
}
```

`role` is `"holdout_test"` or `"dev"`. `disjoint_from` (only on dev)
records which holdout file the dev cohort was excluded against.

## Re-cutting a cohort

If you need a fresh cohort split (e.g. seed change or larger holdout):

```bash
python -m dataset.data_parser.sample_holdout \
    --data_dir <ssl_dir> \
    --n 150 --n_dev 75 --seed 888 --stratify_by_cvp \
    --force            # required to overwrite existing JSONs
```

Re-run `verify_split_disjoint.py` afterwards. Then re-run **every** later
stage that has already touched these JSONs (Phase-1 ckpts captured the
holdout in their training distribution, so they should be re-trained).
