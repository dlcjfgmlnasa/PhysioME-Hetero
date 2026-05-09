# PhysioME-Hetero v2 — pipeline scripts

End-to-end orchestration for the 6-modal cohort (ABP/ECG/PPG/CVP/CO2/AWP).
Every step is an idempotent bash script that sources `_env.sh` for shared
paths and knobs. Re-run any single stage in isolation, or call `run_all.sh`
to chain them.

## Files

| Script | Stage | Wall clock (KHDP, 2× L40S) |
|---|---|---|
| `_env.sh`            | (sourced) shared env vars + helpers     | — |
| `00_parse_data.sh`   | VitalDB → SSL shards + downstream npz   | ~9 h (1×) |
| `01_make_cohorts.sh` | case_index + holdout/dev cohorts + verify | ~110 min (1×) |
| `02_phase1.sh`       | Phase-1 unimodal SSL × 6                | ~18 h (2-GPU) |
| `03_phase2.sh`       | Phase-2 PhysioME-Hetero multimodal SSL  | ~12-18 h |
| `04_downstream.sh`   | IOH + Hypoxemia + AKI + A2 ablation     | ~30 min × 4 |
| `05_external.sh`     | MIMIC-III mortality (zero-shot + LP)    | ~1 h |
| `run_all.sh`         | Driver — chains 00..05 with skip toggles | ≈ 2-3 days |

## Quick start

```bash
# 1) review and override paths in _env.sh OR via env vars below
export BASE=/home/coder/workspace/updown/physiome_hetero_v2
export VITAL_SRC=/home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files

# 2) run everything
bash scripts/run_all.sh
```

## Skip toggles (env vars)

```bash
SKIP_PARSE=1       bash scripts/run_all.sh   # data already parsed
SKIP_PARSE=1 SKIP_COHORT=1 bash scripts/run_all.sh
SKIP_PHASE1=1      bash scripts/run_all.sh   # Phase-1 ckpts already saved
ONLY_DOWNSTREAM=1  bash scripts/run_all.sh   # = SKIP_PARSE+COHORT+PHASE1+PHASE2
```

Combine freely: `SKIP_PARSE=1 SKIP_COHORT=1 SKIP_PHASE1=1 bash scripts/run_all.sh`
re-runs only Phase-2 → downstream → external.

## Knobs (override in `_env.sh` or via env vars)

| Var | Default | Meaning |
|---|---|---|
| `BASE`            | `/home/coder/workspace/updown/physiome_hetero_v2` | Root for all v2 outputs |
| `VITAL_SRC`       | `…/vitaldb_open/1.0.0/vital_files` | Raw vital files |
| `MIMIC_NPZ_DIR`   | `$BASE/mimic3_hetero` | Step 5 input (set when ready) |
| `MIMIC_COHORT_CSV`| `$BASE/icu_mortality_cohort.csv` | Step 5 + verify I4 |
| `AKI_CLINICAL_CSV`| `$BASE/clinical_data.csv` | Step 4d (auto-skip if missing) |
| `AKI_LAB_CSV`     | `$BASE/lab_data.csv` | Step 4d (auto-skip if missing) |
| `HOLDOUT_N`       | 100 | downstream test cohort size |
| `DEV_N`           | 50  | per-epoch probe cohort size |
| `COHORT_SEED`     | 777 | reproducible cohort split |
| `PHASE1_PARALLEL` | 2   | 1 = sequential, 2 = pair on cuda:0 + cuda:1 |
| `MAX_SUBSETS`     | 15  | downstream subset cap (0 = enumerate all 63) |
| `NUM_WORKERS`     | 16  | DataLoader workers |
| `PREFETCH_FACTOR` | 4   | DataLoader prefetch (Phase-1) |
| `SHARD_CACHE_SIZE`| 2   | LRU cache size in shards |

## Logs

* Per-step stdout: `$LOG_DIR/<NN_step>.log` (e.g. `02_phase1_ABP.log`)
* Pipeline timeline: `$LOG_DIR/pipeline.log`
* Per-modality training metrics:
  * Phase-1: `$CKPT_ROOT/neuronet/<MODAL>/logs/train.log`
  * Phase-2: `$CKPT_ROOT/physiome_hetero/logs/train.log`

## Outputs

```
$BASE/
├── train/              # SSL shards + manifest + case_index + holdout/dev JSONs
├── downstream/         # full-signal npz (vital_db_downstream output)
├── ckpt/
│   ├── neuronet/<M>/model/best_model.pth     # Phase-1, 6 modalities
│   └── physiome_hetero/model/best_model.pth  # Phase-2
├── results/            # downstream + external CSVs (+ preds/ if --save_preds)
└── logs/               # pipeline + per-step stdout
```

## Re-running after a code change

* **Model code changed (loss / arch)**: skip 00 + 01, re-run 02 onward.
* **Phase-2 yaml tweaked**: skip 00 + 01 + 02, re-run 03 onward.
* **Downstream-only change** (e.g. probe subset cap): `ONLY_DOWNSTREAM=1`.
* **New cohort split**: clear holdout/dev JSONs (or bump `COHORT_SEED`) and
  re-run 01 onward — every later stage reads the JSONs at start.
