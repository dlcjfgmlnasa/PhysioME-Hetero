# Pipeline

End-to-end shell driver under `scripts/`. See `scripts/README.md` for the
shortest reference; this page explains *what* and *why* of each stage.

## Stages

```
00_parse_data    →  01_make_cohorts   →  02_phase1   →  03_phase2   →  04_downstream  →  05_external
   (~9h, 1×)         (~110min, 1×)         (~18h)         (~12-18h)       (~2h)            (~1h)
```

### `00_parse_data.sh`

Parses raw VitalDB into:
* sharded SSL output (`vital_db_ssl.py` → `manifest.json` + `shard_NNNN.npz`)
* full-signal downstream npz (`vital_db_downstream.py`, one npz per case)

Both parsers share `MODAL_TRACK_NAMES` (`vital_db_ssl.py`) and
`SIGNAL_CONFIGS` (`_signal_filters.py`) so adding a modality is one place.

Re-run only when the modality set changes or the source vital files change.

### `01_make_cohorts.sh`

Three sub-steps:

1. **`build_case_index`** — per-shard sidecar mapping segments to
   case_ids. Required for case-level holdout (the SSL shards themselves
   are not case-aligned).
2. **`sample_holdout --n 100 --n_dev 50`** — emits two disjoint cohort
   JSONs (see [Cohort](Cohort.md)).
3. **`verify_split_disjoint`** — asserts the four leakage invariants.
   Fail = stop (cohort is broken; do not proceed).

### `02_phase1.sh`

Six unimodal Phase-1 SSL runs. With `PHASE1_PARALLEL=2` (default), pairs
two modalities on `cuda:0` + `cuda:1` per round → 3 rounds total. With
`PHASE1_PARALLEL=1`, sequential.

Per modality the trainer:
* loads SSL shards filtered by `--ch_idx`,
* excludes `holdout_subjects_file` case_ids from training,
* enables per-modality dev-cohort linear probe (see [Probing](Probing.md)),
* writes step-level loss + per-epoch val + per-epoch probe AUROC to
  `<ckpt_root>/neuronet/<MODAL>/logs/train.log`,
* saves the best epoch by held-out TF-C loss to
  `<ckpt_root>/neuronet/<MODAL>/model/best_model.pth`.

### `03_phase2.sh`

PhysioME-Hetero multimodal SSL. The script writes a yaml override to
`$LOG_DIR/physiome_hetero.run.yaml` so the env-driven paths (SSL dir,
ckpt root, holdout/dev JSONs, probe downstream dir) are propagated
without modifying the committed config. The trainer:
* requires every Phase-1 ckpt to exist (script checks first),
* runs `BucketBatchSampler` for batch-uniform availability,
* per-epoch dev-cohort multimodal probe over ABP/ECG/PPG subsets
  (sparse modalities filled by `inference_missing_modality`),
* saves the best epoch by mean macro-F1 to
  `<ckpt_root>/physiome_hetero/model/best_model.pth`.

### `04_downstream.sh`

Frozen-encoder linear-probe evaluation on the holdout (=test) cohort:
* `run_ioh.py` — IOH (sustained MAP < 65 mmHg in 5-min lookahead).
* `run_hypoxemia.py` — sustained SpO2 < 92% for ≥30 s in 5-min lookahead.
* `run_ablation_a2.py` — A2 ablation (real-missing vs synth-missing OOD gap).
* `run_aki.py` — KDIGO Cr-based AKI (skipped automatically if
  `clinical_data.csv` / `lab_data.csv` are absent).

Every script gained `--max_subsets <int>` (default 0 = enumerate all
2^N-1; recommended 15 to bound the 6-modal cost via `select_probe_subsets`).

### `05_external.sh`

VitalDB-pretrained encoder → MIMIC-III WDB mortality.
* `zero_shot` mode: 5-fold subject-level CV inside MIMIC; encoder never
  retrained. `random_state=seed` for reproducibility.
* `linear_probe` mode: sorted 80/20 inside MIMIC, or
  `--mimic_holdout_subjects_file` for an external test cohort.

Auto-skips if `MIMIC_NPZ_DIR` or `MIMIC_COHORT_CSV` are absent.

### `run_all.sh`

Master driver. Honours skip toggles:

```bash
SKIP_PARSE=1 SKIP_COHORT=1 bash scripts/run_all.sh   # learn from existing data/cohorts
SKIP_PHASE1=1 bash scripts/run_all.sh                # Phase-1 ckpts already saved
ONLY_DOWNSTREAM=1 bash scripts/run_all.sh            # 04 + 05 only
```

## Configuration

Override paths/seeds/GPU/cohort knobs from the calling shell. See
`scripts/_env.sh` for the exhaustive list. Default `BASE` =
`/home/coder/workspace/updown/physiome_hetero_v2` (KHDP layout). Common
overrides:

```bash
BASE=/data/physiome_v2 \
PHASE1_PARALLEL=1 \
MAX_SUBSETS=0 \
NUM_WORKERS=8 \
bash scripts/run_all.sh
```

## Wall-clock estimates (KHDP, 2× L40S)

| Stage | Time |
|---|---|
| 00 parse_data | ~9h (1×) |
| 01 make_cohorts | ~110 min (1×) |
| 02 phase1 (6 modal, 2-GPU) | ~18 h |
| 03 phase2 | ~12-18 h |
| 04 downstream (4 tasks) | ~30 min × 4 |
| 05 external | ~1 h |
| **Total** | **~2-3 days** |

Phase-1 is the dominant cost. `02_phase1.sh` writes per-modality logs in
parallel, so monitoring is straightforward (`tail -f log_phase1_*.log`).

## When to re-run what

| Code/data change | Stages to re-run |
|---|---|
| `_signal_filters.py` (preprocessing) | 00, 01, 02, 03, 04 |
| Modality added to `MODAL_TRACK_NAMES` | 00, 01, 02, 03, 04 |
| Cohort decision (n / n_dev / seed) | 01 onward |
| Phase-1 hparam (mask_ratio, lr) | 02 onward |
| Phase-2 yaml (encoder/decoder hparam) | 03, 04 |
| Downstream-only (subset cap, threshold) | 04 (or 05 for MIMIC) |
| Plot/calibration reanalysis | none — runs on saved preds |
