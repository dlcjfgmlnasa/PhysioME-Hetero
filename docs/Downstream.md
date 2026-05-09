# Downstream

Frozen-encoder linear-probe evaluation on the holdout (=test) cohort.
The encoder is loaded from a Phase-2 ckpt; per modal-subset latents are
extracted via `inference_missing_modality` (so absent modalities are
filled by `dropped_modality_token`), then a `LogisticRegression` head is
fit per subset and evaluated on the test set.

## Tasks

| Task | Script | Label | Lookahead | Cohort | Notes |
|---|---|---|---|---|---|
| **IOH** | `run_ioh.py` | sustained MAP < 65 mmHg ≥ 60 s | 5 min | VitalDB | primary outcome |
| **Hypoxemia** | `run_hypoxemia.py` | sustained SpO2 < 92% ≥ 30 s | 5 min | VitalDB | SpO2 channel = label-only (NOT in MODAL_ORDER) |
| **AKI** | `run_aki.py` | KDIGO Cr-based stage ≥ 1 | postop 7 days | VitalDB | needs `clinical_data.csv` + `lab_data.csv` |
| **A2 ablation** | `run_ablation_a2.py` | IOH (per-subset) | 5 min | VitalDB | real-missing vs synth-missing OOD gap |
| **Mortality** | `run_mortality.py` | hospital_expire_flag | — | MIMIC-III | external (cross-dataset) |

## Common protocol (VitalDB tasks)

```
1. Load full-signal npz from data_dir (vital_db_downstream output).
2. Subject-level split:
     test  = case_ids in --holdout_subjects_file
     train = remaining (- --dev_subjects_file if given)
3. Slide 60 s windows with 60 s stride over each case.
4. Generate task-specific labels via the lookahead horizon.
5. For each modal subset:
     extract latents through frozen encoder
     fit StandardScaler + LogisticRegression(class_weight='balanced')
     evaluate on test windows
     report AUROC + AUPRC + Sens@Sp90%
6. Print and save results/<tag>_<task>.csv.
   --save_preds also dumps (y_true, y_score) npy for calibration analysis.
```

## Subset cap (`--max_subsets`)

At N=6 modalities, full enumeration = 2^6 - 1 = 63 subsets. Each subset
costs an encoder forward + LR fit. Recommended cap:

| `--max_subsets` | Subsets evaluated | Use case |
|---|---|---|
| 0 (default) | all 63 | supplementary / completeness |
| 15 | full + each single + ~8 random | main paper table |
| 7 | full + each single | quick sanity check |

Implementation: `pretrained.physiome.probe_utils.select_probe_subsets` —
always includes the full set + every single-modal subset, then
deterministically random-samples remaining subsets up to the cap.

## Per-task knobs (most-used)

### `run_ioh.py`

```bash
--window_sec 60 --stride_sec 60 --horizon_sec 300
--map_threshold 65.0 --sustained_sec 60.0
--max_subsets 15
--save_preds --tag hetero
```

### `run_hypoxemia.py`

```bash
--window_sec 60 --stride_sec 60 --horizon_sec 300
--spo2_threshold 92.0 --sustained_sec 30.0
--max_subsets 15
```

### `run_aki.py`

```bash
--clinical_csv <path> --lab_csv <path>
--window_sec 600 --stride_sec 300        # AKI uses longer windows
--max_subsets 15
```

KDIGO stages computed from preop_cr (clinical CSV) + postop creatinine
trajectory (lab CSV); per the VitalDB official `mbp_aki` / `xgb_aki`
example logic. AKI is binarised as stage ≥ 1.

### `run_ablation_a2.py`

```bash
--window_sec 60 --stride_sec 60 --horizon_sec 300
--max_subsets 15
```

A2 splits the test set into:
* **synth-missing**: complete-modality cases where one modality is masked
  at inference (= our training distribution).
* **real-missing**: cases where one modality was naturally absent in the
  recording (= the OOD condition our model is built for).

The "OOD gap" = AUROC(synth-missing) − AUROC(real-missing). Hetero models
should narrow this gap vs synthetic-only baselines (A1).

### `run_mortality.py`

```bash
--mimic_npz_dir <dir> --cohort_csv <csv>
--mode {zero_shot, linear_probe}
--seed 42
[--mimic_holdout_subjects_file <json>]   # linear_probe only
```

* `zero_shot`: 5-fold subject-level CV inside MIMIC, encoder never sees
  MIMIC during training. **Important**: this is *frozen-encoder LR over
  CV folds*, not a true zero-shot classifier. The "zero-shot" label
  refers to the encoder, not the LR head.
* `linear_probe`: subject-level 80/20 (sorted) or external
  `--mimic_holdout_subjects_file`.

## External validation narrative

VitalDB pretrain (6-modal) → MIMIC inference. MIMIC records typically
expose only ABP/ECG/PPG (sometimes CVP). CO2/AWP almost never present.
This is exactly the *"trained-on-N, infer-on-M<N"* hetero scenario:

```
   VitalDB train          MIMIC test
   -------------          ----------
   ABP ✓                  ABP ✓
   ECG ✓                  ECG ✓
   PPG ✓                  PPG ✓
   CVP ✓ (~25%)           CVP ? (record-dependent)
   CO2 ✓ (~80%)           CO2 ✗ (naturally absent)
   AWP ✓ (~50%)           AWP ✗ (naturally absent)
```

Empty slots are filled by the trained `dropped_modality_token`. Cross-
dataset disjoint by namespace (case_id str vs subject_id int) — verified
by `verify_split_disjoint.py I4`.

## Output files

```
$RESULTS/
├── hetero_ioh.csv                # one row per modal subset
├── hetero_hypoxemia.csv
├── hetero_aki.csv
├── hetero_a2_synth.csv           # A2: synth-missing rows
├── hetero_a2_real.csv            # A2: real-missing rows
├── hetero_a2_gap.csv             # A2: gap = synth - real
├── hetero_mortality.csv          # zero_shot folds averaged
├── hetero_lp_mortality.csv       # linear_probe rows
└── preds/
    ├── hetero_ioh_<subset>_ytrue.npy
    └── hetero_ioh_<subset>_yscore.npy
```

CSV header: `Subset,AUROC,AUPRC,Sens@Sp90`. Subset = `+`-joined modality
names (e.g. `ABP+ECG+PPG`).

## Calibration

`downstream/calibration.py` reads saved `(y_true, y_score)` npys and
plots:
* reliability diagram (predicted vs observed positive rate)
* expected calibration error (ECE) on 10 quantile bins
* Brier score

Recommended for IOH and Hypoxemia (class-imbalanced binary tasks where
AUROC alone overstates clinical usability).

## Reproducibility

* All splits are deterministic given `--seed` and the cohort JSONs.
* `LogisticRegression(random_state=42, max_iter=1000)`.
* `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` in
  `run_mortality zero_shot`.
* Per-task CSV outputs are byte-identical across runs at fixed seed.
