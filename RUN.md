# PhysioME-Hetero — Run Reference

Centralised CLI cheat sheet. All commands assume CWD = project root.

## 0. Data parsing (one-shot)

### 0.1. Parse VitalDB → sharded SSL data

```bash
python -m dataset.data_parser.vital_db_ssl \
    --src_path  /home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files \
    --trg_path  /home/coder/workspace/updown/physiome_hetero/train \
    --sfreq 100 --duration 60 --nan_max_ratio 0.1 \
    --segments_per_shard 2048
```

Output: `manifest.json` + `shard_NNNN.npz` under `--trg_path`. Wall clock
~9 h on KHDP (6,388 cases → 521 shards / 1.13 M segments).

### 0.2. Parse VitalDB → full-signal downstream npz

Used by IOH / hypoxemia downstream + by per-epoch dev probing in both
Phase-1 and Phase-2 trainers.

```bash
python -m dataset.data_parser.vital_db_downstream \
    --src_path  /home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files \
    --trg_path  /home/coder/workspace/updown/physiome_hetero/downstream \
    --sfreq 100 --require_abp
```

### 0.3. Build the per-shard case index sidecar

```bash
python -m dataset.data_parser.build_case_index \
    --data_dir  /home/coder/workspace/updown/physiome_hetero/train
```

Output: `case_index.json`. Required by `sample_holdout` and the SSL
loaders' `exclude_case_ids` path.

## 1. Define cohorts — holdout (test) + dev (probe)

Single source of truth for every later split:

```bash
python -m dataset.data_parser.sample_holdout \
    --data_dir /home/coder/workspace/updown/physiome_hetero/train \
    --n 100 --n_dev 50 --seed 777 --stratify_by_cvp
```

Outputs (next to the SSL shards):
* `holdout_case_ids.json` — downstream test cohort. Excluded from SSL.
* `dev_case_ids.json`     — probe cohort (Phase-1/Phase-2 monitoring).
                            Disjoint from holdout by construction.

## 2. Verify split integrity (one-shot)

```bash
python experiments/verify_split_disjoint.py \
    --holdout        /home/coder/workspace/updown/physiome_hetero/train/holdout_case_ids.json \
    --dev            /home/coder/workspace/updown/physiome_hetero/train/dev_case_ids.json \
    --downstream_dir /home/coder/workspace/updown/physiome_hetero/downstream \
    --ssl_dir        /home/coder/workspace/updown/physiome_hetero/train \
    --mimic_cohort   /home/coder/workspace/updown/physiome_hetero/data/icu_mortality_cohort.csv
```

I1: holdout & dev disjoint — I2: every holdout id present downstream —
I3: holdout∪dev ⊆ SSL universe — I4: MIMIC subject_id ↔ VitalDB case_id
namespaces disjoint. All four must pass before training.

## 3. Phase-1 NeuroNet (per-modality SSL)

### 3.1. Sequential sweep over the 4 modalities

```bash
HOLDOUT=/home/coder/workspace/updown/physiome_hetero/train/holdout_case_ids.json
DEV=/home/coder/workspace/updown/physiome_hetero/train/dev_case_ids.json
DSDIR=/home/coder/workspace/updown/physiome_hetero/downstream

for IDX in 0 1 2 3; do
    python -m pretrained.dp_neuronet.train \
        --config_yaml config/vital_db/dp_neuronet.yaml \
        --ch_idx $IDX \
        --num_workers 16 --prefetch_factor 4 \
        --holdout_subjects_file $HOLDOUT \
        --probe_downstream_dir  $DSDIR \
        --probe_subjects_file   $DEV \
        --probe_every 1 \
        > log_phase1_${IDX}.txt 2>&1
done
```

Per-epoch probe AUROC + macro-F1 land in
`<ckpt_path>/neuronet/<MODAL>/logs/train.log`. Best epoch is still chosen
by held-out TF-C loss; probing is for monitoring only. CVP (`ch_idx=3`)
auto-skips probing (vital_db_downstream.py does not emit CVP windows).

### 3.2. Two GPUs in parallel

```bash
CUDA_VISIBLE_DEVICES=0 python -m pretrained.dp_neuronet.train \
    --config_yaml config/vital_db/dp_neuronet.yaml --ch_idx 0 \
    --holdout_subjects_file $HOLDOUT \
    --probe_downstream_dir  $DSDIR --probe_subjects_file $DEV \
    > log_phase1_ABP.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m pretrained.dp_neuronet.train \
    --config_yaml config/vital_db/dp_neuronet.yaml --ch_idx 1 \
    --holdout_subjects_file $HOLDOUT \
    --probe_downstream_dir  $DSDIR --probe_subjects_file $DEV \
    > log_phase1_ECG.txt 2>&1 &

wait
# then PPG (ch_idx 2) + CVP (ch_idx 3)
```

Output ckpts: `<ckpt_path>/neuronet/<MODAL>/model/best_model.pth`.

## 4. Phase-2 PhysioME-Hetero

Set `holdout_subjects_file`, `probe_subjects_file`, `probe_downstream_dir`
in `config/vital_db/physiome_hetero.yaml` then run:

```bash
python -m pretrained.physiome.train_hetero \
    --config_yaml config/vital_db/physiome_hetero.yaml
```

Per-epoch dev-cohort linear probing (ABP/ECG/PPG subsets only; CVP slot
filled by `inference_missing_modality`). Best epoch = highest mean
macro-F1 on dev probe. Logs at `<ckpt_path>/logs/train.log`.

### 4.1. A1 baseline (synthetic-only PhysioME)

```bash
python -m pretrained.physiome.train \
    --config_yaml config/vital_db/physiome.yaml
```

### 4.2. A3 ablation (`restoration_only_on_complete: false`)

```bash
python -m pretrained.physiome.train_hetero \
    --config_yaml config/vital_db/physiome_hetero_a3.yaml
```

## 5. Downstream — holdout cohort = test

```bash
HOLDOUT=/home/coder/workspace/updown/physiome_hetero/train/holdout_case_ids.json
DEV=/home/coder/workspace/updown/physiome_hetero/train/dev_case_ids.json
CKPT=/home/coder/workspace/updown/physiome_hetero/ckpt/physiome_hetero/model/best_model.pth
DSDIR=/home/coder/workspace/updown/physiome_hetero/downstream

# IOH
python -m downstream.run_ioh \
    --ckpt_path $CKPT --data_dir $DSDIR \
    --holdout_subjects_file $HOLDOUT --dev_subjects_file $DEV

# Hypoxemia
python -m downstream.run_hypoxemia \
    --ckpt_path $CKPT --data_dir $DSDIR \
    --holdout_subjects_file $HOLDOUT --dev_subjects_file $DEV

# AKI (KDIGO Cr-based)
python -m downstream.run_aki --ckpt_path $CKPT
```

`--dev_subjects_file` is optional but recommended — it excludes the dev
cohort from downstream *train* too, eliminating the (mild) leak channel
where probe-set subjects influence downstream training.

## 6. External validation — VitalDB → MIMIC-III

Cross-dataset transfer; `case_id (str)` vs `subject_id (int)` namespaces
are disjoint by construction. I4 of `verify_split_disjoint.py` confirms.

```bash
# zero-shot (5-fold subject CV inside MIMIC; encoder never sees MIMIC)
python -m downstream.run_mortality \
    --ckpt_path $CKPT \
    --mimic_npz_dir <mimic_npz_dir> \
    --cohort_csv    <icu_mortality_cohort.csv> \
    --mode zero_shot --seed 42

# linear probing inside MIMIC (optional reproducible test cohort)
python -m downstream.run_mortality \
    --ckpt_path $CKPT \
    --mimic_npz_dir <mimic_npz_dir> \
    --cohort_csv    <icu_mortality_cohort.csv> \
    --mode linear_probe --seed 42 \
    [--mimic_holdout_subjects_file <mimic_holdout.json>]
```

## 7. Smoke tests (no real data, ~1 min each)

```bash
python experiments/smoke_test_dp_neuronet.py    # Phase-1 TF-C
python experiments/smoke_test_hetero.py         # Phase-2 hetero
python experiments/smoke_test_downstream.py     # frozen-encoder eval
python experiments/smoke_test_probe.py          # subset enumeration
python experiments/smoke_test_holdout.py        # subject-level holdout + dev
```

## 8. Common overrides (all train scripts)

| Flag | Purpose |
|---|---|
| `--ssl_data_dir <path>` | Override yaml SSL data dir |
| `--ckpt_path <path>` | Override yaml ckpt save dir |
| `--num_workers <int>` | DataLoader workers |
| `--prefetch_factor <int>` | DataLoader prefetch (Phase-1 only) |
| `--shard_cache_size <int>` | LRU cache size in shards |
| `--holdout_subjects_file <json>` | Subject-level exclusion list |
| `--probe_subjects_file <json>` | Dev cohort for periodic LR probing |
| `--probe_downstream_dir <dir>` | vitaldb_downstream npz dir for probing |
| `--probe_every <int>` | Probe cadence (default every epoch) |
| `--ch_idx <0..3>` | Phase-1 only: which modality to train |
