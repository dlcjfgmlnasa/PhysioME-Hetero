# PhysioME-Hetero — Run Reference

Centralised CLI cheat sheet. All commands assume CWD = project root.

## 1. Data preparation

### 1.1. Parse VitalDB → sharded SSL data

```bash
python -m dataset.data_parser.vital_db_ssl \
    --src_path  /home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files \
    --trg_path  /home/coder/workspace/updown/physiome_hetero/train \
    --sfreq 100 --duration 60 --nan_max_ratio 0.1 \
    --segments_per_shard 2048
```

Output: `manifest.json` + `shard_NNNN.npz` under `--trg_path`. Wall clock
~9 h on KHDP (6,388 cases → 521 shards / 1.13 M segments).

### 1.2. Build the per-shard case index sidecar (required for holdout)

```bash
python -m dataset.data_parser.build_case_index \
    --data_dir  /home/coder/workspace/updown/physiome_hetero/train
```

Output: `case_index.json` next to `manifest.json`. ~100 min one-shot
(reads each shard once on the network drive).

### 1.3. Sample a held-out subject cohort

```bash
python -m dataset.data_parser.sample_holdout \
    --data_dir  /home/coder/workspace/updown/physiome_hetero/train \
    --n 100 --seed 777
```

Output: `holdout_case_ids.json`. Use the same file in **all** training
stages so the cohort never leaks into SSL.
Add `--stratify_by_cvp` to guarantee CVP-bearing subjects in the holdout
(otherwise random sampling can under-represent the 25%-prevalence CVP set).

### 1.4. Parse VitalDB → full-signal downstream data (separate parser)

Used by IOH / AKI / hypotension labelers (raw signal, no windowing):

```bash
python -m dataset.data_parser.vital_db_downstream \
    --src_path  /home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files \
    --trg_path  /home/coder/workspace/updown/physiome_hetero/downstream \
    --sfreq 100 --require_abp
```

## 2. Phase-1 NeuroNet (per-modality SSL)

### 2.1. Single GPU, sequential sweep over the 4 modalities

```bash
for IDX in 0 1 2 3; do
    python -m pretrained.dp_neuronet.train \
        --config_yaml config/vital_db/dp_neuronet.yaml \
        --ch_idx $IDX \
        --num_workers 16 --prefetch_factor 4 \
        --holdout_subjects_file /home/coder/workspace/updown/physiome_hetero/train/holdout_case_ids.json \
        > log_phase1_${IDX}.txt 2>&1
done
```

`--ch_idx` maps to `ch_names`: 0=ABP, 1=ECG, 2=PPG, 3=CVP.

### 2.2. Two GPUs in parallel (faster)

```bash
CUDA_VISIBLE_DEVICES=0 python -m pretrained.dp_neuronet.train \
    --config_yaml config/vital_db/dp_neuronet.yaml --ch_idx 0 \
    --num_workers 16 --prefetch_factor 4 \
    --holdout_subjects_file /home/coder/workspace/updown/physiome_hetero/train/holdout_case_ids.json \
    > log_phase1_ABP.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m pretrained.dp_neuronet.train \
    --config_yaml config/vital_db/dp_neuronet.yaml --ch_idx 1 \
    --num_workers 16 --prefetch_factor 4 \
    --holdout_subjects_file /home/coder/workspace/updown/physiome_hetero/train/holdout_case_ids.json \
    > log_phase1_ECG.txt 2>&1 &

wait
# then PPG (ch_idx 2) + CVP (ch_idx 3)
```

Output ckpts: `<ckpt_path>/neuronet/<MODAL>/model/best_model.pth`.

## 3. Phase-2 PhysioME (multimodal hetero SSL)

### 3.1. Hetero recipe (main run)

```bash
python -m pretrained.physiome.train_hetero \
    --config_yaml config/vital_db/physiome_hetero.yaml
```

The yaml carries `holdout_subjects_file` so the same cohort is excluded
from Phase-2 SSL. Output ckpt: `<ckpt_path>/physiome_hetero/...`.

### 3.2. A1 baseline (synthetic-only PhysioME, no hetero buckets)

```bash
python -m pretrained.physiome.train \
    --config_yaml config/vital_db/physiome.yaml
```

### 3.3. A3 ablation (`restoration_only_on_complete: false`)

```bash
python -m pretrained.physiome.train_hetero \
    --config_yaml config/vital_db/physiome_hetero_a3.yaml
```

## 4. Downstream

```bash
# IOH (primary)
python -m downstream.run_ioh --ckpt_path <phase2_ckpt>

# AKI (KDIGO Cr-based)
python -m downstream.run_aki --ckpt_path <phase2_ckpt>

# Mortality cross-dataset transfer (MIMIC-III WDB)
python -m downstream.run_mortality --ckpt_path <phase2_ckpt> \
    --mode {zero_shot,linear_probe}

# A2 ablation (real-missing vs synth-missing OOD gap)
python -m downstream.run_ablation_a2 --ckpt_path <phase2_ckpt>

# Calibration plot from saved preds
python -m downstream.calibration --preds_path <preds_npz>
```

## 5. Smoke tests (no real data, ~1 min each)

```bash
python experiments/smoke_test_dp_neuronet.py    # Phase-1 TF-C
python experiments/smoke_test_hetero.py         # Phase-2 hetero
python experiments/smoke_test_downstream.py     # frozen-encoder eval
python experiments/smoke_test_probe.py          # subset enumeration
python experiments/smoke_test_holdout.py        # subject-level holdout
```

## 6. Common overrides (all train scripts)

| Flag | Purpose |
|---|---|
| `--ssl_data_dir <path>` | Override yaml SSL data dir (e.g. point at local-SSD copy) |
| `--ckpt_path <path>` | Override yaml ckpt save dir |
| `--num_workers <int>` | DataLoader workers |
| `--prefetch_factor <int>` | DataLoader prefetch (Phase-1 only) |
| `--shard_cache_size <int>` | LRU cache size in shards |
| `--holdout_subjects_file <json>` | Subject-level exclusion list |
| `--ch_idx <0..3>` | Phase-1 only: which modality to train |
