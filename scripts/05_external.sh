#!/usr/bin/env bash
# 05_external.sh — VitalDB → MIMIC-III WDB external validation (mortality).
# Skipped automatically if MIMIC artifacts are absent.
source "$(dirname "$0")/_env.sh"
print_env

CKPT="$CKPT_ROOT/physiome_hetero/model/best_model.pth"
if [[ ! -f "$CKPT" ]]; then
    log "✘ Phase-2 ckpt not found: $CKPT"
    exit 1
fi
if [[ ! -d "$MIMIC_NPZ_DIR" || ! -f "$MIMIC_COHORT_CSV" ]]; then
    log "External validation: SKIP (need $MIMIC_NPZ_DIR and $MIMIC_COHORT_CSV)"
    exit 0
fi
RESULTS="$BASE/results"
mkdir -p "$RESULTS"

# Zero-shot transfer: encoder never sees MIMIC during training; LR refit per fold.
run_step "05a_run_mortality_zero_shot" \
    python -m downstream.run_mortality \
        --ckpt_path "$CKPT" \
        --mimic_npz_dir "$MIMIC_NPZ_DIR" \
        --cohort_csv    "$MIMIC_COHORT_CSV" \
        --mode zero_shot --seed 42 \
        --out_dir "$RESULTS" --tag hetero

# Linear probe inside MIMIC (sorted 80/20 split unless a holdout JSON is given).
run_step "05b_run_mortality_linear_probe" \
    python -m downstream.run_mortality \
        --ckpt_path "$CKPT" \
        --mimic_npz_dir "$MIMIC_NPZ_DIR" \
        --cohort_csv    "$MIMIC_COHORT_CSV" \
        --mode linear_probe --seed 42 \
        --out_dir "$RESULTS" --tag hetero_lp

log "05_external: complete — csv outputs under $RESULTS"
