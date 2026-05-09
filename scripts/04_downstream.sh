#!/usr/bin/env bash
# 04_downstream.sh — Downstream evaluation on the holdout cohort.
# Tasks: IOH, Hypoxemia, AKI (if labels present), A2 ablation.
source "$(dirname "$0")/_env.sh"
print_env

CKPT="$CKPT_ROOT/physiome_hetero/model/best_model.pth"
if [[ ! -f "$CKPT" ]]; then
    log "✘ Phase-2 ckpt not found: $CKPT"
    log "  run scripts/03_phase2.sh first."
    exit 1
fi
RESULTS="$BASE/results"
mkdir -p "$RESULTS"

# IOH (primary)
run_step "04a_run_ioh" \
    python -m downstream.run_ioh \
        --ckpt_path "$CKPT" --data_dir "$DS_DIR" \
        --holdout_subjects_file "$HOLDOUT_JSON" \
        --dev_subjects_file     "$DEV_JSON" \
        --max_subsets "$MAX_SUBSETS" \
        --out_dir "$RESULTS" --tag hetero --save_preds

# Hypoxemia
run_step "04b_run_hypoxemia" \
    python -m downstream.run_hypoxemia \
        --ckpt_path "$CKPT" --data_dir "$DS_DIR" \
        --holdout_subjects_file "$HOLDOUT_JSON" \
        --dev_subjects_file     "$DEV_JSON" \
        --max_subsets "$MAX_SUBSETS" \
        --out_dir "$RESULTS" --tag hetero --save_preds

# A2 ablation (real-missing vs synth-missing)
run_step "04c_run_ablation_a2" \
    python -m downstream.run_ablation_a2 \
        --ckpt_path "$CKPT" --data_dir "$DS_DIR" \
        --max_subsets "$MAX_SUBSETS" \
        --out_dir "$RESULTS" --tag hetero

# AKI — only if both VitalDB CSVs are available locally
if [[ -f "$AKI_CLINICAL_CSV" && -f "$AKI_LAB_CSV" ]]; then
    run_step "04d_run_aki" \
        python -m downstream.run_aki \
            --ckpt_path "$CKPT" --data_dir "$DS_DIR" \
            --clinical_csv "$AKI_CLINICAL_CSV" \
            --lab_csv      "$AKI_LAB_CSV" \
            --max_subsets "$MAX_SUBSETS" \
            --out_dir "$RESULTS" --tag hetero
else
    log "AKI: skipping (missing $AKI_CLINICAL_CSV or $AKI_LAB_CSV)"
fi

log "04_downstream: complete — csv outputs under $RESULTS"
