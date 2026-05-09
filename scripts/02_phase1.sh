#!/usr/bin/env bash
# 02_phase1.sh — Phase-1 unimodal SSL for all 6 modalities.
# Honors $PHASE1_PARALLEL: 1 = sequential, 2 = pair on (cuda:0, cuda:1) per round.
source "$(dirname "$0")/_env.sh"
print_env

CH_NAMES=(ABP ECG PPG CVP CO2 AWP)

train_one() {
    local idx="$1" gpu="$2" name="${CH_NAMES[$idx]}"
    local out="$LOG_DIR/02_phase1_${name}.log"
    log "▶ START phase1[$name] (idx=$idx gpu=$gpu) → $out"
    CUDA_VISIBLE_DEVICES="$gpu" python -m pretrained.dp_neuronet.train \
        --config_yaml config/vital_db/dp_neuronet.yaml \
        --ch_idx "$idx" \
        --ckpt_path "$CKPT_ROOT" \
        --ssl_data_dir "$SSL_DIR" \
        --num_workers "$NUM_WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        --shard_cache_size "$SHARD_CACHE_SIZE" \
        --holdout_subjects_file "$HOLDOUT_JSON" \
        --probe_downstream_dir  "$DS_DIR" \
        --probe_subjects_file   "$DEV_JSON" \
        --probe_every 1 \
        > "$out" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        log "✔ DONE  phase1[$name]"
    else
        log "✘ FAIL  phase1[$name]  (exit=$rc)"
    fi
    return $rc
}

if [[ "$PHASE1_PARALLEL" == "2" ]]; then
    log "Phase-1: 2-GPU parallel (3 rounds × 2 modals)"
    for round in 0 1 2; do
        a=$((round * 2)); b=$((round * 2 + 1))
        train_one "$a" 0 &
        pid_a=$!
        train_one "$b" 1 &
        pid_b=$!
        wait "$pid_a" "$pid_b"
    done
else
    log "Phase-1: sequential on cuda:0"
    for idx in 0 1 2 3 4 5; do
        train_one "$idx" 0
    done
fi

log "02_phase1: complete — ckpts under $CKPT_ROOT/neuronet/<MODAL>/model/"
