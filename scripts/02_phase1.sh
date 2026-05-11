#!/usr/bin/env bash
# 02_phase1.sh — Phase-1 unimodal SSL for all 6 modalities, in order.
#
# Sequential by default (PHASE1_PARALLEL=1). Honors $PHASE1_PARALLEL:
#   1 = run modalities one at a time on cuda:0
#   2 = pair on (cuda:0, cuda:1) per round  (needs ~24 GB free/GPU + 60 GB RAM)
#
# Auto-resume:
#   A modality whose best_model.pth already exists is skipped, so re-running
#   after a partial run (e.g. ABP done, ECG killed) picks up at ECG without
#   re-training what's already finished. Set FORCE=1 to retrain everything.
source "$(dirname "$0")/_env.sh"
print_env

CH_NAMES=(ABP ECG PPG CVP CO2 AWP)
CH_IDXS=(0 1 2 3 4 5)
FORCE="${FORCE:-0}"

ckpt_done() {
    local name="$1"
    local p="$CKPT_ROOT/neuronet/${name}/model/best_model.pth"
    [[ -f "$p" ]]
}

train_one() {
    # NOTE: split across lines because `set -u` evaluates ${CH_NAMES[$idx]}
    # in the same `local` statement *before* idx itself is bound — the
    # one-liner version raises "unbound variable: idx".
    local idx="$1"
    local gpu="$2"
    local name="${CH_NAMES[$idx]}"
    local out="$LOG_DIR/02_phase1_${name}.log"
    if [[ "$FORCE" != "1" ]] && ckpt_done "$name"; then
        log "↷ SKIP  phase1[$name] — best_model.pth already exists "
            "(set FORCE=1 to retrain)"
        return 0
    fi
    log "▶ START phase1[$name] (idx=$idx gpu=$gpu eager=$EAGER) → $out"
    local eager_args=()
    if [[ "$EAGER" == "1" ]]; then
        eager_args+=(--eager --eager_workers "$EAGER_WORKERS")
    fi
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
        "${eager_args[@]}" \
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
    log "Phase-1: 2-GPU parallel (3 rounds × 2 modals, skipping done)"
    for round in 0 1 2; do
        a=$((round * 2)); b=$((round * 2 + 1))
        train_one "$a" 0 &
        pid_a=$!
        train_one "$b" 1 &
        pid_b=$!
        wait "$pid_a" "$pid_b"
    done
else
    log "Phase-1: sequential on cuda:0 (skipping done)"
    for idx in "${CH_IDXS[@]}"; do
        train_one "$idx" 0
    done
fi

log "02_phase1: complete — ckpts under $CKPT_ROOT/neuronet/<MODAL>/model/"
