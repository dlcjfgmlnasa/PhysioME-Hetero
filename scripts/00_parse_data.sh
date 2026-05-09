#!/usr/bin/env bash
# 00_parse_data.sh — Parse raw VitalDB into 6-modal SSL shards + downstream npz.
# One-shot (~9h on KHDP). Re-run only after dataset upgrades.
source "$(dirname "$0")/_env.sh"
print_env

mkdir -p "$SSL_DIR" "$DS_DIR"

run_step "00a_vital_db_ssl" \
    python -m dataset.data_parser.vital_db_ssl \
        --src_path "$VITAL_SRC" \
        --trg_path "$SSL_DIR" \
        --sfreq 100 --duration 60 --nan_max_ratio 0.1 \
        --segments_per_shard 2048

run_step "00b_vital_db_downstream" \
    python -m dataset.data_parser.vital_db_downstream \
        --src_path "$VITAL_SRC" \
        --trg_path "$DS_DIR" \
        --sfreq 100 --require_abp

log "00_parse_data: complete"
