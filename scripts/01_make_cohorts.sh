#!/usr/bin/env bash
# 01_make_cohorts.sh — Build case_index, sample holdout(test) + dev(probe),
# verify split integrity. Run once per cohort decision.
source "$(dirname "$0")/_env.sh"
print_env

# 1) per-shard case index sidecar (~100 min on slow NFS)
run_step "01a_build_case_index" \
    python -m dataset.data_parser.build_case_index \
        --data_dir "$SSL_DIR"

# 2) sample disjoint cohorts (CVP-stratified)
#    --force lets us re-cut the same cohort layout under a fresh seed.
run_step "01b_sample_holdout" \
    python -m dataset.data_parser.sample_holdout \
        --data_dir "$SSL_DIR" \
        --n "$HOLDOUT_N" --n_dev "$DEV_N" --seed "$COHORT_SEED" \
        --stratify_by_cvp --force

# 3) verify the four invariants (I1: holdout ∩ dev = ∅,
#    I2: holdout ⊆ downstream, I3: holdout∪dev ⊆ SSL universe,
#    I4: MIMIC vs VitalDB id namespaces — only if mimic cohort csv exists).
VERIFY_ARGS=(
    --holdout        "$HOLDOUT_JSON"
    --dev            "$DEV_JSON"
    --downstream_dir "$DS_DIR"
    --ssl_dir        "$SSL_DIR"
)
if [[ -f "$MIMIC_COHORT_CSV" ]]; then
    VERIFY_ARGS+=(--mimic_cohort "$MIMIC_COHORT_CSV")
fi

run_step "01c_verify_split_disjoint" \
    python experiments/verify_split_disjoint.py "${VERIFY_ARGS[@]}"

log "01_make_cohorts: complete"
log "  holdout = $HOLDOUT_JSON"
log "  dev     = $DEV_JSON"
