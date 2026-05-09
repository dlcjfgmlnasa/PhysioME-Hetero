#!/usr/bin/env bash
# run_all.sh — Run the full PhysioME-Hetero v2 pipeline end-to-end.
#
# Toggle individual stages with env vars (default = run):
#     SKIP_PARSE=1      bash scripts/run_all.sh   # data already parsed
#     SKIP_PARSE=1 SKIP_COHORT=1 bash scripts/run_all.sh
#     ONLY_DOWNSTREAM=1 bash scripts/run_all.sh   # 04+05 only
#
# Override paths/sizes inline:
#     BASE=/data/physiome_v2 PHASE1_PARALLEL=1 MAX_SUBSETS=0 \
#         bash scripts/run_all.sh
source "$(dirname "$0")/_env.sh"
print_env

if [[ "${ONLY_DOWNSTREAM:-0}" == "1" ]]; then
    SKIP_PARSE=1; SKIP_COHORT=1; SKIP_PHASE1=1; SKIP_PHASE2=1
fi

run_stage() {
    local script="$1" skip_var="$2"
    if [[ "${!skip_var:-0}" == "1" ]]; then
        log "── SKIP $script (${skip_var}=1)"
        return 0
    fi
    log "════════════════════════════════════════════════════════"
    log "── RUN  $script"
    log "════════════════════════════════════════════════════════"
    bash "$SCRIPT_DIR/$script"
}

run_stage 00_parse_data.sh    SKIP_PARSE
run_stage 01_make_cohorts.sh  SKIP_COHORT
run_stage 02_phase1.sh        SKIP_PHASE1
run_stage 03_phase2.sh        SKIP_PHASE2
run_stage 04_downstream.sh    SKIP_DOWNSTREAM
run_stage 05_external.sh      SKIP_EXTERNAL

log "run_all: pipeline complete"
log "  results: $BASE/results"
log "  logs:    $LOG_DIR"
