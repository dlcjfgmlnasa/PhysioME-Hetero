#!/usr/bin/env bash
# Shared environment for the PhysioME-Hetero training pipeline scripts.
# Source this from every step script:  source "$(dirname "$0")/_env.sh"
# Override any variable from the calling shell BEFORE invoking the script.

set -euo pipefail

# ── Repo root (resolves regardless of CWD) ──────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Data paths (override per environment) ───────────────────────────
# Source vital files (raw VitalDB downloads).
: "${VITAL_SRC:=/home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files}"

# v2 cohort outputs go under a versioned root so the v1 (4-modal) shards stay
# untouched on disk. Override BASE if you want a different layout.
: "${BASE:=/home/coder/workspace/updown/physiome_hetero_v2}"
: "${SSL_DIR:=$BASE/train}"           # sharded SSL output (vital_db_ssl)
: "${DS_DIR:=$BASE/downstream}"       # full-signal downstream npz
: "${CKPT_ROOT:=$BASE/ckpt}"          # Phase-1 + Phase-2 checkpoints
: "${LOG_DIR:=$BASE/logs}"            # per-step stdout logs

# External validation (MIMIC-III WDB).
: "${MIMIC_NPZ_DIR:=$BASE/mimic3_hetero}"
: "${MIMIC_COHORT_CSV:=$BASE/icu_mortality_cohort.csv}"

# AKI label CSVs (VitalDB official; download manually from vitaldb.net).
: "${AKI_CLINICAL_CSV:=$BASE/clinical_data.csv}"
: "${AKI_LAB_CSV:=$BASE/lab_data.csv}"

# ── Cohort knobs (sample_holdout) ───────────────────────────────────
: "${HOLDOUT_N:=100}"
: "${DEV_N:=50}"
: "${COHORT_SEED:=777}"

# Derived cohort JSON paths (do NOT override unless paths above changed).
HOLDOUT_JSON="$SSL_DIR/holdout_case_ids.json"
DEV_JSON="$SSL_DIR/dev_case_ids.json"
export HOLDOUT_JSON DEV_JSON

# ── DataLoader knobs ────────────────────────────────────────────────
: "${NUM_WORKERS:=16}"
: "${PREFETCH_FACTOR:=4}"
: "${SHARD_CACHE_SIZE:=2}"
# EAGER=1 pre-loads every (modality, kept-segment) slice into RAM at
# init (one parallel shard sweep). Eliminates per-batch network I/O at
# the cost of ~13-20 GB RAM per Phase-1 modality. Strongly recommended
# on slow / network filesystems (e.g. KHDP /home/coder/workspace NFS).
# Two modalities run in parallel under PHASE1_PARALLEL=2, so plan for
# ~30-40 GB peak RAM.
: "${EAGER:=0}"
: "${EAGER_WORKERS:=8}"

# ── GPU layout ──────────────────────────────────────────────────────
# How many GPUs to parallelise Phase-1 across (1 = sequential, 2 = pair).
: "${PHASE1_PARALLEL:=2}"

# ── Downstream knobs ────────────────────────────────────────────────
# 0 = enumerate every 2^N-1 modal subset (63 at N=6, slow).
# 15 = full + each single-modal + ~8 random — recommended for the main paper
# table; bump to 0 only when the supplementary needs every subset.
: "${MAX_SUBSETS:=15}"

# ── Helpers ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

stamp() { date +%Y-%m-%dT%H:%M:%S; }

log() {
    echo "[$(stamp)] $*" | tee -a "$LOG_DIR/pipeline.log"
}

run_step() {
    # run_step <step_name> <command...>
    local name="$1"; shift
    local out="$LOG_DIR/${name}.log"
    log "▶ START  $name  → $out"
    if "$@" > "$out" 2>&1; then
        log "✔ DONE   $name"
    else
        local rc=$?
        log "✘ FAILED $name  (exit=$rc)  see $out"
        return $rc
    fi
}

print_env() {
    cat <<EOF
[env] REPO_ROOT       = $REPO_ROOT
[env] VITAL_SRC       = $VITAL_SRC
[env] BASE            = $BASE
[env]  SSL_DIR        = $SSL_DIR
[env]  DS_DIR         = $DS_DIR
[env]  CKPT_ROOT      = $CKPT_ROOT
[env]  LOG_DIR        = $LOG_DIR
[env] HOLDOUT_JSON    = $HOLDOUT_JSON
[env] DEV_JSON        = $DEV_JSON
[env] cohort: n=$HOLDOUT_N  n_dev=$DEV_N  seed=$COHORT_SEED
[env] PHASE1_PARALLEL = $PHASE1_PARALLEL
[env] MAX_SUBSETS     = $MAX_SUBSETS
EOF
}

# When sourced, change CWD to repo root so relative paths in configs work.
cd "$REPO_ROOT"
