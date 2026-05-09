#!/usr/bin/env bash
# 03_phase2.sh — Phase-2 PhysioME-Hetero (multimodal SSL on 6 modalities).
# Reads the per-modal Phase-1 ckpts via yaml ``modality_name2path``.
source "$(dirname "$0")/_env.sh"
print_env

# Sanity: every Phase-1 ckpt must be present before Phase-2 can wire backbones.
for m in ABP ECG PPG CVP CO2 AWP; do
    p="$CKPT_ROOT/neuronet/$m/model/best_model.pth"
    if [[ ! -f "$p" ]]; then
        log "✘ missing Phase-1 ckpt: $p"
        log "  run scripts/02_phase1.sh first."
        exit 1
    fi
done

# Phase-2 reads holdout/probe/probe_dir paths via yaml. To keep the script
# self-contained we also pass them as overrides at the dataclass level —
# train_hetero.py uses getattr(self.args, ...) so unknown keys are ignored.
# However the current trainer reads strictly from yaml; the cleanest path is
# to write a small override yaml at runtime. We do that here.

OVERRIDE_YAML="$LOG_DIR/physiome_hetero.run.yaml"
python - <<PY
import yaml, os
src = os.path.join("config", "vital_db", "physiome_hetero.yaml")
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["ssl_data_dir"] = "$SSL_DIR"
cfg["ckpt_path"]    = os.path.join("$CKPT_ROOT", "physiome_hetero")
cfg["holdout_subjects_file"] = "$HOLDOUT_JSON"
cfg["probe_subjects_file"]   = "$DEV_JSON"
cfg["probe_downstream_dir"]  = "$DS_DIR"
cfg["num_workers"] = int($NUM_WORKERS)
with open("$OVERRIDE_YAML", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"wrote {os.path.abspath('$OVERRIDE_YAML')}")
PY

run_step "03_phase2_train_hetero" \
    python -m pretrained.physiome.train_hetero \
        --config_yaml "$OVERRIDE_YAML"

log "03_phase2: complete — ckpt at $CKPT_ROOT/physiome_hetero/model/best_model.pth"
