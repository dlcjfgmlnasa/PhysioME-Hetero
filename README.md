# PhysioME-Hetero

Hetero-availability multimodal SSL foundation model for biosignal data
(VitalDB + MIMIC-III). v2 covers 6 modalities (ABP / ECG / PPG / CVP /
CO2 / AWP) with availability-aware loss decomposition, 3-state presence
embedding, and learnable `dropped_modality_token` for missing modalities.

## Quick start

```bash
# 1) edit paths in scripts/_env.sh OR override via env vars
export BASE=/home/coder/workspace/updown/physiome_hetero_v2
export VITAL_SRC=/home/coder/workspace/datasets/vitaldb_open/1.0.0/vital_files

# 2) run end-to-end (~2-3 days on KHDP 2× L40S)
bash scripts/run_all.sh

# resume from a specific stage
SKIP_PARSE=1 SKIP_COHORT=1 bash scripts/run_all.sh
ONLY_DOWNSTREAM=1 bash scripts/run_all.sh
```

See [`scripts/README.md`](scripts/README.md) for all stages and knobs.

## Documentation

In-repo wiki under [`docs/`](docs/Home.md):

| Page | Topic |
|---|---|
| [Home](docs/Home.md) | landing + quick orientation |
| [Architecture](docs/Architecture.md) | model internals: encoder, two decoders, presence embedding, dropped_modality_token, bug fixes vs original PhysioME |
| [Pipeline](docs/Pipeline.md) | what each `scripts/` stage does and when to re-run it |
| [Cohort](docs/Cohort.md) | holdout / dev / test split as single source of truth |
| [Probing](docs/Probing.md) | per-modality dev probes (IOH + venous-congestion + hypercapnia + high-AWP) |
| [Modalities](docs/Modalities.md) | 6 modality definitions, prevalence, preprocessing pipeline, quality checks |
| [Downstream](docs/Downstream.md) | IOH / Hypoxemia / AKI / Mortality / A2 ablation |
| [Troubleshooting](docs/Troubleshooting.md) | manifest mismatch, OOM, probe disabled, namespace collisions |

## Repo layout (top-level)

```
config/             # yaml configs (Phase-1 + Phase-2 variants)
dataset/data_parser # VitalDB / MIMIC parsers + cohort tooling + quality checks
downstream/         # frozen-encoder evaluators (IOH / Hypoxemia / AKI / Mortality / A2)
experiments/        # smoke + integrity tests (run before training)
models/             # PhysioME, NeuroNet, BFM TransformerEncoder, hand-rolled LoRA
pretrained/         # Phase-1 (dp_neuronet) + Phase-2 (physiome) trainers + dev probe loader
scripts/            # end-to-end shell pipeline (00 → 05 + run_all)
docs/               # wiki (this file links there)
```

## Status

* Active branch: `main`
* See `git log -1` for the current HEAD.
* Target venue: IEEE JBHI (~10 page methodology). v1 paper: arXiv:2510.11110.
