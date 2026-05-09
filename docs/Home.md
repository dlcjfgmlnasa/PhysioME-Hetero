# PhysioME-Hetero — Documentation

Living docs for the PhysioME-Hetero v2 codebase (6-modal hetero-availability
multimodal SSL on VitalDB + MIMIC-III). This is the in-repo wiki — re-publish
to GitHub Wiki by copying these pages once `Repository → Settings → Wikis`
is enabled.

## Pages

| Page | Topic |
|---|---|
| [Architecture](Architecture.md) | PhysioME model: encoder, two decoder kinds, presence embedding, dropped_modality_token |
| [Pipeline](Pipeline.md) | `scripts/` end-to-end runner; what each stage does; how to skip stages |
| [Cohort](Cohort.md) | holdout / dev / test split, single source of truth, leakage invariants |
| [Probing](Probing.md) | Phase-1 per-modality probe tasks (IOH + venous-congestion + hypercapnia + high-AWP), Phase-2 multimodal probe |
| [Modalities](Modalities.md) | 6-modality definitions (ABP/ECG/PPG/CVP/CO2/AWP), prevalence, signal preprocessing, quality checks |
| [Downstream](Downstream.md) | IOH / Hypoxemia / AKI / Mortality / A2 ablation; subset cap; external validation (MIMIC-III) |
| [Troubleshooting](Troubleshooting.md) | manifest mismatch, probe disabled, OOM, missing CSVs, namespace collisions |

## Quick links

- Repo: https://github.com/dlcjfgmlnasa/PhysioME-Hetero
- v1 paper (4-modal, original PhysioME): arXiv:2510.11110
- v2 target venue: IEEE JBHI (~10 page methodology)
- Active branch: `main` — see `git log` for current HEAD

## Five-minute orientation

PhysioME-Hetero is a multimodal SSL foundation model for biosignal data
(arterial pressure / ECG / pulse oximetry / central venous pressure /
capnography / airway pressure) that explicitly models *which modalities
are present* via a 3-state embedding and a learnable
`dropped_modality_token`. The recipe handles three scenarios uniformly:

1. **Real-present** — modality measured for this subject.
2. **Synthetically dropped** — present but masked at training time for
   the restoration objective.
3. **Naturally absent** — never measured (e.g. a subject without a
   central line has no CVP signal).

The v2 release adds two respiratory modalities (CO2/AWP), a single
source-of-truth cohort split (holdout=test + dev=probe, both disjoint
from SSL), per-modality dev probing during SSL training, and an
end-to-end shell pipeline under `scripts/`.
