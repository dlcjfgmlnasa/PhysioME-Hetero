# Probing

Per-epoch dev-cohort linear probing turns SSL training from
*"loss is going down, hopefully that means progress"* into *"AUROC at
epoch 7 jumped from 0.61 to 0.68, learning is on track"*. Without it
backbone collapse / mode collapse / hyperparam mismatch only shows up
after Phase-2 + downstream eval (~36 h wasted per failed modality).

## Phase-1 — per-modality probe

Every Phase-1 backbone gets its own probe task. Dispatch via
`pretrained/probe_dev_data.PHASE1_PROBE_TASK_FOR_MODAL`:

| modality | task key | label | clinical meaning | cutoff source |
|---|---|---|---|---|
| ABP | `ioh` | sustained MAP < 65 mmHg ≥ 60 s in 5-min lookahead | intra-op hypotension | Hatib et al. Anesthesiology 2018 |
| ECG | `ioh` | (same) | encoder-quality probe — *"is HR/HRV info informative for IOH?"* | (same) |
| PPG | `ioh` | (same) | encoder-quality probe via PPG-derived hemodynamic features | (same) |
| **CVP** | `cvp` | mean CVP > 12 mmHg sustained 30 s in 5-min lookahead | venous congestion / volume overload | ICU venous-congestion threshold |
| **CO2** | `co2` | mean EtCO2 > 50 mmHg sustained 30 s | hypercapnia | ASA acute-hypercapnia line |
| **AWP** | `awp` | **peak** AWP > 30 cmH2O sustained 30 s | barotrauma risk | ARDSNet plateau pressure |

CVP/CO2/AWP cutoffs are clinical-guideline standards (not learned). Each
probe uses the IOH protocol (60 s window, 5-min lookahead, sliding 60 s
stride) so the same `extract_*_samples` machinery applies to all six.

Implementation: `downstream/tasks/modality_forecast.py` (CVP/CO2/AWP
preset configs + generic `extract_modality_samples`) +
`pretrained/probe_dev_data.py::load_dev_probe_modality_split` + the
Phase-1 trainer's `_run_probe`.

## Logging format

Phase-1 `logs/train.log` contains one line per probe row:

```
2026-05-12 03:14:22 probe step=00050000 epoch=023 task=ioh probe_auroc=0.7234 probe_macro_f1=0.6512
2026-05-12 03:14:25 probe step=00050000 epoch=023 task=cvp probe_auroc=0.6841 probe_macro_f1=0.6202
```

`task=<key>` lets you grep per-modality curves out of the merged log:

```bash
grep "task=cvp"  $CKPT/neuronet/CVP/logs/train.log | awk '{print $5,$7}'
```

## Phase-1 — what the probe doesn't decide

**Best epoch is still chosen by held-out TF-C loss, not by probe AUROC.**
The probe is a *monitoring* signal, not a *selection* signal. Two
reasons:

1. The probe label is a downstream task (IOH or modality-specific
   forecast); using it for selection would leak downstream information
   into pretraining hparam choice.
2. SSL loss minimum and downstream maximum are decoupled in general
   (well-known SSL gap); but TF-C loss is at least *unsupervised*
   ground truth for the SSL objective itself.

If the probe AUROC is improving while TF-C loss is monotone decreasing,
that's the green-light signal. If TF-C loss decreases but probe AUROC
stays flat at 0.5, **kill the run** — backbone is mode-collapsing.

## Phase-2 — multimodal probe

`pretrained/physiome/train_hetero.py::linear_probing` runs every epoch
(or every `probe_every` epochs):

1. Pick subsets via `select_probe_subsets(PROBE_MODALITIES, max_subsets,
   seed=epoch)`. `PROBE_MODALITIES = (ABP, ECG, PPG)` because
   `vital_db_downstream.py` only emits these three as label-bearing
   inputs for IOH.
2. For each subset, extract dev-cohort latents through
   `inference_missing_modality` (so absent modalities are filled by the
   trained `dropped_modality_token`).
3. Fit `LogisticRegression(max_iter=1000, C=1.0)` on probe-train,
   evaluate on probe-eval. Report mean AUROC + mean macro-F1 across
   subsets.

**Best epoch = highest mean macro-F1.** This *is* a selection signal
(supervised), so the dev cohort must stay disjoint from the test
holdout — which it does by construction (see [Cohort](Cohort.md)).

## Probe disabled cases

If `probe_downstream_dir` or `probe_subjects_file` is unset, probing is
skipped (training continues with TF-C loss / SSL loss as the only
monitoring signal). The trainer prints a clear `>> Probe : skipped`
line at startup.

If the dev cohort contains zero subjects with a given Phase-1 modality
(e.g. `--n_dev` was tiny and no CVP-bearing subject was sampled), the
modality-specific loader raises `RuntimeError`. The trainer catches it
and prints `[warn] modality probe disabled: ...` — Phase-1 SSL continues
without that probe rather than crashing the whole run.

## Probe overhead

Each Phase-1 probe per epoch:
* dev forward pass (~50 subjects × ~10 segments per subject × 60 s @ 100 Hz)
* `LogisticRegression.fit` on ~500 samples × encoder_embed_dim features

Wall-clock: typically 5-15 s per probe, dwarfed by epoch SSL time.
Phase-2 probe runs `max_subsets` LR fits per epoch (default 10-15) so
~30-90 s — still negligible vs the ~30 min/epoch SSL cost.

## Re-using the probe presets as auxiliary downstream tables

`TASK_PRESETS` in `downstream/tasks/modality_forecast.py` is intended to
be re-used as standalone downstream evaluations, not just probes:

```python
from downstream.tasks.modality_forecast import (
    TASK_PRESETS, load_cases_modality, extract_modality_samples,
    ModalityForecastDataset,
)

task = TASK_PRESETS['co2']
cases = load_cases_modality('<downstream_dir>', task)
# filter to holdout cohort, extract samples, ModalityForecastDataset, ...
```

Promoting them to full holdout-cohort downstream rows adds 3 extra
clinically-meaningful predictions to the paper table without writing
any new task code. (Listed as extension-lever #9 in
`memory/project_physiome_state.md`.)
