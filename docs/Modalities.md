# Modalities

PhysioME-Hetero v2 trains over 6 biosignal modalities, grouped clinically:

| Group | Modality | VitalDB track | Sampling (native → target) | Prevalence | Role |
|---|---|---|---|---|---|
| Cardiovascular | **ABP** | `SNUADC/ART` | 500 Hz → 100 Hz | ~99% | invasive arterial pressure |
| Cardiovascular | **ECG** | `SNUADC/ECG_II` | 500 Hz → 100 Hz | ~99% | lead-II electrocardiogram |
| Cardiovascular | **PPG** | `SNUADC/PLETH` | 500 Hz → 100 Hz | ~95% | pulse oximeter plethysmogram |
| Cardiovascular (sparse) | **CVP** | `SNUADC/CVP` | 500 Hz → 100 Hz | ~25% | central venous pressure (Swan-Ganz / CV catheter; major surgery only) |
| Respiratory | **CO2** | `Primus/CO2` | 62.5 Hz → 100 Hz | ~80% | end-tidal capnography (GE Primus ventilator) |
| Respiratory (mech-vent) | **AWP** | `Primus/AWP` | 62.5 Hz → 100 Hz | ~50% | airway pressure (mechanical-ventilation only) |

All channels are resampled to a common 100 Hz so the multimodal encoder
sees aligned tensors. Mappings live in
`dataset/data_parser/vital_db_ssl.py::MODAL_TRACK_NAMES` +
`MODAL_TO_SIGNAL_KEY` + `MODAL_ORDER`.

## Why these six?

* ABP/ECG/PPG = standard hemodynamic trio in any anesthesia monitor.
* CVP = the *headline sparse modality*; its 25% prevalence is exactly the
  *"hetero-availability"* scenario the paper targets.
* CO2 + AWP = respiratory bundle, makes the model a true *cardio-pulmonary*
  foundation. CO2 = gas exchange; AWP = lung mechanics; together they
  cover the two axes of ventilation monitoring.

PAP (Swan-Ganz pulmonary artery, ~3% prevalence) and ICP (~1%) are
retained in `_quality_checks.py` + `_signal_filters.py` for an optional
"extreme-sparse" sub-experiment, but excluded from the main 6-modal
recipe.

## Per-channel preprocessing pipeline

`dataset/data_parser/_signal_filters.preprocess_channel` runs a fixed
sequence per channel:

```
raw → range check (NaN out-of-physiology) → spike detection (NaN spikes)
    → median (smoothing) → notch (50/60 Hz) → bandpass / lowpass
    → re-NaN at originally-bad samples
```

`SIGNAL_CONFIGS[<key>]` (`SignalConfig` dataclass) tunes each step per
channel:

| key | valid_range | filter | notch | spike | median |
|---|---|---|---|---|---|
| `ecg` | (-5, 5) mV | bandpass 0.5-40 Hz | 60 Hz | yes (10σ) | — |
| `abp` | (20, 300) mmHg | lowpass 15 Hz | — | yes (6σ) | k=5 |
| `ppg` | (0, 2000) | lowpass 8 Hz | 60 Hz | yes (6σ) | k=5 |
| `cvp` | (-5, 40) mmHg | lowpass 10 Hz | — | yes (8σ) | — |
| `co2` | (0, 100) mmHg | lowpass 5 Hz | — | — | — |
| `awp` | (-20, 80) cmH2O | lowpass 20 Hz | — | — | — |
| `pap` | (5, 80) mmHg | lowpass 15 Hz | — | yes (6σ) | k=5 |
| `icp` | (-10, 80) mmHg | lowpass 10 Hz | — | yes (8σ) | — |

Out-of-range or spike samples become `NaN`; downstream code (extractors,
quality checks) treat NaN as missing. Filters operate on a NaN-filled
copy then re-NaN at the original positions to avoid filtfilt smearing
artifacts across gaps.

## Three-layer segment validity

For each fixed-length window of each channel `vital_db_ssl.py::validate_segment`
checks:

1. **NaN ratio** ≤ `nan_max_ratio` (default 0.1).
2. **Generic quality score** (`segment_quality_score`):
   flatline ratio < 0.5, clip ratio < 0.1, high-freq ratio bounded,
   amplitude ≥ `min_amplitude`.
3. **Domain-specific check** (`domain_quality_check` in
   `_quality_checks.py`): physiology-aware HR / pulse / autocorr /
   respiratory-rate checks per signal type.

A window passes only if all three layers pass for that modality. Other
modalities are independent — `extract_segments` keeps any window where
*at least one* modality passed (the hetero-availability premise).

## Quality checks (per-signal physiology)

`_quality_checks.py` dispatcher:

| key | function | check |
|---|---|---|
| `ecg` | `ecg_quality_check` | HR within 30-200 bpm, R-R regularity |
| `abp` | `abp_quality_check` | HR within range, pulse pressure plausible |
| `ppg` | `ppg_quality_check` | similar to ABP, pulse-shape regularity |
| `cvp` | `cvp_quality_check` | HR + venous-pulse autocorr |
| `co2` | `co2_quality_check` | respiratory rate 4-40 /min |
| `awp` | `awp_quality_check` | respiratory rate 4-40 /min |
| `pap` | `pap_quality_check` | similar to ABP (pulsatile pressure) |
| `icp` | `icp_quality_check` | HR + venous-style autocorr |

CO2 and AWP share the generic `_respiration_quality_check` parameterised
by allowed RR range — both modalities oscillate with breathing.

## Adding a new modality

A modality is a one-place change if you've already got `_quality_checks`
+ `SIGNAL_CONFIGS` rows for it:

1. Edit `dataset/data_parser/vital_db_ssl.py`:
   - `MODAL_TRACK_NAMES['NEW'] = '<vendor>/<track>'`
   - `MODAL_TO_SIGNAL_KEY['NEW'] = 'new'`
   - append to `MODAL_ORDER`
2. Edit `pretrained/physiome/hetero_data_loader.py::MODAL_ORDER` to match.
3. Edit `pretrained/dp_neuronet/hetero_data_loader.py::MODAL` (only the
   `__main__` self-test).
4. Edit `dataset/data_parser/mimic3_waveform_ssl.py::MIMIC_CHANNEL_CANDIDATES`
   if MIMIC has a candidate name (otherwise it stays naturally-absent).
5. yamls: append to `dp_neuronet.ch_names`, `physiome_hetero.modality_name2path`.
6. (Optional) Add a Phase-1 probe task: extend `TASK_PRESETS` in
   `downstream/tasks/modality_forecast.py` + register in
   `pretrained/probe_dev_data.PHASE1_PROBE_TASK_FOR_MODAL`.
7. Re-run 00 → 01 → 02 → 03.

`SIGNAL_CONFIGS` and `_quality_checks` already cover ABP/ECG/PPG/CVP/CO2/AWP/PAP/ICP
— so adding any of these is steps 1-6 (skip the dispatcher edit).

## MIMIC-III channel mapping

`MIMIC_CHANNEL_CANDIDATES` (first match wins per modality):

| modality | candidates |
|---|---|
| ABP | `ABP`, `ART` |
| ECG | `II`, `I`, `III`, `V`, `aVR`, `aVL`, `aVF`, `MCL1` |
| PPG | `PLETH` |
| CVP | `CVP`, `CVP1`, `CVP2` |
| CO2 | `CO2`, `EtCO2`, `PETCO2`, `ETCO2` |
| AWP | `AWP`, `PAW`, `AWP1` |

CVP/CO2/AWP are rare in MIMIC-III WDB — most records leave these slots
empty (subject_modality_set bit = False). The model then handles them
as naturally-absent at inference time via `inference_missing_modality`.
This is the *trained-on-N, infer-on-M < N* hetero-availability demo.
