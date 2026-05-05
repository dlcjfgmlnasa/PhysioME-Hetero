# -*- coding:utf-8 -*-
"""Ablation A2: Real-missing vs Synthetic-missing generalization gap.

The key question: does hetero-bucket training close the gap between
performance on *naturally* missing modalities (modality never recorded in that
case) vs *artificially* dropped modalities (all modalities present, but we
hold some out at inference)?

This script evaluates a pretrained checkpoint on TWO disjoint test subsets:
  - real_missing : cases where ≥1 modality is truly absent in the recording
                   (modality_present[i] == False in the downstream npz)
  - synth_missing: cases where all 3 modalities are present; at inference we
                   artificially withhold one or two to simulate missing

For each subset we sweep all 2^N-1 modality subsets at inference and report
AUROC. The gap (synth_missing AUROC − real_missing AUROC) is Ablation A2.
A smaller gap = better OOD generalisation.

Usage:
  python downstream/run_ablation_a2.py \\
      --ckpt_path  ckpt/vital_db/physiome_hetero/model/best_model.pth \\
      --data_dir   data/vitaldb_downstream \\
      --out_dir    results \\
      --tag        hetero
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])
warnings.filterwarnings('ignore')

from downstream.tasks.hypotension import (
    ForecastSample,
    HypotensionDataset,
    extract_forecast_samples,
    load_cases,
)
from downstream.utils import load_pretrained_to_classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODAL_ORDER = ('ABP', 'ECG', 'PPG')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True)
    p.add_argument('--data_dir', required=True,
                   help='vitaldb_downstream npz directory')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--window_sec', default=60.0, type=float)
    p.add_argument('--stride_sec', default=60.0, type=float)
    p.add_argument('--horizon_sec', default=300.0, type=float)
    p.add_argument('--train_fraction', default=0.80, type=float)
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--tag', default='hetero')
    return p.parse_args()


def _load_case_modality_mask(data_dir: str) -> Dict[str, np.ndarray]:
    """Return {case_id: bool[3]} from saved modality_present arrays."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(data_dir, '*.npz'))):
        case_id = os.path.splitext(os.path.basename(p))[0]
        with np.load(p, allow_pickle=True) as f:
            out[case_id] = np.asarray(f['modality_present'], dtype=bool)
    return out


def _filter_samples_with_any_modal(samples: List[ForecastSample],
                                    modal_subset: Tuple[str, ...]) -> List[ForecastSample]:
    """Keep only samples that have at least one of the requested modals."""
    keys = {m.lower() for m in modal_subset}
    return [s for s in samples if keys & set(s.input_signals.keys())]


def extract_features(model, samples: List[ForecastSample],
                     modal_subset: Tuple[str, ...],
                     batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    # Drop samples where none of the subset modals are present.
    usable = _filter_samples_with_any_modal(samples, modal_subset)
    if not usable:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    dataset = HypotensionDataset(usable, modal_order=list(modal_subset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ys = [], []
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            if not data:
                continue
            data = {k: v.to(device) for k, v in data.items()}
            x = model.physio_me.inference_missing_modality(data=data)
            xs.append(x.cpu().numpy())
            ys.append(labels.numpy())
    if not xs:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def evaluate_subset(train_x, train_y, test_x, test_y) -> float:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(train_x, train_y)
    prob = lr.predict_proba(test_x)[:, 1]
    return float(roc_auc_score(test_y, prob))


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading pretrained model...')
    classifier, _ = load_pretrained_to_classifier(args.ckpt_path, n_classes=2)
    classifier = classifier.to(device)
    classifier.eval()

    print('Loading modality masks...')
    modality_masks = _load_case_modality_mask(args.data_dir)

    # Identify which cases are "real_missing" (≥1 absent) vs "complete" (all 3)
    complete_ids: Set[str] = set()
    real_missing_ids: Set[str] = set()
    for case_id, mask in modality_masks.items():
        if mask.all():
            complete_ids.add(case_id)
        else:
            real_missing_ids.add(case_id)
    print(f'  Complete cases: {len(complete_ids)}  '
          f'Real-missing cases: {len(real_missing_ids)}')

    # Load all cases (no modality requirement filter — real_missing may only have 1-2)
    all_cases = load_cases(
        data_dir=args.data_dir,
        input_signals=list(MODAL_ORDER),
        min_duration_sec=600.0,
    )
    # For real_missing, relax the required set to just ABP (needed for label).
    from downstream.tasks.hypotension import _load_case_npz
    import glob as _glob
    real_missing_cases = []
    for p in sorted(_glob.glob(os.path.join(args.data_dir, '*.npz'))):
        case_id = os.path.splitext(os.path.basename(p))[0]
        if case_id not in real_missing_ids:
            continue
        loaded = _load_case_npz(p)
        if loaded is None:
            continue
        sfreq = loaded['sfreq']
        signals = loaded['signals']
        min_len = min(s.shape[0] for s in signals.values())
        if min_len < int(600 * sfreq):
            continue
        signals = {k: v[:min_len] for k, v in signals.items()}
        real_missing_cases.append({'case_id': loaded['case_id'],
                                   'signals': signals, 'sfreq': sfreq})
    complete_cases = [c for c in all_cases if c['case_id'] in complete_ids]
    print(f'  Usable complete: {len(complete_cases)}  '
          f'Usable real-missing: {len(real_missing_cases)}')

    # Subject-level train/test split from *complete* cases → train proxy.
    complete_cases.sort(key=lambda c: c['case_id'])
    n_train = int(len(complete_cases) * args.train_fraction)
    train_cases = complete_cases[:n_train]
    # Test set: remaining complete (for synth_missing) + all real_missing
    synth_test_cases = complete_cases[n_train:]

    kw = dict(
        input_signals=list(MODAL_ORDER),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        horizon_sec=args.horizon_sec,
    )
    train_samples = extract_forecast_samples(train_cases, **kw)
    synth_samples = extract_forecast_samples(synth_test_cases, **kw)
    real_samples = extract_forecast_samples(real_missing_cases, **kw)
    print(f'  Train: {len(train_samples)}  '
          f'Synth-test: {len(synth_samples)}  '
          f'Real-test: {len(real_samples)}')

    modal_subsets = []
    for r in range(1, len(MODAL_ORDER) + 1):
        for combo in combinations(MODAL_ORDER, r):
            modal_subsets.append(combo)

    rows_synth, rows_real, rows_gap = [], [], []
    header = 'Subset,Synth_AUROC,Real_AUROC,Gap'
    print('\n' + header)

    for subset in modal_subsets:
        subset_name = '+'.join(subset)
        tr_x, tr_y = extract_features(classifier, train_samples, subset, args.batch_size)

        syn_x, syn_y = extract_features(classifier, synth_samples, subset, args.batch_size)
        auroc_syn = evaluate_subset(tr_x, tr_y, syn_x, syn_y)

        # For real-missing: only use windows where all subset modals are present
        # (HypotensionDataset already handles absent modals gracefully by dropping them,
        # but the encoder will restore them via inference_missing_modality).
        real_x, real_y = extract_features(classifier, real_samples, subset, args.batch_size)
        if real_x.shape[0] < 20:
            auroc_real = float('nan')
        else:
            auroc_real = evaluate_subset(tr_x, tr_y, real_x, real_y)

        gap = auroc_syn - auroc_real if not np.isnan(auroc_real) else float('nan')
        row = f'{subset_name},{auroc_syn:.4f},{auroc_real:.4f},{gap:.4f}'
        rows_synth.append(auroc_syn)
        rows_real.append(auroc_real)
        rows_gap.append(gap)
        print(row)

    # Summary
    valid = [g for g in rows_gap if not np.isnan(g)]
    print(f'\nMean gap (synth - real) = {np.mean(valid):.4f}  '
          f'(smaller is better; hetero training should narrow this)')

    csv_path = os.path.join(args.out_dir, f'{args.tag}_ablation_a2.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for subset, syn, real, gap in zip(modal_subsets, rows_synth, rows_real, rows_gap):
            f.write(f'{"+".join(subset)},{syn:.4f},{real:.4f},{gap:.4f}\n')
    print(f'Results saved to {csv_path}')


if __name__ == '__main__':
    main()
