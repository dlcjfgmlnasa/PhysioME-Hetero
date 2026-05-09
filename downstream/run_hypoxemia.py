# -*- coding:utf-8 -*-
"""Hypoxemia downstream evaluation.

Protocol (mirrors run_ioh.py):
  1. Load VitalDB full-signal downstream npz (vital_db_downstream.py output)
  2. Subject-level split driven by holdout JSON (sample_holdout output):
       test  = case_ids in --holdout_subjects_file
       train = remaining cases (optionally minus --dev_subjects_file)
  3. Slide 60 s input windows with 5-min look-ahead horizon
  4. Label = sustained SpO2 < 92% for ≥30 s within the horizon
  5. Freeze PhysioME-Hetero encoder; extract latent for every 2^N-1 modal subset
  6. Fit LogisticRegression; report AUROC, AUPRC, Sensitivity@Sp90% per subset
  7. Print table and write results/<tag>_hypoxemia.csv

Usage:
  python downstream/run_hypoxemia.py \\
      --ckpt_path  ckpt/vital_db/physiome_hetero/model/best_model.pth \\
      --data_dir   data/vitaldb_downstream \\
      --holdout_subjects_file data/vitaldb_ssl/holdout_case_ids.json \\
      --out_dir    results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from itertools import combinations
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])
warnings.filterwarnings('ignore')

from downstream.tasks.hypoxemia import (
    HypoxemiaDataset,
    HypoxemiaSample,
    extract_hypoxemia_samples,
    load_cases,
)
from downstream.utils import load_pretrained_to_classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODAL_ORDER = ('ABP', 'ECG', 'PPG', 'CVP', 'CO2', 'AWP')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True, type=str)
    p.add_argument('--data_dir', required=True, type=str,
                   help='Directory of vitaldb_downstream npz files')
    p.add_argument('--holdout_subjects_file', required=True, type=str,
                   help='JSON written by sample_holdout.py — defines the '
                        'downstream test cohort.')
    p.add_argument('--dev_subjects_file', type=str, default=None,
                   help='JSON of dev cohort (Phase-2 probe). Excluded from '
                        'BOTH train and test of this script if provided.')
    p.add_argument('--out_dir', default='results', type=str)
    p.add_argument('--window_sec', default=60.0, type=float)
    p.add_argument('--stride_sec', default=60.0, type=float)
    p.add_argument('--horizon_sec', default=300.0, type=float)
    p.add_argument('--spo2_threshold', default=92.0, type=float)
    p.add_argument('--sustained_sec', default=30.0, type=float)
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--tag', default='hetero', type=str,
                   help='Output file prefix (e.g. "hetero", "baseline")')
    p.add_argument('--save_preds', action='store_true',
                   help='Also save raw (y_true, y_score) npy for calibration analysis')
    p.add_argument('--max_subsets', type=int, default=0,
                   help='Cap on number of modal subsets to evaluate '
                        '(0 = all 2^N-1).')
    return p.parse_args()


def _load_case_ids(path: Optional[str]) -> Set[str]:
    """Read a sample_holdout JSON and return its case_ids as a str set."""
    if not path:
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return {str(c) for c in payload['case_ids']}


def extract_features(
    model,
    samples: List[HypoxemiaSample],
    modal_subset: Tuple[str, ...],
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = HypoxemiaDataset(samples, modal_order=list(modal_subset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_x, all_y = [], []
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data = {k: v.to(device) for k, v in data.items()}
            x = model.physio_me.inference_missing_modality(data=data)
            all_x.append(x.cpu().numpy())
            all_y.append(labels.numpy())
    return np.concatenate(all_x, 0), np.concatenate(all_y, 0)


def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    mask = spec >= target_spec
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading pretrained model...')
    classifier, _ = load_pretrained_to_classifier(args.ckpt_path, n_classes=2)
    classifier = classifier.to(device)
    classifier.eval()

    print(f'Loading cases from {args.data_dir}...')
    all_cases = load_cases(
        data_dir=args.data_dir,
        input_signals=list(MODAL_ORDER),
        min_duration_sec=600.0,
    )
    print(f'  {len(all_cases)} valid cases (with SpO2 label)')

    holdout_ids = _load_case_ids(args.holdout_subjects_file)
    dev_ids = _load_case_ids(args.dev_subjects_file)
    overlap = holdout_ids & dev_ids
    if overlap:
        raise ValueError(
            f'holdout and dev cohorts overlap on {len(overlap)} case(s); '
            'they must be disjoint.'
        )

    all_cases.sort(key=lambda c: str(c['case_id']))
    test_cases = [c for c in all_cases if str(c['case_id']) in holdout_ids]
    train_cases = [
        c for c in all_cases
        if str(c['case_id']) not in holdout_ids
        and str(c['case_id']) not in dev_ids
    ]
    missing = holdout_ids - {str(c['case_id']) for c in all_cases}
    if missing:
        print(f'  [warn] {len(missing)} holdout case_ids not present in '
              f'{args.data_dir} (e.g. filtered by min_duration_sec).')
    print(f'  Train: {len(train_cases)}  Test: {len(test_cases)} '
          f'(dev excluded: {len(dev_ids)})')
    if not test_cases:
        raise RuntimeError(
            'Test set is empty after applying holdout filter. '
            'Check that --holdout_subjects_file matches data_dir cohort.'
        )

    print('Extracting windows...')
    kw = dict(
        input_signals=list(MODAL_ORDER),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        horizon_sec=args.horizon_sec,
        spo2_threshold=args.spo2_threshold,
        sustained_sec=args.sustained_sec,
    )
    train_samples = extract_hypoxemia_samples(train_cases, **kw)
    test_samples = extract_hypoxemia_samples(test_cases, **kw)
    pos_rate = np.mean([s.label for s in train_samples]) if train_samples else 0.0
    print(f'  Train samples: {len(train_samples)}  positive rate: {pos_rate:.3f}')
    print(f'  Test  samples: {len(test_samples)}')

    if args.max_subsets and args.max_subsets > 0:
        from pretrained.physiome.probe_utils import select_probe_subsets
        modal_subsets = select_probe_subsets(
            list(MODAL_ORDER), max_subsets=args.max_subsets, seed=42,
        )
    else:
        modal_subsets = []
        for r in range(1, len(MODAL_ORDER) + 1):
            for combo in combinations(MODAL_ORDER, r):
                modal_subsets.append(combo)

    rows = []
    header = 'Subset,AUROC,AUPRC,Sens@Sp90'
    print('\n' + header)

    for subset in modal_subsets:
        subset_name = '+'.join(subset)
        train_x, train_y = extract_features(classifier, train_samples, subset, args.batch_size)
        test_x, test_y = extract_features(classifier, test_samples, subset, args.batch_size)

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(train_x, train_y)
        prob = lr.predict_proba(test_x)[:, 1]

        auroc = roc_auc_score(test_y, prob)
        auprc = average_precision_score(test_y, prob)
        sens90 = sensitivity_at_specificity(test_y, prob, target_spec=0.90)

        row = f'{subset_name},{auroc:.4f},{auprc:.4f},{sens90:.4f}'
        rows.append(row)
        print(row)

        if args.save_preds:
            preds_dir = os.path.join(args.out_dir, 'preds')
            os.makedirs(preds_dir, exist_ok=True)
            np.save(os.path.join(preds_dir, f'{args.tag}_hypoxemia_{subset_name}_ytrue.npy'), test_y)
            np.save(os.path.join(preds_dir, f'{args.tag}_hypoxemia_{subset_name}_yscore.npy'), prob)

    csv_path = os.path.join(args.out_dir, f'{args.tag}_hypoxemia.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
