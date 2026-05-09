# -*- coding:utf-8 -*-
"""AKI downstream evaluation — linear probing on pretrained PhysioME-Hetero.

Protocol:
  1. Load VitalDB downstream npz + AKI labels from VitalDB clinical/lab CSVs
  2. Subject-level 80/20 train/test split
  3. Slide 10-min windows over intra-operative waveforms
  4. Extract latent for all 2^N-1 modality subsets (frozen encoder)
  5. Fit LogisticRegression; report AUROC, AUPRC, Sens@Sp90%

Usage:
  python downstream/run_aki.py \\
      --ckpt_path  ckpt/vital_db/physiome_hetero/model/best_model.pth \\
      --data_dir   data/vitaldb_downstream \\
      --clinical_csv  data/vitaldb_clinical.csv \\
      --lab_csv       data/vitaldb_lab.csv \\
      --out_dir    results
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])
warnings.filterwarnings('ignore')

from downstream.tasks.aki import AKISample, AKIDataset, extract_aki_samples, load_aki_labels
from downstream.tasks.hypotension import load_cases
from downstream.utils import load_pretrained_to_classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODAL_ORDER = ('ABP', 'ECG', 'PPG', 'CVP', 'CO2', 'AWP')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True, type=str)
    p.add_argument('--data_dir', required=True, type=str)
    p.add_argument('--clinical_csv', required=True, type=str,
                   help='VitalDB clinical_data.csv (caseid, preop_cr, opend)')
    p.add_argument('--lab_csv', required=True, type=str,
                   help='VitalDB lab_data.csv (caseid, dt, name, result)')
    p.add_argument('--out_dir', default='results', type=str)
    p.add_argument('--window_sec', default=600.0, type=float,
                   help='Input window length (default 10 min)')
    p.add_argument('--stride_sec', default=300.0, type=float)
    p.add_argument('--train_fraction', default=0.80, type=float)
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--tag', default='hetero', type=str)
    p.add_argument('--max_subsets', type=int, default=0,
                   help='Cap on number of modal subsets to evaluate '
                        '(0 = all 2^N-1).')
    return p.parse_args()


def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    mask = spec >= target_spec
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def extract_features(model, samples: List[AKISample],
                     modal_subset: Tuple[str, ...],
                     batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    dataset = AKIDataset(samples, modal_order=list(modal_subset))
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


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading pretrained model...')
    classifier, _ = load_pretrained_to_classifier(args.ckpt_path, n_classes=2)
    classifier = classifier.to(device)
    classifier.eval()

    print('Loading AKI labels...')
    aki_labels = load_aki_labels(args.clinical_csv, args.lab_csv)
    print(f'  {len(aki_labels)} cases labeled')
    pos = sum(1 for v in aki_labels.values() if v.binary == 1)
    print(f'  AKI positive: {pos} ({pos / len(aki_labels) * 100:.1f}%)')

    print(f'Loading cases from {args.data_dir}...')
    all_cases = load_cases(
        data_dir=args.data_dir,
        input_signals=list(MODAL_ORDER),
        min_duration_sec=args.window_sec,
    )
    all_cases.sort(key=lambda c: c['case_id'])
    n_train = int(len(all_cases) * args.train_fraction)
    train_cases, test_cases = all_cases[:n_train], all_cases[n_train:]
    print(f'  Train: {len(train_cases)}  Test: {len(test_cases)}')

    kw = dict(
        aki_labels=aki_labels,
        input_signals=list(MODAL_ORDER),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
    train_samples = extract_aki_samples(train_cases, **kw)
    test_samples = extract_aki_samples(test_cases, **kw)
    print(f'  Train windows: {len(train_samples)}  Test windows: {len(test_samples)}')

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
        sens90 = sensitivity_at_specificity(test_y, prob)

        row = f'{subset_name},{auroc:.4f},{auprc:.4f},{sens90:.4f}'
        rows.append(row)
        print(row)

    csv_path = os.path.join(args.out_dir, f'{args.tag}_aki.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
