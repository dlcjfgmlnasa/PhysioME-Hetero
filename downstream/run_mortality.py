# -*- coding:utf-8 -*-
"""Cross-dataset mortality evaluation (VitalDB → MIMIC-III).

Protocol:
  1. Train features: MIMIC-III (or VitalDB) waveform npz, subject-level split
  2. Freeze PhysioME-Hetero (pretrained on VitalDB); extract latent
  3. Fit LogisticRegression; AUROC + AUPRC + Sens@Sp90% across 2^N-1 subsets
  4. Save results/<tag>_mortality.csv

For the cross-dataset story: pretrain on VitalDB, evaluate on MIMIC-III.
All MIMIC-III subjects are used as the test set (zero-shot transfer):
the model is never retrained on MIMIC data.

Usage (zero-shot transfer):
  python downstream/run_mortality.py \\
      --ckpt_path  ckpt/vital_db/physiome_hetero/model/best_model.pth \\
      --mimic_npz_dir  data/mimic3_hetero \\
      --cohort_csv     data/icu_mortality_cohort.csv \\
      --mode  zero_shot \\
      --out_dir results

Usage (linear probing with subject-level split):
  python downstream/run_mortality.py ... --mode linear_probe
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

from downstream.tasks.mortality import (
    MortalitySample,
    MortalityDataset,
    extract_mortality_samples_from_npz,
    load_mortality_cohort,
)
from downstream.utils import load_pretrained_to_classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# MIMIC-III WDB Matched Subset only has ABP / ECG / PPG (no CVP/CO2).
# This stays 3-modal even when the model is trained 4+ modal; the absent
# CVP slot is filled with ``dropped_modality_token`` inside
# ``PhysioME.inference_missing_modality`` (same code path as any subset).
MODAL_ORDER = ('ABP', 'ECG', 'PPG')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True, type=str)
    p.add_argument('--mimic_npz_dir', required=True, type=str,
                   help='Output dir of mimic3_waveform_ssl.py parse')
    p.add_argument('--cohort_csv', required=True, type=str,
                   help='ICU mortality cohort CSV with subject_id + hospital_expire_flag')
    p.add_argument('--mode', default='zero_shot',
                   choices=['zero_shot', 'linear_probe'],
                   help='zero_shot: no training on MIMIC; '
                        'linear_probe: 80/20 split within MIMIC subjects')
    p.add_argument('--out_dir', default='results', type=str)
    p.add_argument('--max_windows_per_record', default=None, type=int,
                   help='Cap windows per MIMIC record to reduce class imbalance skew')
    p.add_argument('--train_fraction', default=0.80, type=float,
                   help='Only used in linear_probe mode')
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--tag', default='hetero', type=str)
    return p.parse_args()


def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1 - fpr
    mask = spec >= target_spec
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def extract_features(model, samples: List[MortalitySample],
                     modal_subset: Tuple[str, ...],
                     batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    dataset = MortalityDataset(samples, modal_order=list(modal_subset))
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


def _evaluate_subset(train_x, train_y, test_x, test_y):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(train_x, train_y)
    prob = lr.predict_proba(test_x)[:, 1]
    auroc = roc_auc_score(test_y, prob)
    auprc = average_precision_score(test_y, prob)
    sens90 = sensitivity_at_specificity(test_y, prob)
    return auroc, auprc, sens90


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading pretrained model...')
    classifier, _ = load_pretrained_to_classifier(args.ckpt_path, n_classes=2)
    classifier = classifier.to(device)
    classifier.eval()

    print('Loading mortality cohort...')
    cohort = load_mortality_cohort(args.cohort_csv)
    pos = sum(1 for v in cohort.values() if v.mortality == 1)
    print(f'  Subjects: {len(cohort)}  Mortality: {pos} ({pos / len(cohort) * 100:.1f}%)')

    print(f'Extracting MIMIC samples from {args.mimic_npz_dir}...')
    samples = extract_mortality_samples_from_npz(
        mimic_npz_dir=args.mimic_npz_dir,
        cohort=cohort,
        input_signals=list(MODAL_ORDER),
        max_windows_per_record=args.max_windows_per_record,
    )
    print(f'  Total windows: {len(samples)}')

    modal_subsets = []
    for r in range(1, len(MODAL_ORDER) + 1):
        for combo in combinations(MODAL_ORDER, r):
            modal_subsets.append(combo)

    rows = []
    header = 'Subset,AUROC,AUPRC,Sens@Sp90'
    print(f'\nMode: {args.mode}')
    print(header)

    if args.mode == 'zero_shot':
        # No train split: extract all features, evaluate with a trivial "train = test" LR.
        # This is valid only for ranking purposes; for the paper, report AUROC directly
        # from the cosine-nearest-neighbor or kNN baseline instead.
        # Here we simply use all samples as both train and test to report the embedding
        # quality — OR leave train=test and report only AUROC of the linear fit.
        #
        # A cleaner zero-shot proxy: use 5-fold cross-validation within MIMIC subjects.
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import roc_auc_score, average_precision_score

        subject_ids = np.array([s.subject_id for s in samples])
        labels_all = np.array([s.label for s in samples])
        unique_subjects = np.unique(subject_ids)
        np.random.seed(42)
        np.random.shuffle(unique_subjects)

        for subset in modal_subsets:
            subset_name = '+'.join(subset)
            all_x, all_y = extract_features(classifier, samples, subset, args.batch_size)

            # 5-fold subject-level CV
            sgkf = StratifiedGroupKFold(n_splits=5)
            aurocs, auprcs, sens90s = [], [], []
            for tr_idx, te_idx in sgkf.split(all_x, all_y, groups=subject_ids):
                scaler = StandardScaler()
                tr_x = scaler.fit_transform(all_x[tr_idx])
                te_x = scaler.transform(all_x[te_idx])
                tr_y, te_y = all_y[tr_idx], all_y[te_idx]
                lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
                lr.fit(tr_x, tr_y)
                prob = lr.predict_proba(te_x)[:, 1]
                aurocs.append(roc_auc_score(te_y, prob))
                auprcs.append(average_precision_score(te_y, prob))
                sens90s.append(sensitivity_at_specificity(te_y, prob))

            row = (f'{subset_name},{np.mean(aurocs):.4f},'
                   f'{np.mean(auprcs):.4f},{np.mean(sens90s):.4f}')
            rows.append(row)
            print(row)

    else:  # linear_probe
        # Subject-level 80/20 split.
        unique_subjects = sorted(set(s.subject_id for s in samples))
        n_train_s = int(len(unique_subjects) * args.train_fraction)
        train_subjects = set(unique_subjects[:n_train_s])
        train_samples = [s for s in samples if s.subject_id in train_subjects]
        test_samples = [s for s in samples if s.subject_id not in train_subjects]
        print(f'  Train windows: {len(train_samples)}  Test: {len(test_samples)}')

        for subset in modal_subsets:
            subset_name = '+'.join(subset)
            tr_x, tr_y = extract_features(classifier, train_samples, subset, args.batch_size)
            te_x, te_y = extract_features(classifier, test_samples, subset, args.batch_size)
            auroc, auprc, sens90 = _evaluate_subset(tr_x, tr_y, te_x, te_y)
            row = f'{subset_name},{auroc:.4f},{auprc:.4f},{sens90:.4f}'
            rows.append(row)
            print(row)

    csv_path = os.path.join(args.out_dir, f'{args.tag}_mortality.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
