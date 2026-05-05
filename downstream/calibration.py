# -*- coding:utf-8 -*-
"""Calibration analysis for PhysioME-Hetero IOH predictions.

Requires raw prediction files saved by run_ioh.py --save_preds.

Usage:
  python downstream/calibration.py \\
      --preds_dir results/preds \\
      --tag       hetero \\
      --task      ioh \\
      --out_dir   results/figures
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preds_dir', required=True,
                   help='Directory with *_ytrue.npy and *_yscore.npy files')
    p.add_argument('--tags', nargs='+', default=['hetero'],
                   help='Model tags to compare (e.g. hetero synth_only)')
    p.add_argument('--task', default='ioh', choices=['ioh', 'aki', 'mortality'])
    p.add_argument('--subset', default='ABP+ECG+PPG',
                   help='Modality subset to plot (default: full set)')
    p.add_argument('--n_bins', default=10, type=int)
    p.add_argument('--out_dir', default='results/figures')
    return p.parse_args()


# ── ECE ──────────────────────────────────────────────────────────


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray,
                               n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if not mask.any():
            continue
        frac_pos = y_true[mask].mean()
        mean_conf = y_score[mask].mean()
        ece += mask.sum() / n * abs(frac_pos - mean_conf)
    return float(ece)


# ── Calibration curve ────────────────────────────────────────────


def calibration_curve_data(y_true: np.ndarray, y_score: np.ndarray,
                            n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='uniform')
    return prob_pred, prob_true


# ── Main ─────────────────────────────────────────────────────────


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for tag, color in zip(args.tags, colors):
        ytrue_path = os.path.join(args.preds_dir,
                                  f'{tag}_{args.task}_{args.subset}_ytrue.npy')
        yscore_path = os.path.join(args.preds_dir,
                                   f'{tag}_{args.task}_{args.subset}_yscore.npy')
        if not os.path.exists(ytrue_path):
            print(f'SKIP {tag}: {ytrue_path} not found')
            continue

        y_true = np.load(ytrue_path)
        y_score = np.load(yscore_path)
        ece = expected_calibration_error(y_true, y_score, n_bins=args.n_bins)
        prob_pred, prob_true = calibration_curve_data(y_true, y_score, n_bins=args.n_bins)

        # Calibration plot
        axes[0].plot(prob_pred, prob_true, 's-', color=color, label=f'{tag} (ECE={ece:.3f})')

        # Score distribution
        axes[1].hist(y_score[y_true == 0], bins=30, alpha=0.5, color=color,
                     label=f'{tag} negative', density=True, linestyle='--')
        axes[1].hist(y_score[y_true == 1], bins=30, alpha=0.5, color=color,
                     label=f'{tag} positive', density=True, linestyle='-')

    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
    axes[0].set_xlabel('Mean predicted probability')
    axes[0].set_ylabel('Fraction of positives')
    axes[0].set_title(f'Calibration — {args.task.upper()} [{args.subset}]')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    axes[1].set_xlabel('Predicted score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Score distribution')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fname = f'calibration_{args.task}_{args.subset.replace("+", "_")}.pdf'
    fig_path = os.path.join(args.out_dir, fname)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {fig_path}')

    # ECE table
    print('\nECE summary:')
    for tag in args.tags:
        ytrue_path = os.path.join(args.preds_dir,
                                  f'{tag}_{args.task}_{args.subset}_ytrue.npy')
        if not os.path.exists(ytrue_path):
            continue
        y_true = np.load(ytrue_path)
        y_score = np.load(os.path.join(args.preds_dir,
                                       f'{tag}_{args.task}_{args.subset}_yscore.npy'))
        ece = expected_calibration_error(y_true, y_score, args.n_bins)
        print(f'  {tag}: ECE = {ece:.4f}')


if __name__ == '__main__':
    main()
