# -*- coding:utf-8 -*-
"""Oracle information-bound analysis of hetero-bucket restoration.

Question:
    "restoration_only_on_complete=False 로 sparse bucket 까지 켜면 학습이
     어떻게 망가지는가?"

Instead of training a tiny PhysioME (compute-limited, signal hidden by
deep-network learning dynamics on a toy pool), we measure the
INFORMATION-THEORETIC UPPER BOUND on restoration quality given a bucket.

Setup:
  * Synthesize per-modal tokens with a known group-latent structure:
        cardiovascular (ABP/ECG/PPG/CVP): all derived from shared latent z_cv
        respiratory    (CO2/AWP):         both derived from shared latent z_rp
    Each modal token = proj_modal(z_group) + per_modal_noise.
  * For each bucket size k and each drop-modal m, fit a LINEAR REGRESSION
    from the concatenated context tokens (k-1 modals) to the dropped-modal
    token. The R^2 of this fit is the BEST POSSIBLE restoration any
    decoder (linear or non-linear) can achieve on this bucket pattern --
    a Cramer-Rao-style upper bound for the recon task.
  * Aggregate over all (k, drop_modal, bucket_pattern) combinations.

Pattern this reveals (PhysioME-Hetero context):
  * k=6 (drop 1 of 6 → 5 context): nearly all drop-modals retrievable from
    same-group context. Avg R^2 is high. Restoration loss is meaningful.
  * k=5: 4 context modals. Mostly OK except when the drop+context split
    leaves no same-group context for the drop modal.
  * k=3..2: many bucket patterns leave the drop modal with NO same-group
    context. Oracle R^2 ≈ 0 → restoration is information-theoretically
    impossible. Any gradient the decoder receives in these batches is
    noise — pushing the decoder weights in random directions. THIS is
    what "training breaks" means concretely.

Usage:
    python -m experiments.sim_sparse_bucket_restoration \
        --n_samples 2000 --out experiments/_out/sparse_bucket_sim.png
"""
from __future__ import annotations

import argparse
import itertools
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


MODAL_GROUPS = {
    'ABP': 'cardiovascular', 'ECG': 'cardiovascular',
    'PPG': 'cardiovascular', 'CVP': 'cardiovascular',
    'CO2': 'respiratory',    'AWP': 'respiratory',
}
MODAL_NAMES = list(MODAL_GROUPS.keys())


def generate_correlated_tokens(n_samples: int, embed_dim: int,
                               noise_std: float = 0.5, seed: int = 0
                               ) -> Dict[str, np.ndarray]:
    """Generate per-modal token matrix [n_samples, embed_dim] for each modal,
    with the following structure:
        cardiovascular modals share latent z_cv (n_samples, embed_dim);
        respiratory    modals share latent z_rp (n_samples, embed_dim);
        each modal applies a per-modal random orthogonal rotation and adds
        i.i.d. Gaussian noise.

    With noise_std small, all 4 cardio modals are nearly recoverable from
    each other; cardio-vs-respiratory pairs are mutually uninformative."""
    rng = np.random.default_rng(seed)
    latents = {
        'cardiovascular': rng.standard_normal((n_samples, embed_dim)),
        'respiratory':    rng.standard_normal((n_samples, embed_dim)),
    }
    tokens: Dict[str, np.ndarray] = {}
    for m in MODAL_NAMES:
        z = latents[MODAL_GROUPS[m]]
        # Per-modal random rotation so within-group modals are not identical.
        rot = rng.standard_normal((embed_dim, embed_dim)) / np.sqrt(embed_dim)
        token = z @ rot + rng.standard_normal((n_samples, embed_dim)) * noise_std
        tokens[m] = token
    return tokens


def oracle_recon_r2(tokens: Dict[str, np.ndarray], drop_modal: str,
                    context_modals: List[str]) -> float:
    """Linear-regression R^2 for predicting drop_modal's tokens from
    concatenated context_modals' tokens. Trained / evaluated on the SAME
    pool (oracle / in-distribution upper bound)."""
    if not context_modals:
        return 0.0
    X = np.concatenate([tokens[m] for m in context_modals], axis=1)   # [n, k*D]
    Y = tokens[drop_modal]                                            # [n, D]
    # Least-squares solve. Add tiny ridge for stability.
    XtX = X.T @ X
    XtX += np.eye(XtX.shape[0]) * 1e-3
    W = np.linalg.solve(XtX, X.T @ Y)
    Yhat = X @ W
    ss_res = float(((Y - Yhat) ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return r2


def enumerate_bucket_patterns(k: int) -> List[Tuple[str, ...]]:
    """All bucket patterns of exactly k real-present modals."""
    return list(itertools.combinations(MODAL_NAMES, k))


def main(args):
    tokens = generate_correlated_tokens(
        n_samples=args.n_samples, embed_dim=args.embed_dim,
        noise_std=args.noise_std, seed=args.seed,
    )
    print(f'tokens: per-modal shape = '
          f'{tokens[MODAL_NAMES[0]].shape}, noise_std={args.noise_std}')

    # For every bucket-size k, enumerate all C(6,k) patterns. For each pattern
    # try each modal in turn as the drop target; the rest is context.
    rows: List[Dict] = []
    for k in (6, 5, 4, 3, 2):
        for pattern in enumerate_bucket_patterns(k):
            for drop_m in pattern:
                context = [m for m in pattern if m != drop_m]
                if not context:
                    continue
                same_group_ctx = sum(1 for c in context
                                     if MODAL_GROUPS[c] == MODAL_GROUPS[drop_m])
                r2 = oracle_recon_r2(tokens, drop_m, context)
                rows.append({
                    'k': k,
                    'pattern': ''.join('1' if m in pattern else '0'
                                        for m in MODAL_NAMES),
                    'drop_modal': drop_m,
                    'drop_group': MODAL_GROUPS[drop_m],
                    'same_group_ctx': same_group_ctx,
                    'r2': r2,
                })

    # ---- aggregate ----
    print('\n=== Oracle restoration R^2 by bucket size ===')
    print('  k  |  patterns | trials | R^2 mean | R^2 std | %trials w/ R^2<0.1')
    print('  ---+-----------+--------+----------+---------+---------------------')
    by_k = defaultdict(list)
    for r in rows:
        by_k[r['k']].append(r['r2'])
    for k in (6, 5, 4, 3, 2):
        vals = np.array(by_k[k])
        n_patterns = len(enumerate_bucket_patterns(k))
        broken = float((vals < 0.1).mean()) * 100.0
        print(f'  {k}  |   {n_patterns:6d}  |  {len(vals):4d}  '
              f'|  {vals.mean():.3f}   |  {vals.std():.3f}  |  {broken:5.1f}%')

    print('\n=== Same-group context effect (k=2 only) ===')
    print('  same_group_ctx | trials | R^2 mean')
    print('  ---------------+--------+----------')
    by_sg = defaultdict(list)
    for r in rows:
        if r['k'] == 2:
            by_sg[r['same_group_ctx']].append(r['r2'])
    for sg in sorted(by_sg):
        vals = np.array(by_sg[sg])
        print(f'       {sg}        |  {len(vals):4d}  |  {vals.mean():.3f}')

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'matplotlib unavailable: {e}')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ks = [6, 5, 4, 3, 2]
    boxes = [by_k[k] for k in ks]
    axes[0].boxplot(boxes, tick_labels=[f'k={k}' for k in ks],
                    showmeans=True, meanline=True, widths=0.55)
    axes[0].set_ylabel('oracle R^2  (best-case restoration)')
    axes[0].set_xlabel('# real-present modalities in bucket')
    axes[0].set_title('Restoration ceiling vs bucket sparsity')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0.1, color='red', linestyle='--', linewidth=0.8,
                    alpha=0.6, label='R^2 = 0.1 (≈ noise)')
    axes[0].legend()

    # Right panel: fraction of (k, pattern, drop) configurations where the
    # oracle is below 0.1 -- i.e. fraction of batches where the decoder gets
    # only noise signal.
    frac = [float((np.array(by_k[k]) < 0.1).mean()) * 100.0 for k in ks]
    axes[1].bar([f'k={k}' for k in ks], frac, color='crimson', alpha=0.7)
    axes[1].set_ylabel('% configurations with R^2 < 0.1')
    axes[1].set_xlabel('# real-present modalities in bucket')
    axes[1].set_title('Fraction of "training-noise" batches\n'
                      '(decoder receives no recoverable signal)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(frac):
        axes[1].text(i, v + 1.5, f'{v:.0f}%', ha='center', fontsize=9)

    fig.suptitle(
        'Oracle restoration ceiling vs bucket sparsity '
        f'(n_samples={args.n_samples}, embed_dim={args.embed_dim}, '
        f'noise_std={args.noise_std})',
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f'\nplot saved -> {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_samples', type=int, default=2000)
    p.add_argument('--embed_dim', type=int, default=32)
    p.add_argument('--noise_std', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--out', type=str,
                   default='experiments/_out/sparse_bucket_sim.png')
    main(p.parse_args())
