# -*- coding:utf-8 -*-
"""Linear-probe utilities for PhysioME / PhysioME-Hetero training loops.

The original ``linear_probing`` enumerated **all 2^N - 1 modality subsets**
and trained a fresh SVC on each, every epoch. That is fine at N=3 (7 subsets)
but does not scale: at N=4 it becomes 15 subsets, at N=5 it is 31. The SVC
itself is also O(n^2)-O(n^3) in fitting time, so the per-epoch cost grows
faster than the modality count.

This module replaces both bottlenecks:

* ``select_probe_subsets`` — keep "must-include" subsets (the full set and
  each single-modal subset), then random-sample the remainder up to
  ``max_subsets``. Coverage of the corner cases stays guaranteed while the
  runtime stops scaling exponentially.
* ``run_probe`` — drop SVC for ``LogisticRegression`` (≈10x faster fitting
  on probe-scale features) and run the encoder forward exactly once per
  (subset, dataloader). Returns the per-subset acc/macro-F1 plus aggregate
  means.

Drop-in: ``linear_probing`` in train.py / train_hetero.py becomes a thin
wrapper that calls ``run_probe`` with a closure that maps a ``modal_subset``
to ``(latent, label)`` tensors.
"""
from __future__ import annotations

import random
from itertools import combinations
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


ModalSubset = Tuple[str, ...]
LatentFn = Callable[[ModalSubset], Tuple[np.ndarray, np.ndarray]]


def select_probe_subsets(ch_names: Sequence[str], max_subsets: int = 10,
                         seed: Optional[int] = None) -> List[ModalSubset]:
    """Pick a representative set of modality subsets to probe.

    Always includes:
      * the full set ``(ch_names[0], ..., ch_names[-1])``
      * every single-modal subset ``(ch,)``
    Then random-samples additional subsets (without replacement) from the
    remaining ``2^N - 1`` until ``max_subsets`` is reached.

    For N <= 3 the full enumeration fits within ``max_subsets``, so this
    function falls back to returning every subset and is fully backwards-
    compatible with the prior all-subset behaviour.
    """
    n = len(ch_names)
    all_subsets: List[ModalSubset] = []
    for r in range(1, n + 1):
        all_subsets.extend(tuple(c) for c in combinations(ch_names, r))
    if len(all_subsets) <= max_subsets:
        return all_subsets

    full = tuple(ch_names)
    singles = [(ch,) for ch in ch_names]
    must_include: List[ModalSubset] = [full] + [s for s in singles if s != full]

    extras = [s for s in all_subsets if s not in must_include]
    n_extras = max(0, max_subsets - len(must_include))
    if n_extras > 0 and extras:
        rng = random.Random(seed)
        sampled_extras = rng.sample(extras, min(n_extras, len(extras)))
    else:
        sampled_extras = []

    return must_include + sampled_extras


def run_probe(modal_subsets: Iterable[ModalSubset],
              train_latent_fn: LatentFn, eval_latent_fn: LatentFn,
              *,
              max_iter: int = 1000, C: float = 1.0,
              log_prefix: str = '') -> Tuple[float, float, List[Tuple[ModalSubset, float, float]]]:
    """Train a LogisticRegression probe per subset on (train, eval) latents.

    ``train_latent_fn`` and ``eval_latent_fn`` each take a modal subset and
    return ``(features [N, D], labels [N])``. They are responsible for
    running the (frozen) encoder under ``torch.no_grad`` and moving the
    resulting tensors to numpy on CPU.

    Returns ``(mean_acc, mean_macro_f1, per_subset_records)`` where each
    record is ``(modal_subset, acc, macro_f1)`` so the caller can log them.
    """
    accs: List[float] = []
    mf1s: List[float] = []
    records: List[Tuple[ModalSubset, float, float]] = []
    for subset in modal_subsets:
        train_x, train_y = train_latent_fn(subset)
        test_x, test_y = eval_latent_fn(subset)

        clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1)
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)

        acc = float(accuracy_score(test_y, pred_y))
        mf1 = float(f1_score(test_y, pred_y, average='macro'))
        accs.append(acc)
        mf1s.append(mf1)
        records.append((subset, acc, mf1))

        if log_prefix:
            print(f'{log_prefix} [{",".join(subset):<24}] '
                  f'=> Acc {acc * 100:7.4f}  Macro-F1 {mf1 * 100:7.4f}')

    if not accs:
        return 0.0, 0.0, []
    return float(np.mean(accs)), float(np.mean(mf1s)), records
