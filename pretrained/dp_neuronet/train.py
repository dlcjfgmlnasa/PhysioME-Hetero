# -*- coding:utf-8 -*-
"""Phase-1 NeuroNet (TF-C) pretraining — single modality at a time.

Reads the sharded SSL output written by ``dataset/data_parser/vital_db_ssl.py``,
filtered to a single modality (``ch_idx`` selects which one). Trains the
unimodal NeuroNet TF-C objective (recon + L_T + L_F + L_TF) and saves the
best-loss checkpoint.

Compared to the previous lookahead-coupled per-case loader, this version:
  * Uses ALL valid SSL segments (no IOH-label filter).
  * Reads the shard layout once via manifest -> O(1) init even on slow NFS.
  * Drops the KNN linear-probe validation (no labels in SSL data); selects
    the best epoch by held-out total TF-C loss instead. Phase-2 hetero +
    downstream IOH probe remain the authoritative quality signal.
"""
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import argparse
import logging
import random
import warnings
from typing import Tuple

import mne
import numpy as np
import torch
import torch.optim as opt
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from downstream.tasks.hypotension import HypotensionDataset
from downstream.tasks.modality_forecast import ModalityForecastDataset
from models.dp_neuronet.model import NeuroNet
from models.utils import model_size
from pretrained.dp_neuronet.hetero_data_loader import (
    ShardSequentialSampler,
    ShardSingleModalDataset,
    load_holdout_case_ids,
    split_shards,
)
from pretrained.probe_dev_data import (
    PHASE1_PROBE_TASK_FOR_MODAL,
    load_dev_probe_modality_split,
    load_dev_probe_split,
)


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml',
                        type=str,
                        default=os.path.join('..', '..', 'config', 'vital_db', 'dp_neuronet.yaml'))
    parser.add_argument('--ch_idx', type=int, default=None,
                        help='override ch_idx from yaml (0=ABP, 1=ECG, 2=PPG, 3=CVP)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='override ckpt_path from yaml')
    parser.add_argument('--ssl_data_dir', type=str, default=None,
                        help='override ssl_data_dir from yaml '
                             '(e.g. point at a local-SSD copy of the shards)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='override num_workers from yaml')
    parser.add_argument('--shard_cache_size', type=int, default=None,
                        help='override shard_cache_size from yaml')
    parser.add_argument('--prefetch_factor', type=int, default=None,
                        help='override DataLoader prefetch_factor from yaml')
    parser.add_argument('--eager', action='store_const', const=True, default=None,
                        help='pre-load all (modality, kept-segment) slices into '
                             'RAM at init (one parallel shard sweep). Eliminates '
                             'per-batch network I/O at the cost of ~13-20 GB RAM '
                             'per modality.')
    parser.add_argument('--eager_workers', type=int, default=None,
                        help='parallel shard readers used by --eager (default 8)')
    parser.add_argument('--no_amp', action='store_const', const=True, default=None,
                        help='disable bf16 autocast (default: AMP on for cuda).')
    parser.add_argument('--holdout_subjects_file', type=str, default=None,
                        help='override holdout_subjects_file from yaml '
                             '(JSON written by sample_holdout). Excludes '
                             'those case_ids from Phase-1 SSL training.')
    parser.add_argument('--probe_downstream_dir', type=str, default=None,
                        help='vitaldb_downstream npz dir for IOH probing. '
                             'If given together with --probe_subjects_file, '
                             'enables periodic linear probing on the dev '
                             'cohort to monitor learning progress.')
    parser.add_argument('--probe_subjects_file', type=str, default=None,
                        help='dev_case_ids.json used as the probe cohort. '
                             'Must be disjoint from holdout_subjects_file '
                             '(verify_split_disjoint.py I1).')
    parser.add_argument('--probe_every', type=int, default=1,
                        help='Run dev probing every N epochs (default 1).')
    parser.add_argument('--viz_every', type=int, default=None,
                        help='Save a reconstruction-quality figure (real vs '
                             'predicted waveform on a fixed val batch) every '
                             'N epochs. 0 disables. Default reads yaml '
                             '(viz_every, default 5).')
    parser.add_argument('--viz_mask_ratio', type=float, default=None,
                        help='Mask ratio used ONLY for the viz forward (does '
                             'not affect training). Lower than the training '
                             'mask_ratio so the figure has enough visible '
                             'context to be readable. Default yaml '
                             '(viz_mask_ratio, default 0.5).')
    parser.add_argument('--smoke_test_shards', type=int, default=None,
                        help='Smoke-test mode: cap to the first N shards from '
                             'manifest (train/val/eval split then applies on '
                             'top). 0 / unset = use all shards. Recommended '
                             'N>=3 (one per split). Use to verify the loop '
                             'runs end-to-end without a multi-hour wait.')
    return parser.parse_args()


def load_config(path, overrides=None):
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config_dict[k] = v
    return argparse.Namespace(**config_dict)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.modal_name = args.ch_names[args.ch_idx]

        self.model = NeuroNet(
            fs=args.rfreq, second=args.second,
            time_window=args.time_window, time_step=args.time_step,
            encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            projection_hidden=args.projection_hidden, temperature=args.temperature
        ).to(device)

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)

        # Shard split (deterministic, contiguous trailing slices for val/eval).
        from pretrained.dp_neuronet.hetero_data_loader import _read_manifest
        manifest = _read_manifest(self.args.ssl_data_dir)
        self.num_shards = len(manifest['shards'])
        smoke_n = int(getattr(self.args, 'smoke_test_shards', 0) or 0)
        if smoke_n > 0 and smoke_n < self.num_shards:
            print(f'[smoke_test] capping to first {smoke_n}/{self.num_shards} '
                  f'shards (split_shards applies on top)')
            self.num_shards = smoke_n
        self.train_shards, self.val_shards, self.eval_shards = split_shards(
            num_shards=self.num_shards,
            val_ratio=getattr(self.args, 'val_shard_ratio', 0.1),
            eval_ratio=getattr(self.args, 'eval_shard_ratio', 0.1),
        )

        self.logger = self._build_logger()

        print('[NeuroNet Parameter]')
        print('   >> Device     : {0}'.format(device))
        if device.type == 'cuda':
            print('   >> GPU Name   : {0}'.format(torch.cuda.get_device_name(device)))
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Modal Name : {0}'.format(self.modal_name))
        print('   >> Data Dir   : {0}'.format(self.args.ssl_data_dir))
        print('   >> Frame Size : {0}'.format(self.model.num_patches))
        print('   >> Learning Rate : {0}'.format(self.lr))
        print('   >> Shards : {0} total -> train {1} / val {2} / eval {3}'.format(
            self.num_shards, len(self.train_shards),
            len(self.val_shards), len(self.eval_shards),
        ))
        print('   >> Workers : {0} / shard_cache_size : {1}'.format(
            getattr(self.args, 'num_workers', 0),
            getattr(self.args, 'shard_cache_size', 4),
        ))
        holdout_path = getattr(self.args, 'holdout_subjects_file', None)
        print('   >> Holdout : {0}'.format(holdout_path or '<none>'))
        eager = bool(getattr(self.args, 'eager', False))
        if eager:
            print('   >> Eager   : True (workers={0})'.format(
                int(getattr(self.args, 'eager_workers', 8) or 8)
            ))

        # ── Dev-cohort linear probing (optional) ──────────────────
        # Per-modality dispatch (see PHASE1_PROBE_TASK_FOR_MODAL):
        #   ABP/ECG/PPG → IOH probe (MAP-based, sustained <65 mmHg in 5 min).
        #   CVP        → venous-congestion forecast (mean CVP > 12 mmHg).
        #   CO2        → hypercapnia forecast (mean EtCO2 > 50 mmHg).
        #   AWP        → high-airway-pressure forecast (peak AWP > 30 cmH2O).
        # Each modality gets a probe so SSL training quality is monitored
        # epoch-by-epoch instead of guessed from raw TF-C loss.
        self.probe_train_loader = None
        self.probe_eval_loader = None
        self.probe_task_key: str = ''
        probe_dir = getattr(self.args, 'probe_downstream_dir', None)
        probe_subj = getattr(self.args, 'probe_subjects_file', None)
        if probe_dir and probe_subj:
            modal_upper = self.modal_name.upper()
            task_key = PHASE1_PROBE_TASK_FOR_MODAL.get(modal_upper, '')
            self.probe_task_key = task_key
            if not task_key:
                print(f'   >> Probe   : skipped (no probe task registered '
                      f'for modal {self.modal_name})')
            elif task_key == 'ioh':
                print(f'   >> Probe   : IOH (dev cohort) -> {probe_subj}')
                tr_s, ev_s = load_dev_probe_split(
                    downstream_dir=probe_dir,
                    dev_subjects_file=probe_subj,
                    input_signals=(modal_upper,),
                )
                self.probe_train_loader = DataLoader(
                    HypotensionDataset(tr_s, modal_order=[modal_upper]),
                    batch_size=self.args.train_batch_size, shuffle=False,
                )
                self.probe_eval_loader = DataLoader(
                    HypotensionDataset(ev_s, modal_order=[modal_upper]),
                    batch_size=self.args.train_batch_size, shuffle=False,
                )
                print(f'             samples: train={len(tr_s)} '
                      f'eval={len(ev_s)}  probe_every={self.args.probe_every}')
            else:
                print(f'   >> Probe   : self-modality forecast '
                      f'({task_key}) -> {probe_subj}')
                try:
                    tr_s, ev_s = load_dev_probe_modality_split(
                        downstream_dir=probe_dir,
                        dev_subjects_file=probe_subj,
                        task_key=task_key,
                    )
                    self.probe_train_loader = DataLoader(
                        ModalityForecastDataset(tr_s, modal_order=[modal_upper]),
                        batch_size=self.args.train_batch_size, shuffle=False,
                    )
                    self.probe_eval_loader = DataLoader(
                        ModalityForecastDataset(ev_s, modal_order=[modal_upper]),
                        batch_size=self.args.train_batch_size, shuffle=False,
                    )
                    print(f'             samples: train={len(tr_s)} '
                          f'eval={len(ev_s)}  probe_every={self.args.probe_every}')
                except RuntimeError as e:
                    # Dev cohort doesn't include this modality on enough
                    # subjects — log and continue without probing rather
                    # than crash the SSL run.
                    print(f'             [warn] modality probe disabled: {e}')
                    self.probe_train_loader = None
                    self.probe_eval_loader = None

    def _build_logger(self) -> logging.Logger:
        log_dir = os.path.join(self.args.ckpt_path, self.args.model_name,
                               self.modal_name, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'train.log')

        logger = logging.getLogger(f'dp_neuronet.{self.modal_name}.{id(self)}')
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)

        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
        ))
        logger.addHandler(fh)
        return logger

    def _build_loader(self, shard_indices, shuffle: bool, drop_last: bool) -> DataLoader:
        holdout = load_holdout_case_ids(
            getattr(self.args, 'holdout_subjects_file', None)
        )
        dataset = ShardSingleModalDataset(
            data_dir=self.args.ssl_data_dir,
            ch_idx=self.args.ch_idx,
            shard_indices=shard_indices,
            normalize=getattr(self.args, 'dataloader_normalize', True),
            shard_cache_size=getattr(self.args, 'shard_cache_size', 4),
            exclude_case_ids=holdout,
            eager=bool(getattr(self.args, 'eager', False)),
            eager_workers=int(getattr(self.args, 'eager_workers', 8) or 8),
        )
        num_workers = int(getattr(self.args, 'num_workers', 0) or 0)

        # Sequential-shard sampler so each shard is loaded once per epoch.
        # Critical when shards live on a slow / network filesystem -- random
        # segment shuffling there yields ~0% cache hits and pegs the GPU at
        # 0% util. Sampler shuffles shard *order* across epochs and segment
        # order within each shard, preserving training stochasticity.
        if shuffle:
            sampler = ShardSequentialSampler(
                dataset, seed=getattr(self.args, 'seed', 0),
                shuffle_shards=True,
            )
            loader_kwargs = dict(sampler=sampler)
        else:
            loader_kwargs = dict(shuffle=False)

        # persistent_workers + prefetch_factor lets workers keep their shard
        # cache hot across epochs and overlap network I/O with GPU compute.
        if num_workers > 0:
            loader_kwargs.update(
                num_workers=num_workers,
                persistent_workers=True,
                prefetch_factor=int(getattr(self.args, 'prefetch_factor', 4)),
            )
        else:
            loader_kwargs.update(num_workers=0)

        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
            **loader_kwargs,
        )

    def train(self):
        train_dataloader = self._build_loader(self.train_shards, shuffle=True, drop_last=True)
        val_dataloader = self._build_loader(self.val_shards, shuffle=False, drop_last=False)

        print('   >> Train segments : {0}'.format(len(train_dataloader.dataset)))
        print('   >> Val   segments : {0}\n'.format(len(val_dataloader.dataset)))

        # Capture a fixed visualization batch (first 4 val segments) so that
        # epoch-over-epoch reconstruction plots compare the same waveforms.
        # Cached on CPU to keep GPU memory free during training.
        self._viz_batch = None
        viz_every = int(getattr(self.args, 'viz_every', 5) or 0)
        if viz_every > 0:
            for vx, _ in val_dataloader:
                self._viz_batch = vx[:4].detach().clone()
                break
            if self._viz_batch is not None:
                print('   >> Viz     : every {0} epoch(s), {1} sample(s) -> '
                      '<ckpt>/logs/recon/'.format(
                          viz_every, self._viz_batch.shape[0]))
            else:
                print('   >> Viz     : disabled (val loader produced no batch)')
        self._viz_every = viz_every

        # bf16 autocast on CUDA (no GradScaler needed: bf16 has FP32 dynamic
        # range). Cuts step time ~1.5-2x on L40S/Ampere+ vs FP32. Set --no_amp
        # to fall back to FP32 (e.g. for debugging numerical issues).
        amp_enabled = (device.type == 'cuda'
                       and not bool(getattr(self.args, 'no_amp', False)))
        if amp_enabled:
            print('   >> AMP     : bf16 autocast')

        total_step = 0
        best_model_state, best_val_loss = self.model.state_dict(), float('inf')

        for epoch in range(self.args.train_epochs):
            # Reseed the sampler so shard order + intra-shard shuffle change.
            sampler = getattr(train_dataloader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_dataloader:
                x = x.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=amp_enabled):
                    recon_loss, l_t, l_f, l_tf = self.model(x, mask_ratio=self.args.mask_ratio)
                    loss = recon_loss + l_t + l_f + l_tf
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:08d}  '
                          '[Recon] : {2:02.4f}  '
                          '[L_T] : {3:02.4f}  '
                          '[L_F] : {4:02.4f}  '
                          '[L_TF] : {5:02.4f}  '
                          '[Total] : {6:02.4f}'.format(
                            epoch, total_step + 1, recon_loss, l_t, l_f, l_tf, loss))

                self.logger.info(
                    'train step=%08d epoch=%03d '
                    'recon=%.6f l_t=%.6f l_f=%.6f l_tf=%.6f total=%.6f',
                    total_step, epoch,
                    float(recon_loss), float(l_t), float(l_f),
                    float(l_tf), float(loss),
                )

                step += 1
                total_step += 1

            # Validation: held-out TF-C loss (unsupervised — no labels).
            val_loss, val_recon = self.evaluate(val_dataloader)

            if val_loss < best_val_loss:
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_val_loss = val_loss

            print('[Epoch] : {0:03d} \t [Val Total] : {1:.4f} \t [Val Recon] : {2:.4f}\n'.format(
                epoch, val_loss, val_recon))
            self.logger.info(
                'val step=%08d epoch=%03d val_total=%.6f val_recon=%.6f',
                total_step, epoch, float(val_loss), float(val_recon),
            )

            if self.probe_train_loader is not None \
                    and (epoch + 1) % max(1, int(self.args.probe_every)) == 0:
                auroc, mf1 = self._run_probe()
                print('[Epoch] : {0:03d} \t [Probe {3}] AUROC : {1:.4f} '
                      '\t Macro-F1 : {2:.4f}\n'.format(
                        epoch, auroc, mf1, self.probe_task_key.upper()))
                self.logger.info(
                    'probe step=%08d epoch=%03d task=%s '
                    'probe_auroc=%.4f probe_macro_f1=%.4f',
                    total_step, epoch, self.probe_task_key, auroc, mf1,
                )

            if self._viz_batch is not None and self._viz_every > 0 \
                    and (epoch + 1) % self._viz_every == 0:
                self._save_recon_viz(epoch, total_step)

            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

    @torch.no_grad()
    def _extract_probe_latents(self, loader: DataLoader):
        """Forward dev probe samples through frozen backbone, return (X, y)."""
        self.model.eval()
        feats, labels = [], []
        for data, y in loader:
            x = data[self.modal_name].to(device).float()
            z = self.model.forward_latent(x)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            feats.append(z.detach().cpu().numpy())
            labels.append(y.numpy())
        self.model.train()
        if not feats:
            return np.zeros((0, 1)), np.zeros((0,), dtype=np.int64)
        return np.concatenate(feats, 0), np.concatenate(labels, 0)

    def _run_probe(self) -> Tuple[float, float]:
        """One LR-probe pass on the dev cohort. Returns (auroc, macro_f1).

        Latents can contain NaN when the dev npz has NaN-padded segments
        (artifact-rejected regions) or when the still-warming encoder
        underflows on a degenerate input. We drop those rows rather than
        crashing the whole epoch — the probe is a monitoring tool, not a
        training signal, so partial coverage with a warning is the right
        trade-off. If too many rows drop we return NaN metrics.
        """
        tr_x, tr_y = self._extract_probe_latents(self.probe_train_loader)
        ev_x, ev_y = self._extract_probe_latents(self.probe_eval_loader)

        def _drop_nan_rows(x, y, split: str):
            keep = np.isfinite(x).all(axis=1)
            n_drop = int((~keep).sum())
            if n_drop:
                self.logger.info(
                    'probe_nan split=%s dropped=%d of=%d',
                    split, n_drop, x.shape[0],
                )
            return x[keep], y[keep]

        tr_x, tr_y = _drop_nan_rows(tr_x, tr_y, 'train')
        ev_x, ev_y = _drop_nan_rows(ev_x, ev_y, 'eval')

        if len(tr_y) == 0 or len(ev_y) == 0 or len(set(tr_y)) < 2 \
                or len(set(ev_y)) < 2:
            # Degenerate probe (empty after NaN drop, or single class on
            # either side) — skip metric but keep training.
            return float('nan'), float('nan')
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(tr_x)
        ev_x = scaler.transform(ev_x)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 random_state=0)
        clf.fit(tr_x, tr_y)
        prob = clf.predict_proba(ev_x)[:, 1]
        pred = clf.predict(ev_x)
        auroc = float(roc_auc_score(ev_y, prob))
        mf1 = float(f1_score(ev_y, pred, average='macro'))
        return auroc, mf1

    def _save_recon_viz(self, epoch: int, total_step: int) -> None:
        """Save a real-vs-reconstructed waveform plot for the fixed viz batch.

        Output path:
            <ckpt>/<model_name>/<modal>/logs/recon/epoch_NNN.png

        Masked patches are shaded so the reader can tell which segments the
        decoder had to *reconstruct* vs which it just passed through. Plot
        uses Agg backend so it works headless on KHDP / over SSH.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            self.logger.info('viz step=%08d epoch=%03d skipped reason=%s',
                             total_step, epoch, repr(e))
            return

        x = self._viz_batch.to(device).float()
        viz_mr = float(getattr(self.args, 'viz_mask_ratio', 0.5) or 0.5)
        real, pred, mask = self.model.forward_recon_time(
            x, mask_ratio=viz_mr,
        )
        real = real.float().cpu().numpy()         # (B, F, W)
        pred = pred.float().cpu().numpy()
        mask = mask.float().cpu().numpy()         # (B, F)

        b, f, w = real.shape
        fs = float(self.args.rfreq)
        # Stitch frames -> full window. Assumes time_step == time_window
        # (non-overlap). For overlap configs the plot still shows correct
        # per-frame reconstruction but the stitched signal would duplicate
        # overlapped samples; non-overlap is the v2 default.
        t_axis = np.arange(f * w) / fs

        fig, axes = plt.subplots(b, 1, figsize=(12, 2.2 * b), sharex=True)
        if b == 1:
            axes = [axes]
        for i in range(b):
            ax = axes[i]
            ax.plot(t_axis, real[i].reshape(-1), color='steelblue',
                    linewidth=0.9, label='real')
            ax.plot(t_axis, pred[i].reshape(-1), color='crimson',
                    linewidth=0.7, linestyle='--', label='recon')
            for fi in range(f):
                if mask[i, fi] > 0.5:
                    ax.axvspan(fi * w / fs, (fi + 1) * w / fs,
                               color='gold', alpha=0.15, linewidth=0)
            ax.grid(alpha=0.3)
            ax.set_ylabel(f'sample {i}')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[0].set_title(
            f'{self.modal_name}  recon @ epoch {epoch:03d}  '
            f'(viz_mask_ratio={viz_mr}, train_mask_ratio='
            f'{self.args.mask_ratio}, gold = masked patches)'
        )
        axes[-1].set_xlabel('time (s)')

        viz_dir = os.path.join(self.args.ckpt_path, self.args.model_name,
                               self.modal_name, 'logs', 'recon')
        os.makedirs(viz_dir, exist_ok=True)
        out = os.path.join(viz_dir, f'epoch_{epoch:03d}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        self.logger.info('viz step=%08d epoch=%03d saved=%s',
                         total_step, epoch, out)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss, total_recon, n = 0.0, 0.0, 0
        for x, _ in dataloader:
            x = x.to(device)
            recon_loss, l_t, l_f, l_tf = self.model(x, mask_ratio=self.args.mask_ratio)
            bsz = x.shape[0]
            total_loss += float(recon_loss + l_t + l_f + l_tf) * bsz
            total_recon += float(recon_loss) * bsz
            n += bsz
        self.model.train()
        if n == 0:
            return float('nan'), float('nan')
        return total_loss / n, total_recon / n

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(self.args.ckpt_path, self.args.model_name,
                                 self.modal_name, 'model')
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save({
            'model_name': 'DP-NeuroNet',
            'model_state': model_state,
            'model_parameter': {
                'fs': self.args.rfreq, 'second': self.args.second,
                'time_window': self.args.time_window, 'time_step': self.args.time_step,
                'encoder_embed_dim': self.args.encoder_embed_dim,
                'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim,
                'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'projection_hidden': self.args.projection_hidden,
                'temperature': self.args.temperature,
            },
            'hyperparameter': self.args.__dict__,
            'shards': {
                'train_shards': self.train_shards,
                'val_shards': self.val_shards,
                'eval_shards': self.eval_shards,
                'num_shards': self.num_shards,
            },
        }, os.path.join(ckpt_path, 'best_model.pth'))


if __name__ == '__main__':
    cli = get_args()
    augments = load_config(path=cli.config_yaml,
                           overrides={'ch_idx': cli.ch_idx,
                                      'ckpt_path': cli.ckpt_path,
                                      'ssl_data_dir': cli.ssl_data_dir,
                                      'num_workers': cli.num_workers,
                                      'shard_cache_size': cli.shard_cache_size,
                                      'prefetch_factor': cli.prefetch_factor,
                                      'eager': cli.eager,
                                      'eager_workers': cli.eager_workers,
                                      'no_amp': cli.no_amp,
                                      'holdout_subjects_file': cli.holdout_subjects_file,
                                      'probe_downstream_dir': cli.probe_downstream_dir,
                                      'probe_subjects_file': cli.probe_subjects_file,
                                      'probe_every': cli.probe_every,
                                      'viz_every': cli.viz_every,
                                      'viz_mask_ratio': cli.viz_mask_ratio,
                                      'smoke_test_shards': cli.smoke_test_shards})
    trainer = Trainer(augments)
    trainer.train()
