# -*- coding:utf-8 -*-
"""Phase-1 unimodal SSL via DINOv3-style self-distillation.

Reads the sharded SSL output written by ``dataset/data_parser/vital_db_ssl.py``,
filtered to a single modality (``ch_idx`` selects which one). Trains the
BiosignalDINO objective (DINO + optional iBOT) and saves both the full
checkpoint and a standalone encoder state-dict that Phase-2 can load directly.

Compared to the prior NeuroNet (TF-C) trainer this:
  * Drops the masked-recon decoder + L_T / L_F / L_TF heads completely.
  * Adds a multi-crop pipeline (configurable n_global / n_local).
  * Maintains an EMA teacher with cosine-scheduled momentum and a linearly
    warmed-up teacher temperature.
  * Keeps the dev-cohort probing path unchanged so per-epoch mode-collapse
    monitoring still works (probe consumes ``model.forward_latent``).
"""
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import argparse
import logging
import math
import random
import warnings
from typing import Optional, Tuple

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
from models.dino.model import (
    BiosignalDINO, cosine_schedule, linear_warmup,
)
from models.utils import model_size
from pretrained.dino.hetero_data_loader import (
    ShardSequentialSampler,
    ShardSingleModalDataset,
    load_holdout_case_ids,
    split_shards,
)
from pretrained.dino.multicrop import MultiCropConfig, make_crops
from pretrained.dino.augment import AugmentConfig
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
    parser.add_argument('--config_yaml', type=str,
                        default=os.path.join('..', '..', 'config', 'vital_db',
                                             'dino.yaml'))
    parser.add_argument('--ch_idx', type=int, default=None,
                        help='override ch_idx from yaml (0=ABP, ... 5=AWP)')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ssl_data_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--shard_cache_size', type=int, default=None)
    parser.add_argument('--prefetch_factor', type=int, default=None)
    parser.add_argument('--eager', action='store_const', const=True, default=None)
    parser.add_argument('--eager_workers', type=int, default=None)
    parser.add_argument('--no_amp', action='store_const', const=True, default=None)
    parser.add_argument('--holdout_subjects_file', type=str, default=None)
    parser.add_argument('--probe_downstream_dir', type=str, default=None)
    parser.add_argument('--probe_subjects_file', type=str, default=None)
    parser.add_argument('--probe_every', type=int, default=1)
    parser.add_argument('--smoke_test_shards', type=int, default=None,
                        help='cap to first N shards (split applies on top). '
                             'For verifying the loop without a multi-hour wait.')
    return parser.parse_args()


def load_config(path, overrides=None):
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config_dict[k] = v
    return argparse.Namespace(**config_dict)


CH_NAMES = ('ABP', 'ECG', 'PPG', 'CVP', 'CO2', 'AWP')


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.ch_idx is None or not (0 <= args.ch_idx < len(CH_NAMES)):
            raise ValueError(f'ch_idx must be in [0, {len(CH_NAMES)}); got {args.ch_idx}')
        self.ch_idx = int(args.ch_idx)
        self.modal_name = CH_NAMES[self.ch_idx]

        self.model = BiosignalDINO(
            fs=args.sfreq, second=args.second,
            time_window=args.time_window, time_step=args.time_step,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            dino_head_hidden_dim=int(getattr(args, 'dino_head_hidden_dim', 2048)),
            dino_head_bottleneck_dim=int(getattr(args, 'dino_head_bottleneck_dim', 256)),
            dino_head_n_prototypes=int(getattr(args, 'dino_head_n_prototypes', 8192)),
            dino_head_n_layers=int(getattr(args, 'dino_head_n_layers', 3)),
            ibot_head_hidden_dim=int(getattr(args, 'ibot_head_hidden_dim', 2048)),
            ibot_head_bottleneck_dim=int(getattr(args, 'ibot_head_bottleneck_dim', 256)),
            ibot_head_n_prototypes=int(getattr(args, 'ibot_head_n_prototypes', 8192)),
            ibot_head_n_layers=int(getattr(args, 'ibot_head_n_layers', 3)),
            student_temp=float(getattr(args, 'student_temp', 0.1)),
            teacher_temp=float(getattr(args, 'teacher_temp_end', 0.07)),
            dino_weight=float(getattr(args, 'dino_weight', 1.0)),
            ibot_weight=float(getattr(args, 'ibot_weight', 1.0)),
            ibot_mask_ratio_min=float(getattr(args, 'ibot_mask_ratio_min', 0.1)),
            ibot_mask_ratio_max=float(getattr(args, 'ibot_mask_ratio_max', 0.5)),
            ibot_mask_sample_probability=float(
                getattr(args, 'ibot_mask_sample_probability', 0.5),
            ),
            ibot_min_block_size=int(getattr(args, 'ibot_min_block_size', 1)),
            sinkhorn_n_iters=int(getattr(args, 'sinkhorn_n_iters', 3)),
        ).to(device)

        # Multi-crop config.
        self.n_global = int(getattr(args, 'n_global', 2))
        self.n_local = int(getattr(args, 'n_local', 6))
        self.crop_cfg = MultiCropConfig(
            n_global=self.n_global,
            n_local=self.n_local,
            global_samples=int(getattr(args, 'global_samples',
                                       args.sfreq * args.second)),
            local_samples=int(getattr(args, 'local_samples', args.sfreq * 15)),
        )

        # Optimisation. We exclude the teacher (no grad) because its params
        # have requires_grad=False, and we put the DINO/iBOT-head ``last_layer``
        # in a separate AdamW param group so we can zero its LR for the first
        # ``freeze_last_layer_epochs`` (DINOv3 stability trick — replaces the
        # weight_norm-magnitude freeze from v1/v2).
        last_layer_names = {
            'student_dino_head.last_layer.weight',
            'student_ibot_head.last_layer.weight',
        }
        head_params, last_layer_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name in last_layer_names:
                last_layer_params.append(p)
            else:
                head_params.append(p)
        self.eff_batch_size = args.train_batch_size * args.train_batch_accumulation
        self.lr = args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(
            [
                {'params': head_params, 'lr': self.lr},
                {'params': last_layer_params, 'lr': self.lr,
                 'is_last_layer': True},
            ],
            lr=self.lr,
            weight_decay=float(getattr(args, 'weight_decay', 0.04)),
        )
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.train_epochs,
        )
        self.clipping_norm_value = float(getattr(args, 'clip_grad', 3.0))
        self.freeze_last_layer_epochs = int(
            getattr(args, 'freeze_last_layer_epochs', 1),
        )

        # Shard split.
        from pretrained.dino.hetero_data_loader import _read_manifest
        manifest = _read_manifest(args.ssl_data_dir)
        self.num_shards = len(manifest['shards'])
        smoke_n = int(getattr(args, 'smoke_test_shards', 0) or 0)
        if smoke_n > 0 and smoke_n < self.num_shards:
            print(f'[smoke_test] capping to first {smoke_n}/{self.num_shards} '
                  f'shards (split_shards applies on top)')
            self.num_shards = smoke_n
        self.train_shards, self.val_shards, self.eval_shards = split_shards(
            num_shards=self.num_shards,
            val_ratio=getattr(args, 'val_shard_ratio', 0.1),
            eval_ratio=getattr(args, 'eval_shard_ratio', 0.1),
        )

        self.logger = self._build_logger()
        self._print_banner()
        self._build_probe()

    # ─────────────────────────────────────────────────────────────
    # Startup banner / probe / logger
    # ─────────────────────────────────────────────────────────────

    def _print_banner(self):
        print('[BiosignalDINO Parameter]')
        print(f'   >> Device     : {device}')
        if device.type == 'cuda':
            print(f'   >> GPU Name   : {torch.cuda.get_device_name(device)}')
        print(f'   >> Model Size : {model_size(self.model):.2f}MB')
        print(f'   >> Modal Name : {self.modal_name}')
        print(f'   >> Data Dir   : {self.args.ssl_data_dir}')
        print(f'   >> Learning Rate : {self.lr}')
        print(f'   >> Shards : {self.num_shards} total -> '
              f'train {len(self.train_shards)} / val {len(self.val_shards)} / '
              f'eval {len(self.eval_shards)}')
        print(f'   >> Workers : {self.args.num_workers} / '
              f'shard_cache_size : {self.args.shard_cache_size}')
        holdout_path = getattr(self.args, 'holdout_subjects_file', None)
        print(f'   >> Holdout : {holdout_path or "<none>"}')
        if bool(getattr(self.args, 'eager', False)):
            print(f'   >> Eager   : True (workers={self.args.eager_workers})')
        print(f'   >> DINO    : n_global={self.n_global} n_local={self.n_local} '
              f'global_samples={self.crop_cfg.global_samples} '
              f'local_samples={self.crop_cfg.local_samples}')
        print(f'   >> Head    : prototypes={self.model.n_prototypes} '
              f'ibot_weight={self.model.ibot_weight} '
              f'ibot_mask_ratio={self.model.ibot_mask_ratio}')

    def _build_logger(self) -> logging.Logger:
        log_dir = os.path.join(self.args.ckpt_path, self.args.model_name,
                               self.modal_name, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'train.log')

        logger = logging.getLogger(f'dino.{self.modal_name}.{id(self)}')
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

    def _build_probe(self):
        self.probe_train_loader = None
        self.probe_eval_loader = None
        self.probe_task_key = ''
        probe_dir = getattr(self.args, 'probe_downstream_dir', None)
        probe_subj = getattr(self.args, 'probe_subjects_file', None)
        if not (probe_dir and probe_subj):
            return
        modal_upper = self.modal_name.upper()
        task_key = PHASE1_PROBE_TASK_FOR_MODAL.get(modal_upper, '')
        self.probe_task_key = task_key
        if not task_key:
            print(f'   >> Probe   : skipped (no probe task for {self.modal_name})')
            return
        if task_key == 'ioh':
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
            print(f'             samples: train={len(tr_s)} eval={len(ev_s)} '
                  f'probe_every={self.args.probe_every}')
        else:
            print(f'   >> Probe   : self-modality forecast ({task_key}) -> {probe_subj}')
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
                print(f'             samples: train={len(tr_s)} eval={len(ev_s)} '
                      f'probe_every={self.args.probe_every}')
            except RuntimeError as e:
                print(f'             [warn] modality probe disabled: {e}')

    # ─────────────────────────────────────────────────────────────
    # DataLoader
    # ─────────────────────────────────────────────────────────────

    def _build_loader(self, shard_indices, shuffle: bool, drop_last: bool) -> DataLoader:
        holdout = load_holdout_case_ids(
            getattr(self.args, 'holdout_subjects_file', None),
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
        if shuffle:
            sampler = ShardSequentialSampler(
                dataset, seed=getattr(self.args, 'seed', 0),
                shuffle_shards=True,
            )
            loader_kwargs = dict(sampler=sampler)
        else:
            loader_kwargs = dict(shuffle=False)
        if num_workers > 0:
            loader_kwargs.update(
                num_workers=num_workers,
                persistent_workers=True,
                prefetch_factor=int(getattr(self.args, 'prefetch_factor', 4)),
            )
        else:
            loader_kwargs.update(num_workers=0)
        return DataLoader(
            dataset, batch_size=self.args.train_batch_size,
            drop_last=drop_last, pin_memory=torch.cuda.is_available(),
            **loader_kwargs,
        )

    # ─────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────

    def train(self):
        train_dataloader = self._build_loader(
            self.train_shards, shuffle=True, drop_last=True,
        )
        val_dataloader = self._build_loader(
            self.val_shards, shuffle=False, drop_last=False,
        )
        print(f'   >> Train segments : {len(train_dataloader.dataset)}')
        print(f'   >> Val   segments : {len(val_dataloader.dataset)}\n')

        amp_enabled = (device.type == 'cuda'
                       and not bool(getattr(self.args, 'no_amp', False)))
        if amp_enabled:
            print('   >> AMP     : bf16 autocast')

        # Estimate total steps for the schedules.
        steps_per_epoch = max(1, len(train_dataloader))
        total_steps = steps_per_epoch * int(self.args.train_epochs)
        warmup_steps_temp = steps_per_epoch * int(
            getattr(self.args, 'teacher_temp_warmup_epochs', 5),
        )
        ema_start = float(getattr(self.args, 'ema_momentum_start', 0.994))
        ema_end = float(getattr(self.args, 'ema_momentum_end', 1.0))
        temp_start = float(getattr(self.args, 'teacher_temp_start', 0.04))
        temp_end = float(getattr(self.args, 'teacher_temp_end', 0.07))

        total_step = 0
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(self.args.train_epochs):
            sampler = getattr(train_dataloader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

            # DINOv3 stability trick: freeze prototype-layer LR for the first
            # ``freeze_last_layer_epochs`` epochs (default 1). Implemented as
            # a zero-LR window on the dedicated optimizer param group, NOT a
            # weight_norm magnitude freeze.
            for pg in self.optimizer.param_groups:
                if pg.get('is_last_layer'):
                    pg['lr'] = 0.0 if epoch < self.freeze_last_layer_epochs \
                        else self.optimizer.param_groups[0]['lr']

            self.model.train()
            self.optimizer.zero_grad()
            step = 0
            for x, _ in train_dataloader:
                x = x.to(device, non_blocking=True).float()

                # Schedule updates each step.
                self.model.teacher_temp = linear_warmup(
                    temp_start, temp_end, total_step, warmup_steps_temp,
                )

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=amp_enabled):
                    globals_, locals_ = make_crops(x, self.crop_cfg)
                    loss, logs = self.model(
                        globals_, locals_, self.n_global, self.n_local,
                    )

                loss.backward()
                if (step + 1) % self.args.train_batch_accumulation == 0:
                    if self.clipping_norm_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.clipping_norm_value,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # EMA teacher update on optimizer step boundary.
                    mom = cosine_schedule(
                        ema_start, ema_end, total_step, total_steps,
                    )
                    self.model.update_teacher(mom)

                if (total_step + 1) % self.args.print_point == 0:
                    print(
                        f'[Epoch] : {epoch:03d}  [Step] : {total_step + 1:08d}  '
                        f'[DINO] : {logs["dino"].item():.4f}  '
                        f'[iBOT] : {logs["ibot"].item():.4f}  '
                        f'[Total] : {loss.item():.4f}  '
                        f'[t_temp] : {self.model.teacher_temp:.4f}'
                    )

                self.logger.info(
                    'train step=%08d epoch=%03d dino=%.6f ibot=%.6f total=%.6f '
                    't_temp=%.4f ema=%.4f',
                    total_step, epoch,
                    float(logs['dino']), float(logs['ibot']), float(loss),
                    float(self.model.teacher_temp),
                    float(cosine_schedule(ema_start, ema_end, total_step, total_steps)),
                )

                step += 1
                total_step += 1

            self.scheduler.step()

            # Validation: average DINO+iBOT loss on held-out shards.
            val_loss, val_dino, val_ibot = self.evaluate(val_dataloader, amp_enabled)
            self.logger.info(
                'val step=%08d epoch=%03d val_total=%.6f val_dino=%.6f val_ibot=%.6f',
                total_step, epoch, val_loss, val_dino, val_ibot,
            )
            print(f'[Epoch] : {epoch:03d} \t [Val Total] : {val_loss:.4f} '
                  f'\t [Val DINO] : {val_dino:.4f} \t [Val iBOT] : {val_ibot:.4f}\n')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self._save(best_state, epoch, total_step, tag='best')

            # Dev probe.
            if self.probe_train_loader is not None \
                    and (epoch + 1) % max(1, int(self.args.probe_every)) == 0:
                auroc, mf1 = self._run_probe()
                print(f'[Epoch] : {epoch:03d} \t [Probe {self.probe_task_key.upper()}] '
                      f'AUROC : {auroc:.4f} \t Macro-F1 : {mf1:.4f}\n')
                self.logger.info(
                    'probe step=%08d epoch=%03d task=%s '
                    'probe_auroc=%.4f probe_macro_f1=%.4f',
                    total_step, epoch, self.probe_task_key, auroc, mf1,
                )

        self._save(self.model.state_dict(), self.args.train_epochs - 1, total_step,
                   tag='last')

    # ─────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, amp_enabled: bool
                 ) -> Tuple[float, float, float]:
        self.model.eval()
        totals = [0.0, 0.0, 0.0]
        n = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True).float()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=amp_enabled):
                globals_, locals_ = make_crops(x, self.crop_cfg)
                loss, logs = self.model(
                    globals_, locals_, self.n_global, self.n_local,
                )
            bsz = x.shape[0]
            totals[0] += float(loss) * bsz
            totals[1] += float(logs['dino']) * bsz
            totals[2] += float(logs['ibot']) * bsz
            n += bsz
        self.model.train()
        if n == 0:
            return float('nan'), float('nan'), float('nan')
        return totals[0] / n, totals[1] / n, totals[2] / n

    # ─────────────────────────────────────────────────────────────
    # Probe (NaN-tolerant)
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_probe_latents(self, loader: DataLoader):
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
            return float('nan'), float('nan')
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(tr_x)
        ev_x = scaler.transform(ev_x)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 random_state=0)
        clf.fit(tr_x, tr_y)
        prob = clf.predict_proba(ev_x)[:, 1]
        pred = clf.predict(ev_x)
        return float(roc_auc_score(ev_y, prob)), float(f1_score(ev_y, pred, average='macro'))

    # ─────────────────────────────────────────────────────────────
    # Checkpoint save
    # ─────────────────────────────────────────────────────────────

    def _save(self, state_dict, epoch: int, step: int, tag: str) -> None:
        """Save full + encoder-only states. Phase-2 reads ``encoder_state``."""
        save_dir = os.path.join(self.args.ckpt_path, self.args.model_name,
                                self.modal_name, 'model')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{tag}_model.pth')

        # Extract encoder-only sub-state (keys prefixed ``student_encoder.``).
        prefix = 'student_encoder.'
        encoder_state = {
            k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        model_parameter = dict(
            fs=self.args.sfreq, second=self.args.second,
            time_window=self.args.time_window, time_step=self.args.time_step,
            encoder_embed_dim=self.args.encoder_embed_dim,
            encoder_heads=self.args.encoder_heads,
            encoder_depths=self.args.encoder_depths,
        )
        torch.save({
            'model_state': state_dict,
            'encoder_state': encoder_state,
            'model_parameter': model_parameter,
            'epoch': int(epoch),
            'step': int(step),
            'modal_name': self.modal_name,
            'ch_idx': self.ch_idx,
            'ssl_kind': 'dinov3',
        }, save_path)


if __name__ == '__main__':
    cli = get_args()
    augments = load_config(cli.config_yaml, overrides={
        'ch_idx': cli.ch_idx,
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
        'smoke_test_shards': cli.smoke_test_shards,
    })
    trainer = Trainer(augments)
    trainer.train()
