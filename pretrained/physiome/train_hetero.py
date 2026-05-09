# -*- coding:utf-8 -*-
"""PhysioME-Hetero SSL pretraining on VitalDB.

This is the hetero-availability counterpart to ``pretrained/physiome/train.py``.
Differences:

* SSL training data is read from the *hetero* parser output
  (``vital_db_ssl.py``) — every case contributes whatever subset of
  {ABP, ECG, PPG} it actually has, plus per-segment validity masks.
* Batches are sampled via :class:`BucketBatchSampler` so each batch is
  homogeneous in modality availability.
* The :class:`PhysioME` forward receives a per-sample
  ``presence_state`` tensor and the ``restoration_only_on_complete`` toggle
  for ablation A3.
* Per-epoch linear probing runs on the *dev cohort* — case_ids listed in
  ``probe_subjects_file`` (= dev_case_ids.json), windowed for the IOH task
  via ``downstream/tasks/hypotension.py``. The dev cohort is disjoint from
  both SSL training and the downstream test (= holdout) cohort, so probing
  cannot leak into reported test metrics.

Original ``train.py`` is preserved unchanged so that it can be re-used as the
A1 ablation baseline (synthetic-only training on complete-modality data).
"""
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import argparse
import logging
import random
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import yaml
from models.transformer import apply_lora
from torch.utils.data import DataLoader

from downstream.tasks.hypotension import HypotensionDataset
from models.dp_neuronet.model import NeuroNet, NeuroNetEncoder
from models.physiome.model import PhysioME
from models.utils import model_size
from pretrained.physiome.hetero_data_loader import (
    MODAL_ORDER,
    BucketBatchSampler,
    HeteroVitalDBDataset,
    find_ssl_data_dir,
    hetero_collate_fn,
    load_holdout_case_ids,
)
from pretrained.physiome.probe_utils import run_probe, select_probe_subsets
from pretrained.probe_dev_data import load_dev_probe_split


# vital_db_downstream.py only emits ABP/ECG/PPG; CVP is skipped at probe
# time even when the model itself is 4-modal. The dropped channel is
# transparently filled by ``inference_missing_modality``.
PROBE_MODALITIES = ('ABP', 'ECG', 'PPG')


warnings.filterwarnings(action='ignore')
mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str,
                        default=os.path.join('..', '..', 'config', 'vital_db',
                                             'physiome_hetero.yaml'))
    return parser.parse_args()


def load_config(path: str) -> argparse.Namespace:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HeteroTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        set_seed(args.seed)

        self.ch_names: List[str] = list(MODAL_ORDER)

        # SSL pretraining shards (hetero parser output) — manifest sanity check.
        find_ssl_data_dir(args.ssl_data_dir)  # raises with friendly message
        self.ssl_data_dir = args.ssl_data_dir

        # Probe cohort = dev_case_ids.json (disjoint from holdout/test).
        # Loader construction is deferred to ``train()`` so that __init__
        # remains side-effect-free w.r.t. the downstream npz dir.
        self.probe_ch_names: List[str] = [
            m for m in self.ch_names if m in PROBE_MODALITIES
        ]

        self.model = PhysioME(
            backbone_networks=self._encoder_backbones(),
            backbone_embed_dim=args.backbone_embed_dim,
            num_backbone_frames=args.backbone_num_frames,
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            decoder_recon_depths=args.decoder_recon_depths,
            projection_hidden=args.projection_hidden,
            temperature=args.temperature,
        ).to(device)

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)
        self.clipping_norm_value = 2.0

        self.logger = self._build_logger()

        print('[PhysioME-Hetero parameters]')
        print(f'   >> Modal Names      : {", ".join(self.ch_names)}')
        print(f'   >> Model Size       : {model_size(self.model):.2f} MB')
        print(f'   >> Learning rate    : {self.lr}')
        print(f'   >> SSL data dir     : {self.ssl_data_dir}')
        print(f'   >> Holdout subjects : '
              f'{getattr(self.args, "holdout_subjects_file", None) or "<none>"}')
        print(f'   >> Probe cohort     : '
              f'{getattr(self.args, "probe_subjects_file", None) or "<none>"}')
        print(f'   >> Probe modalities : {", ".join(self.probe_ch_names)}')
        print(f'   >> restoration_only_on_complete = {self.args.restoration_only_on_complete}')

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    def _build_logger(self) -> logging.Logger:
        log_dir = os.path.join(self.args.ckpt_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'train.log')

        logger = logging.getLogger(f'physiome_hetero.{id(self)}')
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

    # ------------------------------------------------------------------
    # Probe loader setup (dev cohort, IOH labels)
    # ------------------------------------------------------------------
    def _build_probe_loaders(self) -> Tuple[Optional[DataLoader],
                                            Optional[DataLoader]]:
        probe_dir = getattr(self.args, 'probe_downstream_dir', None)
        probe_subj = getattr(self.args, 'probe_subjects_file', None)
        if not (probe_dir and probe_subj):
            print('[probe] disabled — set probe_downstream_dir + '
                  'probe_subjects_file in yaml to enable.')
            return None, None
        train_s, eval_s = load_dev_probe_split(
            downstream_dir=probe_dir,
            dev_subjects_file=probe_subj,
            input_signals=tuple(self.probe_ch_names),
        )
        train_loader = DataLoader(
            HypotensionDataset(train_s, modal_order=self.probe_ch_names),
            batch_size=self.args.train_batch_size, shuffle=False,
        )
        eval_loader = DataLoader(
            HypotensionDataset(eval_s, modal_order=self.probe_ch_names),
            batch_size=self.args.train_batch_size, shuffle=False,
        )
        print(f'[probe] dev samples: train={len(train_s)} eval={len(eval_s)}')
        return train_loader, eval_loader

    # ------------------------------------------------------------------
    # Encoder backbones (LoRA-wrapped NeuroNet encoders, same as train.py)
    # ------------------------------------------------------------------
    def _encoder_backbones(self) -> Dict[str, nn.Module]:
        return {
            ch_name: self._load_pretrained_unimodal(ckpt_path)
            for ch_name, ckpt_path in self.args.modality_name2path.items()
        }

    def _load_pretrained_unimodal(self, ckpt_path: str) -> nn.Module:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_parameter = ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        # strict=False: tolerates Phase-1 ckpts saved before the MAE decoder
        # was migrated from timm.Block to BFM TransformerEncoder. The encoder
        # / frame_backbone / cls_token weights -- the only ones Phase-2 actually
        # transfers -- are present in both layouts; only the decoder keys
        # differ and Phase-2 discards the decoder anyway.
        pretrained_model.load_state_dict(ckpt['model_state'], strict=False)

        backbone = NeuroNetEncoder(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            encoder_embed_dim=model_parameter['encoder_embed_dim'],
            encoder_heads=model_parameter['encoder_heads'],
            encoder_depths=model_parameter['encoder_depths'],
        )
        # Direct submodule transfer — NeuroNet ↔ NeuroNetEncoder share the same
        # patch_embed / encoder / cls_token shape; no name-substring matching needed.
        backbone.frame_backbone.load_state_dict(pretrained_model.frame_backbone.state_dict())
        backbone.patch_embed.load_state_dict(pretrained_model.autoencoder.patch_embed.state_dict())
        backbone.encoder.load_state_dict(pretrained_model.autoencoder.encoder.state_dict())
        backbone.cls_token = pretrained_model.autoencoder.cls_token

        # Hand-rolled LoRA on GQA's ``out_proj`` (drops the peft dep + cleans
        # up state-dict keys). See ``models/transformer/lora.py``.
        backbone = apply_lora(
            backbone,
            target_attrs=('out_proj',),
            r=self.args.lora_r,
            alpha=self.args.lora_alpha,
            dropout=self.args.lora_dropout,
            rslora=True,
        )
        backbone.to(device)
        return backbone

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        holdout = load_holdout_case_ids(
            getattr(self.args, 'holdout_subjects_file', None)
        )
        ssl_dataset = HeteroVitalDBDataset(
            self.ssl_data_dir,
            eager=self.args.dataloader_eager,
            normalize=self.args.dataloader_normalize,
            shard_cache_size=getattr(self.args, 'shard_cache_size', 4),
            exclude_case_ids=holdout,
        )
        self._ssl_dataset_stats = (
            ssl_dataset.num_shards, len(ssl_dataset), ssl_dataset.num_cases,
        )
        sampler = BucketBatchSampler(
            ssl_dataset.segment_bitmap_keys,
            batch_size=self.args.train_batch_size,
            sampling=self.args.bucket_sampling,
            min_bucket_size=self.args.bucket_min_size,
            shuffle=True, drop_last=True, seed=self.args.seed,
        )
        ssl_loader = DataLoader(
            ssl_dataset, batch_sampler=sampler,
            collate_fn=hetero_collate_fn,
            num_workers=self.args.num_workers, pin_memory=True,
        )
        print(f'[BucketBatchSampler] active buckets: {sorted(sampler.buckets.keys())} '
              f'sizes={[len(sampler.buckets[k]) for k in sorted(sampler.buckets.keys())]}')

        # Dev-cohort probe data (IOH labels, ABP/ECG/PPG only).
        val_dataloader, eval_dataloader = self._build_probe_loaders()
        probe_every = max(1, int(getattr(self.args, 'probe_every', 1)))

        total_step = 0
        best_state, best_score = self.model.state_dict(), 0.0

        for epoch in range(self.args.train_epochs):
            sampler.set_epoch(epoch)
            self.model.train()
            self.optimizer.zero_grad()

            step = 0
            for batch in ssl_loader:
                data = {k: v.to(device).float() for k, v in batch['data'].items()}
                presence_state = batch['presence_state'].to(device)

                inter_recon, missing_recon, cross_contra, cross_acc = self.model(
                    data=data, presence_state=presence_state,
                    mask_ratio=self.args.mask_ratio,
                    restoration_only_on_complete=self.args.restoration_only_on_complete,
                )
                loss = inter_recon + missing_recon + cross_contra
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print(f'[Epoch] : {epoch:03d}  [Step] : {total_step + 1:07d}  '
                          f'[bucket] : {batch["bucket_pattern"]} '
                          f'[Inter] : {inter_recon:2.4f}  [Miss] : {missing_recon:2.4f}  '
                          f'[Contra] : {cross_contra:2.4f}  [Acc] : {cross_acc:2.3f}  '
                          f'[Total] : {loss:2.3f}')

                self.logger.info(
                    'train step=%07d epoch=%03d bucket=%s '
                    'inter_recon=%.6f missing_recon=%.6f '
                    'cross_contra=%.6f cross_acc=%.4f total=%.6f',
                    total_step, epoch, batch['bucket_pattern'],
                    float(inter_recon), float(missing_recon),
                    float(cross_contra), float(cross_acc), float(loss),
                )

                step += 1
                total_step += 1

            if val_dataloader is not None and eval_dataloader is not None \
                    and (epoch + 1) % probe_every == 0:
                acc, mf1 = self.linear_probing(epoch, val_dataloader, eval_dataloader)
                self.logger.info(
                    'probe step=%07d epoch=%03d val_acc=%.4f val_macro_f1=%.4f',
                    total_step, epoch, float(acc), float(mf1),
                )
                if mf1 > best_score:
                    best_score = mf1
                    best_state = self.model.state_dict()
            else:
                # Probe disabled — keep latest state as best.
                best_state = self.model.state_dict()

            self.scheduler.step()

        self.save_ckpt(best_state)

    # ------------------------------------------------------------------
    # Validation: linear probing across modal subsets
    # ------------------------------------------------------------------
    def linear_probing(self, epoch: int, val_dataloader: DataLoader,
                       eval_dataloader: DataLoader) -> Tuple[float, float]:
        """Probe the frozen encoder via Logistic Regression on a sampled
        subset of the modality combinations available in the dev probe set.

        Subsets are drawn from ``self.probe_ch_names`` (= ABP/ECG/PPG) since
        ``vital_db_downstream.py`` does not emit CVP windows. The 4-modal
        case is still represented at inference time via
        ``inference_missing_modality`` filling in CVP with the dropped token.
        """
        self.model.eval()
        max_subsets = int(getattr(self.args, 'probe_max_subsets', 10))
        subsets = select_probe_subsets(
            self.probe_ch_names, max_subsets=max_subsets, seed=epoch,
        )

        train_fn = lambda subset: self._latent_vector(subset, val_dataloader)
        eval_fn = lambda subset: self._latent_vector(subset, eval_dataloader)
        mean_acc, mean_mf1, _ = run_probe(
            subsets, train_fn, eval_fn,
            log_prefix=f'[Epoch {epoch:03d}]',
        )
        self.model.train()
        return mean_acc, mean_mf1

    def _latent_vector(self, modal_combination: Tuple[str, ...],
                       dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        total_x, total_y = [], []
        with torch.no_grad():
            for data_dict, y in dataloader:
                # HypotensionDataset yields {modal_name: [B, T] tensor}.
                # Drop modalities not in this subset; keep only requested ones.
                data = {
                    ch_name: data_dict[ch_name].to(device).float()
                    for ch_name in modal_combination if ch_name in data_dict
                }
                if not data:
                    continue
                latent = self.model.inference_missing_modality(data=data)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        self.model.train()
        if not total_x:
            return np.zeros((0, 1)), np.zeros((0,), dtype=np.int64)
        return np.concatenate(total_x, 0), np.concatenate(total_y, 0)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_ckpt(self, model_state: Dict[str, torch.Tensor]) -> None:
        ckpt_path = os.path.join(self.args.ckpt_path, 'model')
        os.makedirs(ckpt_path, exist_ok=True)

        backbone_ckpt = torch.load(
            list(self.args.modality_name2path.values())[0], map_location='cpu',
        )
        backbone_parameter = backbone_ckpt['model_parameter']

        torch.save({
            'model_name': 'PhysioME-Hetero',
            'ch_names': self.ch_names,
            'modality_backbone_param': {
                'fs': backbone_parameter['fs'],
                'second': backbone_parameter['second'],
                'time_window': backbone_parameter['time_window'],
                'time_step': backbone_parameter['time_step'],
                'encoder_embed_dim': backbone_parameter['encoder_embed_dim'],
                'encoder_heads': backbone_parameter['encoder_heads'],
                'encoder_depths': backbone_parameter['encoder_depths'],
            },
            'entire_model_param': {
                'backbone_embed_dim': self.args.backbone_embed_dim,
                'backbone_num_frames': self.args.backbone_num_frames,
                'encoder_embed_dim': self.args.encoder_embed_dim,
                'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim,
                'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'decoder_recon_depths': self.args.decoder_recon_depths,
                'projection_hidden': self.args.projection_hidden,
                'temperature': self.args.temperature,
            },
            'model_state': model_state,
            'hyperparameter': self.args.__dict__,
            'paths': {
                'ssl_data_dir': self.ssl_data_dir,
                'holdout_subjects_file': getattr(
                    self.args, 'holdout_subjects_file', None),
                'probe_subjects_file': getattr(
                    self.args, 'probe_subjects_file', None),
                'probe_downstream_dir': getattr(
                    self.args, 'probe_downstream_dir', None),
            },
        }, os.path.join(ckpt_path, 'best_model.pth'))


if __name__ == '__main__':
    args_ = get_args()
    cfg = load_config(args_.config_yaml)
    trainer = HeteroTrainer(cfg)
    trainer.train()
