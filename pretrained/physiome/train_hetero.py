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
* Per-epoch linear probing still runs on the IOH-labeled set
  (``labeled_data_dir`` / original ``vital_db.py`` output) so that we have a
  consistent supervised signal to track during pretraining.

Original ``train.py`` is preserved unchanged so that it can be re-used as the
A1 ablation baseline (synthetic-only training on complete-modality data).
"""
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import argparse
import random
import shutil
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import yaml
from models.transformer import apply_lora
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.utils import group_cross_validation
from models.dp_neuronet.model import NeuroNet, NeuroNetEncoder
from models.physiome.model import PhysioME
from models.utils import model_size
from pretrained.physiome.data_loader import TorchDataset as LabeledTorchDataset
from pretrained.physiome.hetero_data_loader import (
    MODAL_ORDER,
    BucketBatchSampler,
    HeteroVitalDBDataset,
    find_ssl_npz_paths,
    hetero_collate_fn,
)
from pretrained.physiome.probe_utils import run_probe, select_probe_subsets


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

        # SSL pretraining paths (hetero parser output)
        ssl_paths_all = find_ssl_npz_paths(args.ssl_data_dir)
        if not ssl_paths_all:
            raise FileNotFoundError(
                f'No SSL npz files in {args.ssl_data_dir}. '
                f'Run dataset/data_parser/vital_db_ssl.py first.'
            )
        self.ssl_train_paths = ssl_paths_all

        # Labeled probe paths (original IOH parser output, complete-modality)
        self.labeled_train_paths, self.labeled_val_paths, self.labeled_eval_paths = \
            self._labeled_paths()

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

        self.tensorboard_path = os.path.join(self.args.ckpt_path, 'tensorboard')
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('[PhysioME-Hetero parameters]')
        print(f'   >> Modal Names      : {", ".join(self.ch_names)}')
        print(f'   >> Model Size       : {model_size(self.model):.2f} MB')
        print(f'   >> Learning rate    : {self.lr}')
        print(f'   >> SSL train cases  : {len(self.ssl_train_paths)}')
        print(f'   >> Labeled subjects : '
              f'{len(self.labeled_train_paths)} train / '
              f'{len(self.labeled_val_paths)} val / '
              f'{len(self.labeled_eval_paths)} eval')
        print(f'   >> restoration_only_on_complete = {self.args.restoration_only_on_complete}')

    # ------------------------------------------------------------------
    # Path setup
    # ------------------------------------------------------------------
    def _labeled_paths(self):
        paths = group_cross_validation(
            base_path=self.args.labeled_data_dir,
            test_size=self.args.test_size,
            holdout_subject_size=self.args.holdout_subject_size,
        )
        return paths['train_paths'], paths['val_paths'], paths['eval_paths']

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
        pretrained_model.load_state_dict(ckpt['model_state'])

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
        ssl_dataset = HeteroVitalDBDataset(
            self.ssl_train_paths,
            eager=self.args.dataloader_eager,
            normalize=self.args.dataloader_normalize,
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

        # Labeled probe data — complete-modality, IOH-labeled
        val_dataset = LabeledTorchDataset(
            paths=self.labeled_val_paths, ch_names=self.ch_names,
            sfreq=self.args.sfreq, rfreq=self.args.rfreq,
            scaler=self.args.data_scaler, downsampling=self.args.class_downsampling,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_batch_size)
        eval_dataset = LabeledTorchDataset(
            paths=self.labeled_eval_paths, ch_names=self.ch_names,
            sfreq=self.args.sfreq, rfreq=self.args.rfreq,
            scaler=self.args.data_scaler,
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size)

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

                self.tensorboard_writer.add_scalar('Inter Recon Loss', inter_recon, total_step)
                self.tensorboard_writer.add_scalar('Missing Recon Loss', missing_recon, total_step)
                self.tensorboard_writer.add_scalar('Cross Contra Loss', cross_contra, total_step)
                self.tensorboard_writer.add_scalar('Cross Contra Accuracy', cross_acc, total_step)
                self.tensorboard_writer.add_scalar('Total Loss', loss, total_step)
                self.tensorboard_writer.add_text(
                    'Bucket Pattern', batch['bucket_pattern'], total_step,
                )

                step += 1
                total_step += 1

            acc, mf1 = self.linear_probing(epoch, val_dataloader, eval_dataloader)
            self.tensorboard_writer.add_scalar('Validation Accuracy', acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', mf1, total_step)

            if mf1 > best_score:
                best_score = mf1
                best_state = self.model.state_dict()

            self.scheduler.step()

        self.save_ckpt(best_state)

    # ------------------------------------------------------------------
    # Validation: linear probing across modal subsets
    # ------------------------------------------------------------------
    def linear_probing(self, epoch: int, val_dataloader: DataLoader,
                       eval_dataloader: DataLoader) -> Tuple[float, float]:
        """Probe the frozen encoder via Logistic Regression on a sampled
        subset of the ``2^N - 1`` modality combinations.

        See ``probe_utils`` for the sampling rule (always include the full
        set + every single-modal subset, then random-sample up to
        ``probe_max_subsets``). This keeps probe runtime bounded even at
        N=4/5 while preserving the corner-case coverage that matters for
        hetero ablations.
        """
        self.model.eval()
        max_subsets = int(getattr(self.args, 'probe_max_subsets', 10))
        subsets = select_probe_subsets(
            self.ch_names, max_subsets=max_subsets, seed=epoch,
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
            for x, y in dataloader:
                # ``x`` shape from LabeledTorchDataset: [B, num_total_modals, T]
                # in MODAL_ORDER order.
                data = {
                    ch_name: x[:, self.ch_names.index(ch_name), :].squeeze().float().to(device)
                    for ch_name in modal_combination
                }
                latent = self.model.inference_missing_modality(data=data)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        self.model.train()
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
                'ssl_train_paths': self.ssl_train_paths,
                'labeled_train_paths': self.labeled_train_paths,
                'labeled_val_paths': self.labeled_val_paths,
                'labeled_eval_paths': self.labeled_eval_paths,
            },
        }, os.path.join(ckpt_path, 'best_model.pth'))


if __name__ == '__main__':
    args_ = get_args()
    cfg = load_config(args_.config_yaml)
    trainer = HeteroTrainer(cfg)
    trainer.train()
