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
import random
import shutil
import warnings

import mne
import numpy as np
import torch
import torch.optim as opt
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.dp_neuronet.model import NeuroNet
from models.utils import model_size
from pretrained.dp_neuronet.hetero_data_loader import (
    ShardSingleModalDataset,
    split_shards,
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
        self.train_shards, self.val_shards, self.eval_shards = split_shards(
            num_shards=self.num_shards,
            val_ratio=getattr(self.args, 'val_shard_ratio', 0.1),
            eval_ratio=getattr(self.args, 'eval_shard_ratio', 0.1),
        )

        self.tensorboard_path = os.path.join(self.args.ckpt_path, self.args.model_name,
                                             self.modal_name, 'tensorboard')
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('[NeuroNet Parameter]')
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Modal Name : {0}'.format(self.modal_name))
        print('   >> Frame Size : {0}'.format(self.model.num_patches))
        print('   >> Learning Rate : {0}'.format(self.lr))
        print('   >> Shards : {0} total -> train {1} / val {2} / eval {3}'.format(
            self.num_shards, len(self.train_shards),
            len(self.val_shards), len(self.eval_shards),
        ))

    def _build_loader(self, shard_indices, shuffle: bool, drop_last: bool) -> DataLoader:
        dataset = ShardSingleModalDataset(
            data_dir=self.args.ssl_data_dir,
            ch_idx=self.args.ch_idx,
            shard_indices=shard_indices,
            normalize=getattr(self.args, 'dataloader_normalize', True),
            shard_cache_size=getattr(self.args, 'shard_cache_size', 4),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=getattr(self.args, 'num_workers', 0),
            pin_memory=torch.cuda.is_available(),
        )
        return loader

    def train(self):
        train_dataloader = self._build_loader(self.train_shards, shuffle=True, drop_last=True)
        val_dataloader = self._build_loader(self.val_shards, shuffle=False, drop_last=False)

        print('   >> Train segments : {0}'.format(len(train_dataloader.dataset)))
        print('   >> Val   segments : {0}\n'.format(len(val_dataloader.dataset)))

        total_step = 0
        best_model_state, best_val_loss = self.model.state_dict(), float('inf')

        for epoch in range(self.args.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_dataloader:
                x = x.to(device)
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

                self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                self.tensorboard_writer.add_scalar('L_T (time-time)', l_t, total_step)
                self.tensorboard_writer.add_scalar('L_F (freq-freq)', l_f, total_step)
                self.tensorboard_writer.add_scalar('L_TF (cross-domain)', l_tf, total_step)
                self.tensorboard_writer.add_scalar('Total Loss', loss, total_step)

                step += 1
                total_step += 1

            # Validation: held-out TF-C loss (unsupervised — no labels).
            val_loss, val_recon = self.evaluate(val_dataloader)

            if val_loss < best_val_loss:
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_val_loss = val_loss

            print('[Epoch] : {0:03d} \t [Val Total] : {1:.4f} \t [Val Recon] : {2:.4f}\n'.format(
                epoch, val_loss, val_recon))
            self.tensorboard_writer.add_scalar('Val Total Loss', val_loss, total_step)
            self.tensorboard_writer.add_scalar('Val Recon Loss', val_recon, total_step)

            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

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
                                      'ckpt_path': cli.ckpt_path})
    trainer = Trainer(augments)
    trainer.train()
