# -*- coding:utf-8 -*-
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import logging
import mne
import torch
import yaml
import random
import warnings
import argparse
import numpy as np
import torch.optim as opt
from models.utils import model_size
from dataset.utils import group_cross_validation
from models.dp_neuronet.model import NeuroNet, NeuroNetEncoder
from models.physiome.model import PhysioME
from pretrained.physiome.data_loader import TorchDataset
from pretrained.physiome.probe_utils import run_probe, select_probe_subsets
from torch.utils.data import DataLoader
from models.transformer import apply_lora


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
                        default=os.path.join('..', '..', 'config', 'vital_db', 'physiome.yaml'))
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        return argparse.Namespace(**config_dict)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_paths, self.val_paths, self.eval_paths = self.data_paths()
        self.ch_names = list(self.args.modality_name2path.keys())
        self.model = PhysioME(
            backbone_networks=self.get_encoder_backbone(),
            backbone_embed_dim=args.backbone_embed_dim, num_backbone_frames=args.backbone_num_frames,
            encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            decoder_recon_depths=args.decoder_recon_depths,
            projection_hidden=args.projection_hidden, temperature=args.temperature,
        ).to(device)

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)
        self.clipping_norm_value = 2.0
        self.logger = self._build_logger()

        print('[PhysioME Parameter]')
        print('   >> Modal Names : {0}'.format(', '.join(self.ch_names)))
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Leaning Rate : {0}'.format(self.lr))

    def _build_logger(self):
        log_dir = os.path.join(self.args.ckpt_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'train.log')

        logger = logging.getLogger(f'physiome.{id(self)}')
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

    def train(self):
        train_dataset = TorchDataset(paths=self.train_paths, ch_names=self.ch_names,
                                     sfreq=self.args.sfreq, rfreq=self.args.rfreq, scaler=self.args.data_scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        val_dataset = TorchDataset(paths=self.val_paths, ch_names=self.ch_names,
                                   sfreq=self.args.sfreq, rfreq=self.args.rfreq, scaler=self.args.data_scaler,
                                   downsampling=self.args.class_downsampling)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_batch_size)
        eval_dataset = TorchDataset(paths=self.eval_paths, ch_names=self.ch_names,
                                    sfreq=self.args.sfreq, rfreq=self.args.rfreq, scaler=self.args.data_scaler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size)

        total_step = 0
        best_multimodal_model_state, best_score = self.model.state_dict(), 0
        for epoch in range(self.args.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()
            for x, _ in train_dataloader:
                data = {
                    ch_name: torch.tensor(x[:, i, :].squeeze(), dtype=torch.float32).to(device)
                    for i, ch_name in enumerate(train_dataset.ch_names)
                }

                inter_recon_loss, missing_recon_loss, cross_contra_loss, cross_contra_acc = \
                    self.model(data=data, mask_ratio=self.args.mask_ratio)
                loss = inter_recon_loss + missing_recon_loss + cross_contra_loss
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:07d}  '
                          '[Inter Recon Loss] : {2:2.4f}  '
                          '[Missing Recon Loss]: {3:2.4f}  '
                          '[Cross Contra Loss] : {4:2.4f}  '
                          '[Cross Contra Acc] : {5:2.3f}  '
                          '[Total Loss] : {6:2.3f}'.format(epoch, total_step + 1,
                                                           inter_recon_loss,
                                                           missing_recon_loss,
                                                           cross_contra_loss,
                                                           cross_contra_acc, loss))

                self.logger.info(
                    'train step=%07d epoch=%03d '
                    'inter_recon=%.6f missing_recon=%.6f '
                    'cross_contra=%.6f cross_acc=%.4f total=%.6f',
                    total_step, epoch,
                    float(inter_recon_loss), float(missing_recon_loss),
                    float(cross_contra_loss), float(cross_contra_acc), float(loss),
                )

                step += 1
                total_step += 1

            acc, mf1 = self.linear_probing(epoch, val_dataloader, eval_dataloader)

            self.logger.info(
                'probe step=%07d epoch=%03d val_acc=%.4f val_macro_f1=%.4f',
                total_step, epoch, float(acc), float(mf1),
            )

            if mf1 > best_score:
                best_score = mf1
                best_multimodal_model_state = self.model.state_dict()
            self.scheduler.step()

        self.save_ckpt(best_multimodal_model_state)

    def linear_probing(self, epoch, val_dataloader, eval_dataloader):
        """LR probe over a sampled set of modality subsets (see probe_utils)."""
        self.model.eval()
        max_subsets = int(getattr(self.args, 'probe_max_subsets', 10))
        subsets = select_probe_subsets(
            self.ch_names, max_subsets=max_subsets, seed=epoch,
        )

        train_fn = lambda subset: self.get_latent_vector(subset, val_dataloader)
        eval_fn = lambda subset: self.get_latent_vector(subset, eval_dataloader)
        mean_acc, mean_mf1, _ = run_probe(
            subsets, train_fn, eval_fn,
            log_prefix=f'[Epoch {epoch:03d}]',
        )
        self.model.train()
        return mean_acc, mean_mf1

    def get_latent_vector(self, modal_combination, dataloader):
        self.model.eval()
        total_x, total_y = [], []

        with torch.no_grad():
            for data in dataloader:
                x, y = data
                data = {
                    ch_name: x[:, i, :].squeeze().to(device)
                    for i, ch_name in enumerate(modal_combination)
                }
                latent = self.model.inference_missing_modality(data=data)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())

        total_x = np.concatenate(total_x, axis=0)
        total_y = np.concatenate(total_y, axis=0)

        self.model.train()
        return total_x, total_y

    def get_encoder_backbone(self):
        encoder_backbone = {}
        for ch_name, ckpt_path in self.args.modality_name2path.items():
            encoder_backbone[ch_name] = self.load_pretrained_unimodal(ckpt_path=ckpt_path)
        return encoder_backbone

    def data_paths(self):
        paths = group_cross_validation(base_path=self.args.base_path,
                                       test_size=self.args.test_size,
                                       holdout_subject_size=self.args.holdout_subject_size)
        train_paths, val_paths, eval_paths = paths['train_paths'], paths['val_paths'], paths['eval_paths']
        return train_paths, val_paths, eval_paths

    def load_pretrained_unimodal(self, ckpt_path):
        # 1. pretrained NeuroNet (Phase-1 unimodal MAE)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_parameter = ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        # strict=False -- decoder layout was migrated from timm.Block to BFM
        # TransformerEncoder; old Phase-1 ckpts lack the new decoder keys and
        # vice versa. Phase-2 only transfers encoder + frame_backbone +
        # cls_token, so missing/extra decoder keys are harmless here.
        pretrained_model.load_state_dict(ckpt['model_state'], strict=False)

        # 2. NeuroNetEncoder — direct submodule transfer (no name-substring magic).
        backbone = NeuroNetEncoder(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            encoder_embed_dim=model_parameter['encoder_embed_dim'],
            encoder_heads=model_parameter['encoder_heads'],
            encoder_depths=model_parameter['encoder_depths'],
        )
        backbone.frame_backbone.load_state_dict(pretrained_model.frame_backbone.state_dict())
        backbone.patch_embed.load_state_dict(pretrained_model.autoencoder.patch_embed.state_dict())
        backbone.encoder.load_state_dict(pretrained_model.autoencoder.encoder.state_dict())
        backbone.cls_token = pretrained_model.autoencoder.cls_token

        # 3. Hand-rolled LoRA on the GQA output projection (Hu et al. 2021 + rsLoRA).
        # See ``models/transformer/lora.py`` for the rationale (drops the peft dep
        # and the messy ``base_model.model...`` state-dict prefix).
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

    def save_ckpt(self, multimodal_model_state):
        ckpt_path = os.path.join(self.args.ckpt_path, 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        backbone_ckpt = torch.load(list(self.args.modality_name2path.values())[0], map_location='cpu')
        backbone_parameter = backbone_ckpt['model_parameter']

        torch.save({
            'model_name': 'PhysioME',
            'ch_names': self.ch_names,
            'modality_backbone_param': {
                'fs': backbone_parameter['fs'], 'second': backbone_parameter['second'],
                'time_window': backbone_parameter['time_window'], 'time_step': backbone_parameter['time_step'],
                'encoder_embed_dim': backbone_parameter['encoder_embed_dim'],
                'encoder_heads': backbone_parameter['encoder_heads'],
                'encoder_depths': backbone_parameter['encoder_depths']
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
                'temperature': self.args.temperature
            },
            'model_state': multimodal_model_state,
            'hyperparameter': self.args.__dict__,
            'paths': {'train_paths': self.train_paths, 'val_paths': self.val_paths, 'eval_paths': self.eval_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))


if __name__ == '__main__':
    augments = get_args()
    augments = load_config(path=augments.config_yaml)
    trainer = Trainer(augments)
    trainer.train()
