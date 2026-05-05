# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict

from models.physiome.model import PhysioME


class PhysioMEClassifier(nn.Module):
    """Thin classification wrapper around a pretrained PhysioME encoder.

    The backbone (PhysioME) is frozen; only ``self.fc`` is trained.
    Missing modalities are handled transparently via
    ``PhysioME.inference_missing_modality``.
    """

    def __init__(self, physio_me: PhysioME, n_classes: int):
        super().__init__()
        self.physio_me = physio_me
        self.modal_names = physio_me.modal_names
        d = physio_me.encoder_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.BatchNorm1d(d // 4),
            nn.GELU(),
            nn.Linear(d // 4, n_classes),
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            x = self.physio_me.inference_missing_modality(data=data)
        return self.fc(x)
