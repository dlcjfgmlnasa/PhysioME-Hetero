# -*- coding:utf-8 -*-
"""DINOv3 projection + prototype head for biosignal SSL.

Follows the DINOv2/v3 head design: 3-layer MLP projector → L2-normalize →
weight-normalized linear prototype layer (the "last_layer"). The prototype
linear's weight magnitude is frozen so that only its direction is learned,
which is what stabilises self-distillation. Centering + sharpening of the
teacher logits live in BiosignalDINO; this module is just the head itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256,
                 n_prototypes: int = 8192,
                 n_layers: int = 3):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f'n_layers must be >= 1, got {n_layers}')
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        last = nn.Linear(bottleneck_dim, n_prototypes, bias=False)
        nn.init.trunc_normal_(last.weight, std=0.02)
        self.last_layer = nn.utils.parametrizations.weight_norm(last)
        # Freeze magnitude (only direction learns) — DINO stability trick.
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)
