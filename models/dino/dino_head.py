# -*- coding:utf-8 -*-
"""DINOv3 projection + prototype head for biosignal SSL.

Follows the official DINOv3 reference (``facebookresearch/dinov3``):
3-layer MLP projector → L2-normalize → plain Linear prototype layer
(``last_layer``). The "freeze last layer for N epochs" stability trick is
NOT implemented by wrapping the prototype Linear in weight_norm anymore
(the v1/v2 approach); DINOv3 instead zeros the LR of ``last_layer`` for
the first ``freeze_last_layer_epochs`` via the LR scheduler. We expose
``last_layer`` as its own submodule so the trainer can target it directly.

DINO and iBOT heads are SEPARATE instances in DINOv3 (the codebase hard-
asserts ``ibot.separate_head is True``), so this class is instantiated
twice in BiosignalDINO — once for CLS / DINO, once for masked-patch / iBOT.
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
        self.last_layer = nn.Linear(bottleneck_dim, n_prototypes, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)
