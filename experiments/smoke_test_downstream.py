"""Downstream classifier smoke test — 7 modal subsets.

Builds a PhysioME with toy backbones (no pretrained ckpt), wraps it in
PhysioMEClassifier, and runs inference on every non-empty subset of
{ABP, ECG, PPG}. Verifies:

* PhysioMEClassifier.forward succeeds for all 2^N - 1 = 7 modal subsets.
* Output logits are (B, n_classes) and finite.
* fc head receives gradient on a binary cross-entropy backward pass.
* PhysioME backbone parameters DO NOT receive gradient (frozen-encoder check).
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from downstream.model import PhysioMEClassifier  # noqa: E402
from models.physiome.model import PhysioME  # noqa: E402
from pretrained.physiome.hetero_data_loader import MODAL_ORDER  # noqa: E402


class ToyBackbone(nn.Module):
    def __init__(self, signal_len: int, num_frames: int, embed_dim: int):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.proj = nn.Linear(signal_len, embed_dim * num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feats = self.proj(x).view(b, self.num_frames, self.embed_dim)
        cls = torch.zeros(b, 1, self.embed_dim, device=x.device, dtype=feats.dtype)
        return torch.cat([cls, feats], dim=1)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device('cpu')

    signal_len = 6000  # 60s @ 100Hz
    num_frames = 5
    embed_dim = 64
    encoder_dim = 128
    batch_size = 4
    n_classes = 2

    backbone = {m: ToyBackbone(signal_len, num_frames, embed_dim) for m in MODAL_ORDER}
    physio_me = PhysioME(
        backbone_networks=backbone,
        backbone_embed_dim=embed_dim, num_backbone_frames=num_frames,
        encoder_embed_dim=encoder_dim, encoder_heads=4, encoder_depths=2,
        decoder_embed_dim=64, decoder_heads=4, decoder_depths=2,
        decoder_recon_depths=2,
        projection_hidden=[128, 64], temperature=0.1,
    )
    classifier = PhysioMEClassifier(physio_me=physio_me, n_classes=n_classes).to(device)
    classifier.eval()  # PhysioME is frozen; fc still gets grads

    # Enumerate every non-empty modal subset (2^N - 1 total).
    n_modals = len(MODAL_ORDER)
    all_subsets = []
    for r in range(1, n_modals + 1):
        for combo in combinations(MODAL_ORDER, r):
            all_subsets.append(list(combo))

    expected_n = (1 << n_modals) - 1
    assert len(all_subsets) == expected_n, (
        f'expected {expected_n} subsets for N={n_modals}, got {len(all_subsets)}'
    )

    optim = torch.optim.AdamW(classifier.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    targets = torch.randint(0, n_classes, (batch_size,))

    physio_me_param_grad_seen = False
    for subset in all_subsets:
        data = {m: torch.randn(batch_size, signal_len, device=device) for m in subset}

        # forward (eval-mode but fc still trainable — re-enable training for fc)
        classifier.fc.train()
        logits = classifier(data)
        assert logits.shape == (batch_size, n_classes), (
            f'subset={subset}: unexpected logit shape {logits.shape}'
        )
        assert torch.isfinite(logits).all(), f'subset={subset}: non-finite logits'

        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f'subset={subset}: non-finite loss'

        optim.zero_grad()
        loss.backward()

        # fc must receive gradient
        fc_grad = sum(
            float(p.grad.norm()) if p.grad is not None else 0.0
            for p in classifier.fc.parameters()
        )
        assert fc_grad > 0, f'subset={subset}: fc head received no gradient'

        # PhysioME backbone must NOT receive gradient (frozen-encoder check)
        for p in classifier.physio_me.parameters():
            if p.grad is not None and float(p.grad.norm()) > 0:
                physio_me_param_grad_seen = True

        optim.step()
        print(f'  subset={"+".join(subset):<15} logits={tuple(logits.shape)}  '
              f'loss={loss.item():.3f}  fc_grad_norm={fc_grad:.3f}')

    assert not physio_me_param_grad_seen, (
        'PhysioME backbone received gradient — should be frozen via '
        '`with torch.no_grad():` in PhysioMEClassifier.forward'
    )

    print()
    print(f'[smoke] PASSED -- {len(all_subsets)} modal subsets ({n_modals} modal), '
          'fc grads flow, backbone frozen')


if __name__ == '__main__':
    main()
