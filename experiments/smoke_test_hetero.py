"""End-to-end smoke test of the hetero SSL pipeline (no real VitalDB data).

Synthesises a small set of npz files in the format produced by
``dataset/data_parser/vital_db_ssl.py``, builds a :class:`PhysioME` with
toy 1D backbones (so we don't need pretrained NeuroNet checkpoints), and
runs a couple of optimization steps through
``HeteroVitalDBDataset`` + ``BucketBatchSampler`` + ``hetero_collate_fn``.

Asserts:
  * loss values are finite at every step
  * batches actually rotate through multiple buckets
  * presence_state from the loader is in {0, 2}
  * gradients flow into ``presence_state_embed`` and
    ``dropped_modality_token`` parameters
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from models.physiome.model import (  # noqa: E402
    PRESENCE_REAL,
    PRESENCE_NATURALLY_ABSENT,
    PhysioME,
)
from pretrained.physiome.hetero_data_loader import (  # noqa: E402
    MODAL_ORDER,
    BucketBatchSampler,
    HeteroVitalDBDataset,
    find_ssl_npz_paths,
    hetero_collate_fn,
)


class ToyBackbone(nn.Module):
    """Toy backbone that returns ``[B, num_frames + 1, embed_dim]`` (cls + frames).

    Acts as a stand-in for the LoRA-wrapped NeuroNetEncoder so we can smoke-test
    the hetero pipeline without having pretrained per-modality checkpoints.
    """

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


def synthesise_dataset(tmp_dir: str, num_cases: int, sfreq: int, duration: int,
                       rng: np.random.Generator) -> None:
    expected = sfreq * duration
    n_modals = len(MODAL_ORDER)
    # Every non-empty subset bitmap of length n_modals (e.g. for 4 modal:
    # 1111, 1110, 1101, ..., 0001 — 15 patterns).
    bucket_choices = [
        ''.join('1' if (i >> (n_modals - 1 - k)) & 1 else '0'
                for k in range(n_modals))
        for i in range(1, 1 << n_modals)
    ]
    full_bm = '1' * n_modals  # complete bucket — ensures miss_recon fires
    for ci in range(num_cases):
        # Pick a random subject-level availability bitmap. First few cases
        # are forced to the complete bucket so the synth-drop restoration
        # loss is exercised even on small smoke-test dataset sizes.
        if ci < max(2, num_cases // 8):
            subject_bm = full_bm
        else:
            subject_bm = rng.choice(bucket_choices)
        subject_modality_set = np.array([c == '1' for c in subject_bm], dtype=bool)

        n_seg = int(rng.integers(20, 40))
        x = rng.standard_normal((n_seg, len(MODAL_ORDER), expected)).astype(np.float32)
        # Per-segment mask: if subject doesn't have a modality at all, force False.
        mask = np.zeros((n_seg, len(MODAL_ORDER)), dtype=bool)
        for s in range(n_seg):
            for m_idx, present in enumerate(subject_modality_set):
                if present:
                    # 92% of segments retain the modality (simulating occasional artifact).
                    mask[s, m_idx] = bool(rng.random() < 0.92)
            if not mask[s].any():
                # ensure at least one valid modality in the segment
                avail = np.where(subject_modality_set)[0]
                if len(avail):
                    mask[s, rng.choice(avail)] = True

        # Zero-out positions where mask is False (matches parser convention).
        for s in range(n_seg):
            for m_idx in range(len(MODAL_ORDER)):
                if not mask[s, m_idx]:
                    x[s, m_idx] = 0.0

        np.savez(
            os.path.join(tmp_dir, f'case_{ci:04d}.npz'),
            x=x, mask=mask,
            modal_names=np.array(MODAL_ORDER),
            subject_modality_set=subject_modality_set,
            case_id=f'case_{ci:04d}',
        )


def main() -> None:
    rng = np.random.default_rng(42)
    torch.manual_seed(0)

    sfreq = 100
    duration = 60          # → 6000 samples per segment
    expected_samples = sfreq * duration
    num_frames = 5         # smaller than production but enough to exercise shapes
    embed_dim = 64

    with tempfile.TemporaryDirectory() as tmp:
        synthesise_dataset(tmp, num_cases=32, sfreq=sfreq, duration=duration, rng=rng)
        paths = find_ssl_npz_paths(tmp)
        ds = HeteroVitalDBDataset(paths, eager=True, normalize=True)
        sampler = BucketBatchSampler(
            ds.segment_bitmap_keys, batch_size=8,
            sampling='uniform', min_bucket_size=8, seed=0,
        )
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=hetero_collate_fn)
        print(f'[smoke] active buckets: {sorted(sampler.buckets.keys())}')

        model = PhysioME(
            backbone_networks={
                m: ToyBackbone(expected_samples, num_frames, embed_dim)
                for m in MODAL_ORDER
            },
            backbone_embed_dim=embed_dim, num_backbone_frames=num_frames,
            encoder_embed_dim=128, encoder_heads=4, encoder_depths=2,
            decoder_embed_dim=64, decoder_heads=4, decoder_depths=2,
            decoder_recon_depths=2,
            projection_hidden=[256, 128], temperature=0.1,
        )
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        seen_buckets = set()
        seen_presence_states = set()
        miss_nonzero_seen = False
        miss_zero_seen = False
        max_presence_grad = 0.0
        max_drop_token_grad = 0.0
        max_steps = 60
        n_steps = 0

        for batch in loader:
            n_steps += 1
            data = {k: v.float() for k, v in batch['data'].items()}
            presence_state = batch['presence_state']

            for s in presence_state.unique().tolist():
                seen_presence_states.add(int(s))

            inter, miss, contra, acc = model(
                data=data, presence_state=presence_state,
                mask_ratio=0.4,
                restoration_only_on_complete=True,
            )
            loss = inter + miss + contra
            assert torch.isfinite(loss), f'non-finite loss: {loss}'

            optim.zero_grad()
            loss.backward()

            if model.presence_state_embed.weight.grad is not None:
                max_presence_grad = max(
                    max_presence_grad,
                    float(model.presence_state_embed.weight.grad.norm()),
                )
            if model.dropped_modality_token.grad is not None:
                max_drop_token_grad = max(
                    max_drop_token_grad,
                    float(model.dropped_modality_token.grad.norm()),
                )

            optim.step()

            seen_buckets.add(batch['bucket_pattern'])
            if miss.item() > 0:
                miss_nonzero_seen = True
            else:
                miss_zero_seen = True

            print(f'  step {n_steps:3d}  bucket={batch["bucket_pattern"]} '
                  f'inter={inter.item():.3f}  miss={miss.item():.3f}  '
                  f'contra={contra.item():.3f}')
            if n_steps >= max_steps:
                break

        # Assertions ------------------------------------------------------
        assert n_steps > 0, 'loader yielded no batches'
        assert len(seen_buckets) >= 2, f'only one bucket seen: {seen_buckets}'
        assert seen_presence_states.issubset({PRESENCE_REAL, PRESENCE_NATURALLY_ABSENT}), \
            f'unexpected presence states: {seen_presence_states}'
        assert miss_nonzero_seen, 'restoration loss never fired on complete bucket'
        assert miss_zero_seen, 'restoration loss never zero on hetero bucket (expected for v1)'
        assert max_presence_grad > 0, 'presence_state_embed received no gradient'
        assert max_drop_token_grad > 0, 'dropped_modality_token received no gradient'

        print()
        print('[smoke] PASSED')
        print(f'  steps                  : {n_steps}')
        print(f'  buckets seen           : {sorted(seen_buckets)}')
        print(f'  presence states seen   : {sorted(seen_presence_states)}')
        print(f'  max ||grad presence||  : {max_presence_grad:.4f}')
        print(f'  max ||grad drop_token||: {max_drop_token_grad:.4f}')


if __name__ == '__main__':
    main()
