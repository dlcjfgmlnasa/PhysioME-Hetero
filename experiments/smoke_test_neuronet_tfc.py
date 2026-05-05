"""Phase-1 NeuroNet + TF-C smoke test.

Verifies the SimCLR -> TF-C migration end-to-end:

* Forward returns ``(recon, L_T, L_F, L_TF)`` finite.
* Each of the four losses receives gradient through the optimizer step.
* Time and freq projectors get distinct gradients (i.e., they're actually
  used by their respective contrastive paths).
* TF-C cross-domain projector receives gradient (the new piece vs SimCLR).
* Backbone, frame_backbone_freq, and shared autoencoder all get grads.
* Loss decreases over a handful of steps on a fixed deterministic batch.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from models.dp_neuronet.model import NeuroNet  # noqa: E402


def grad_norm(p) -> float:
    return float(p.grad.norm()) if (p is not None and p.grad is not None) else 0.0


def module_grad_norm(m) -> float:
    return float(sum(grad_norm(p) for p in m.parameters()))


def main() -> None:
    torch.manual_seed(0)
    device = torch.device('cpu')

    fs = 100
    second = 60
    time_window = 3
    time_step = 3
    batch_size = 8

    model = NeuroNet(
        fs=fs, second=second, time_window=time_window, time_step=time_step,
        encoder_embed_dim=64, encoder_depths=2, encoder_heads=4,
        decoder_embed_dim=64, decoder_depths=2, decoder_heads=4,
        projection_hidden=[128, 64], temperature=0.1,
    ).to(device)

    x = torch.randn(batch_size, fs * second, device=device)

    # --- 1. one-shot loss + gradient checks ---------------------------
    recon, l_t, l_f, l_tf = model(x, mask_ratio=0.4)
    print(f'[step0] recon={recon.item():.3f}  L_T={l_t.item():.3f}  '
          f'L_F={l_f.item():.3f}  L_TF={l_tf.item():.3f}')
    for name, lo in [('recon', recon), ('L_T', l_t), ('L_F', l_f), ('L_TF', l_tf)]:
        assert torch.isfinite(lo), f'{name} non-finite: {lo}'
        assert lo.item() > 0, f'{name} should be positive at init: {lo}'

    total = recon + l_t + l_f + l_tf
    total.backward()

    # Gradient routing checks.
    g_proj_t = module_grad_norm(model.projector_time)
    g_proj_f = module_grad_norm(model.projector_freq)
    g_proj_tfc = module_grad_norm(model.projector_tfc)
    g_backbone_t = module_grad_norm(model.frame_backbone)
    g_backbone_f = module_grad_norm(model.frame_backbone_freq)
    g_encoder = module_grad_norm(model.autoencoder.encoder)
    g_decoder = module_grad_norm(model.autoencoder.decoder_embed) \
              + module_grad_norm(model.autoencoder.decoder_pred)

    print(f'[grads] proj_time={g_proj_t:.4f}  proj_freq={g_proj_f:.4f}  '
          f'proj_tfc={g_proj_tfc:.4f}')
    print(f'        frame_t={g_backbone_t:.4f}  frame_f={g_backbone_f:.4f}  '
          f'encoder={g_encoder:.4f}  decoder={g_decoder:.4f}')

    assert g_proj_t > 0, 'projector_time received no gradient (L_T broken)'
    assert g_proj_f > 0, 'projector_freq received no gradient (L_F broken)'
    assert g_proj_tfc > 0, 'projector_tfc received no gradient (L_TF broken)'
    assert g_backbone_t > 0, 'frame_backbone (time) received no gradient'
    assert g_backbone_f > 0, 'frame_backbone_freq received no gradient'
    assert g_encoder > 0, 'shared encoder received no gradient'
    assert g_decoder > 0, 'shared decoder received no gradient (recon broken)'

    # --- 2. loss-decrease check on fixed batch -----------------------
    model.zero_grad()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    n_steps = 30
    for step in range(1, n_steps + 1):
        recon, l_t, l_f, l_tf = model(x, mask_ratio=0.4)
        total = recon + l_t + l_f + l_tf
        assert torch.isfinite(total), f'non-finite loss at step {step}: {total}'

        optim.zero_grad()
        total.backward()
        optim.step()
        history.append((recon.item(), l_t.item(), l_f.item(), l_tf.item(),
                        total.item()))
        if step in (1, 5, 10, 20, 30):
            print(f'  step {step:3d}  recon={recon.item():.3f}  '
                  f'L_T={l_t.item():.3f}  L_F={l_f.item():.3f}  '
                  f'L_TF={l_tf.item():.3f}  total={total.item():.3f}')

    # The shared autoencoder should reduce reconstruction loss and the
    # contrastive losses on a fixed (overfit) batch.
    first_total = history[0][4]
    last_total = history[-1][4]
    delta = last_total - first_total
    print(f'[check] total loss: {first_total:.3f} -> {last_total:.3f} '
          f'(delta={delta:+.3f})')
    assert last_total < first_total, (
        f'total loss did not decrease over {n_steps} steps on a fixed batch '
        f'({first_total:.3f} -> {last_total:.3f})'
    )

    # Each individual loss should not blow up (NaN/Inf).
    for step, (r, lt, lf, ltf, _t) in enumerate(history, 1):
        for name, val in (('recon', r), ('L_T', lt), ('L_F', lf), ('L_TF', ltf)):
            assert val == val and abs(val) < 1e6, (
                f'{name} pathological at step {step}: {val}'
            )

    # --- 3. forward_latent (Phase-2 transfer surface) check ----------
    model.eval()
    with torch.no_grad():
        # global_tokens=False: returns squeezed CLS  (B, D)
        cls = model.forward_latent(x, global_tokens=False)
        # global_tokens=True: returns full latent  (B, F+1, D)
        lat = model.forward_latent(x, global_tokens=True)
    assert cls.shape == (batch_size, model.autoencoder.embed_dim), (
        f'unexpected CLS shape: {cls.shape}'
    )
    assert lat.shape[0] == batch_size and lat.shape[2] == model.autoencoder.embed_dim, (
        f'unexpected latent shape: {lat.shape}'
    )
    print(f'[check] forward_latent CLS shape: {tuple(cls.shape)}  '
          f'global shape: {tuple(lat.shape)}')

    print()
    print('[smoke] PASSED -- TF-C losses fire, gradients route correctly, '
          f'loss decreases ({first_total:.3f} -> {last_total:.3f})')


if __name__ == '__main__':
    main()
