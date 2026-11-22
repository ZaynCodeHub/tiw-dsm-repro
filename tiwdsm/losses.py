"""Training objectives (team implementation, written from the paper's equations).

Everything is in EDM's denoiser parameterisation. With x = y + sigma*n and
s_theta(x) = (D_theta(x) - x) / sigma^2, the paper's TIW-DSM integrand (Eq. 10 /
Algorithm 2, applied to D_obs with w~ = 2d, Eq. 36)

    w~(x) * || s_theta(x) - grad log p(x|y) - grad log w~(x) ||^2

becomes

    (w~(x) / sigma^4) * || D_theta(x) - ( y + sigma^2 * grad_x log d(x, sigma) ) ||^2 .

We keep EDM's lambda(sigma) (which absorbs the 1/sigma^4 factor up to EDM's weighting),
so the code below is the EDM loss with
  * a per-sample weight      w~ = 2 d(x, sigma)                 ("reweighting", W)
  * a shifted target         y + sigma^2 * grad_x log d(x, sigma) ("score correction", C)

Modes
-----
obs : plain DSM on D_obs                               -> DSM(obs) baseline
ref : plain DSM on D_ref only                          -> DSM(ref) baseline (optional)
iw  : time-INDEPENDENT weight 2 d(y, t~0), no correction -> IW-DSM baseline (optional)
tiw : time-dependent weight + correction               -> TIW-DSM (ours to reproduce)
      (use reweight=False or correction=False for the paper's Table 4 ablation)
"""
from __future__ import annotations

import torch

from .discriminator import clean_weight, ratio_and_correction
from .edm import edm_loss_weight, sample_training_sigma


def score_loss(denoiser, y, mode="obs", disc=None, reweight=True, correction=True, disc_amp=True):
    b = y.shape[0]
    bc = (-1,) + (1,) * (y.ndim - 1)
    sigma = sample_training_sigma(b, y.device)
    x = y + torch.randn_like(y) * sigma.view(bc)
    lam = edm_loss_weight(sigma)
    target = y
    w = torch.ones(b, device=y.device)
    stats = {}

    if mode == "tiw":
        w_t, g = ratio_and_correction(disc, x, sigma, with_grad=correction, amp=disc_amp)
        if correction:
            target = y + sigma.view(bc) ** 2 * g
            stats["corr_rms"] = float((sigma.view(bc) ** 2 * g).pow(2).mean().sqrt())
        if reweight:
            w = w_t
        stats["w_mean"], stats["w_std"] = float(w_t.mean()), float(w_t.std())
    elif mode == "iw":
        w = clean_weight(disc, y)
        stats["w_mean"], stats["w_std"] = float(w.mean()), float(w.std())
    elif mode not in ("obs", "ref"):
        raise ValueError(mode)

    D = denoiser(x, sigma)
    per_sample = ((D - target) ** 2).flatten(1).mean(dim=1)
    loss = (lam * w * per_sample).mean()
    return loss, stats
