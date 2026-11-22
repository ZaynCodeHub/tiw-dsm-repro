"""VP-SDE helpers for the time-dependent discriminator.

Adapted from the official TIW-DSM `classifier_lib.vpsde` (Kim et al., 2024), which in
turn follows Discriminator Guidance (Kim et al., 2023). The discriminator is trained in
VP space (x_t = alpha(t) x_0 + sqrt(1-alpha(t)^2) z, beta_min=0.1, beta_max=20), while
the score model uses EDM's sigma. An EDM sample x = x_0 + sigma z is the VP sample
x_vp = alpha * x where alpha = 1/sqrt(1+sigma^2). VP time tau solves alpha(tau) = alpha.
"""
from __future__ import annotations

import math

import torch

BETA_0, BETA_1 = 0.1, 20.0


def marginal(t):
    log_mean = -0.25 * t ** 2 * (BETA_1 - BETA_0) - 0.5 * t * BETA_0
    mean = torch.exp(log_mean)
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean))
    return mean, std


def sigma_to_tau(sigma):
    return (-BETA_0 + torch.sqrt(BETA_0 ** 2 + 2.0 * (BETA_1 - BETA_0) * torch.log1p(sigma ** 2))) / (BETA_1 - BETA_0)


def edm_to_vp(x_edm, sigma):
    """Returns (x_vp, tau). sigma: [B]."""
    sigma = sigma.reshape(-1).to(torch.float32)
    tau = sigma_to_tau(sigma)
    alpha = 1.0 / torch.sqrt(1.0 + sigma ** 2)
    return alpha.view(-1, *[1] * (x_edm.ndim - 1)) * x_edm, tau


def _integral_beta(t):
    return 0.5 * t ** 2 * (BETA_1 - BETA_0) + t * BETA_0


def _antiderivative(t):
    t = torch.as_tensor(t, dtype=torch.float32)
    return torch.log(1.0 - torch.exp(-_integral_beta(t))) + _integral_beta(t)


def sample_tau(n, device, t_min=1e-5, importance=True):
    """Importance-sampled diffusion time used by the official discriminator training."""
    if not importance:
        return torch.rand(n, device=device) * (1 - t_min) + t_min
    Z = _antiderivative(1.0) - _antiderivative(t_min)
    u = torch.rand(n, device=device)
    val = torch.log(1.0 + torch.exp(Z * u + _antiderivative(t_min)))
    return (-BETA_0 + torch.sqrt(BETA_0 ** 2 + 2 * (BETA_1 - BETA_0) * val)) / (BETA_1 - BETA_0)
