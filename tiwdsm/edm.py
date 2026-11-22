"""EDM preconditioning, training-noise distribution and deterministic samplers.

Adapted from Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based
Generative Models" (NVlabs/edm, CC BY-NC-SA 4.0). We re-wrote the formulas in our
own code instead of copying the EDM repository. The official TIW-DSM code builds on
the same EDM formulation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

SIGMA_DATA = 0.5
P_MEAN, P_STD = -1.2, 1.2
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0


class EDMPrecond(nn.Module):
    """D_theta(x; sigma) = c_skip x + c_out F_theta(c_in x, c_noise)."""

    def __init__(self, net: nn.Module, sigma_data: float = SIGMA_DATA):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x, sigma):
        sigma = sigma.reshape(-1, *[1] * (x.ndim - 1)).to(torch.float32)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log().flatten() / 4
        F_x = self.net((c_in * x).to(x.dtype), c_noise)
        return c_skip * x + c_out * F_x.to(torch.float32)


def sample_training_sigma(n, device):
    """log sigma ~ N(P_mean, P_std^2) (EDM)."""
    return (torch.randn(n, device=device) * P_STD + P_MEAN).exp()


def edm_loss_weight(sigma, sigma_data=SIGMA_DATA):
    """lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2 (EDM)."""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def karras_sigmas(num_steps, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO, device="cpu"):
    i = torch.arange(num_steps, dtype=torch.float64, device=device)
    t = (sigma_max ** (1 / rho) + i / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([t, torch.zeros_like(t[:1])])  # append sigma_N = 0


def nfe_for(solver: str, steps: int) -> int:
    return 2 * steps - 1 if solver == "heun" else steps


@torch.no_grad()
def sample(denoiser, latents, num_steps=18, solver="heun", sigma_max=SIGMA_MAX, x_init=None,
           return_trajectory=False):
    """Deterministic EDM ODE sampler (Algorithm 1 of Karras et al., without stochastic churn).

    latents  : standard normal noise [B,3,32,32]
    solver   : 'heun' (2nd order, NFE = 2N-1) or 'euler' (1st order, NFE = N)
    x_init   : optional clean image in [-1,1]; if given we start from x_init + sigma_max * latents
               (SDEdit-style image-to-image used in the demo app)
    """
    device = latents.device
    t_steps = karras_sigmas(num_steps, sigma_max=sigma_max, device=device)
    x = latents.to(torch.float64) * t_steps[0]
    if x_init is not None:
        x = x + x_init.to(torch.float64)
    traj = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        sig = torch.full((x.shape[0],), float(t_cur), device=device)
        denoised = denoiser(x.float(), sig).to(torch.float64)
        if return_trajectory:
            traj.append(denoised.float().cpu())
        d_cur = (x - denoised) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if solver == "heun" and i < num_steps - 1:
            sig_n = torch.full((x.shape[0],), float(t_next), device=device)
            denoised_n = denoiser(x_next.float(), sig_n).to(torch.float64)
            d_prime = (x_next - denoised_n) / t_next
            x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        x = x_next
    out = x.float().clamp(-1, 1)
    return (out, traj) if return_trajectory else out


def seeded_latents(seeds, device="cpu"):
    """One independent generator per seed, so image k depends only on seed k (reproducible demos)."""
    z = [torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
    return torch.stack(z).to(device)


class EMA:
    """EDM-style EMA with half-life in k-images and a ramp-up ratio."""

    def __init__(self, model: nn.Module, halflife_kimg=500.0, rampup_ratio=0.05):
        import copy
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.halflife_kimg, self.rampup_ratio = halflife_kimg, rampup_ratio

    @torch.no_grad()
    def update(self, model, cur_nimg, batch_size):
        hl = self.halflife_kimg * 1000
        if self.rampup_ratio is not None:
            hl = min(hl, cur_nimg * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(hl, 1e-8))
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, beta))
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)
