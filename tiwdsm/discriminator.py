"""Time-dependent discriminator d_phi(x_t, t) and the quantities TIW-DSM needs.

[team implementation, architecture adapted from the official TIW-DSM code]

Labels: D_ref -> 1, D_bias -> 0. With equal sampling, d* = p_data / (p_data + p_bias), so
  w^t   = p_data^t / p_bias^t = d / (1 - d)                       (time-dependent ratio)
  w~^t  = p_data^t / p_obs^t  = 2 d,  with p_obs = (p_bias + p_data)/2   (paper Eq. 36)
  grad log w~^t = grad log d                                        (score correction)

Two backbones are provided:
  'adm'     : frozen pretrained ADM encoder (authors' 32x32_classifier.pt) + shallow ADM
              encoder head on its 8x8x512 features. Same as the paper (Table 6).
  'scratch' : small time-conditioned CNN trained from scratch. Used for smoke tests and as
              a fallback if the feature-extractor checkpoint cannot be downloaded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vpsde
from .unet import PositionalEmbedding

EPS = 1e-5


def _adm_extractor():
    from guided_diffusion.script_util import create_classifier  # reused third-party code
    return create_classifier(image_size=32, classifier_use_fp16=False, classifier_width=128,
                             classifier_depth=4, classifier_attention_resolutions="32,16,8",
                             classifier_use_scale_shift_norm=True, classifier_resblock_updown=True,
                             classifier_pool="attention", out_channels=1000)


def _adm_head():
    from guided_diffusion.script_util import create_classifier
    return create_classifier(image_size=8, classifier_use_fp16=False, classifier_width=128,
                             classifier_depth=2, classifier_attention_resolutions="32,16,8",
                             classifier_use_scale_shift_norm=True, classifier_resblock_updown=True,
                             classifier_pool="attention", out_channels=1, in_channels=512)


class ScratchDisc(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.pos = PositionalEmbedding(ch)
        self.emb = nn.Sequential(nn.Linear(ch, ch * 4), nn.SiLU(), nn.Linear(ch * 4, ch * 4))
        self.c1 = nn.Conv2d(3, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch * 2, 4, 2, 1)
        self.c3 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        self.c4 = nn.Conv2d(ch * 4, ch * 4, 4, 2, 1)
        self.film = nn.Linear(ch * 4, ch * 4)
        self.out = nn.Linear(ch * 4, 1)

    def forward(self, x, tau):
        e = self.emb(self.pos(tau * 999.0))
        h = F.silu(self.c1(x))
        h = F.silu(self.c2(h))
        h = F.silu(self.c3(h))
        h = self.c4(h) + self.film(e)[:, :, None, None]
        h = F.silu(h).mean(dim=(2, 3))
        return self.out(h).squeeze(1)


class TimeDiscriminator(nn.Module):
    """Input: VP-space sample x_vp and VP time tau in (0, 1]. Output: logit of P(reference)."""

    def __init__(self, backbone: str = "adm", extractor_ckpt: str | None = None):
        super().__init__()
        self.backbone = backbone
        if backbone == "adm":
            self.extractor = _adm_extractor()
            if extractor_ckpt is not None:
                self.extractor.load_state_dict(torch.load(extractor_ckpt, map_location="cpu"))
            self.extractor.eval().requires_grad_(False)
            self.head = _adm_head()
        elif backbone == "scratch":
            self.extractor = None
            self.head = ScratchDisc()
        else:
            raise ValueError(backbone)

    def trainable_parameters(self):
        return self.head.parameters()

    def forward(self, x_vp, tau):
        if self.backbone == "adm":
            feat = self.extractor(x_vp, timesteps=tau, feature=True)
            return self.head(feat, tau).view(-1)  # logits (we apply sigmoid / BCE-with-logits ourselves)
        return self.head(x_vp, tau)

    def head_state(self):
        return {"backbone": self.backbone, "head": self.head.state_dict()}


def load_discriminator(path: str, extractor_ckpt: str | None, device):
    state = torch.load(path, map_location="cpu")
    disc = TimeDiscriminator(state["backbone"], extractor_ckpt if state["backbone"] == "adm" else None)
    disc.head.load_state_dict(state["head"])
    return disc.to(device).eval().requires_grad_(False)


def prob_ref(disc, x_edm, sigma):
    """d_phi evaluated on an EDM-space sample, clipped like the official code."""
    x_vp, tau = vpsde.edm_to_vp(x_edm, sigma)
    return torch.sigmoid(disc(x_vp, tau).float()).clamp(EPS, 1 - EPS)


def ratio_and_correction(disc, x_edm, sigma, with_grad=True, amp=True):
    """Returns (w_tilde = 2d, grad_x log d) for EDM-space noisy samples x_edm at noise sigma."""
    x = x_edm.detach().float().requires_grad_(with_grad)
    with torch.enable_grad() if with_grad else torch.no_grad():
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp and x.is_cuda):
            d = prob_ref(disc, x, sigma)
        if not with_grad:
            return (2 * d).detach(), None
        log_d = torch.log(d)
        grad = torch.autograd.grad(log_d.sum(), x)[0]
    return (2 * d).detach(), torch.nan_to_num(grad.detach().float())


@torch.no_grad()
def clean_weight(disc, y, t_min=1e-5):
    """Time-INDEPENDENT weight for the IW-DSM baseline: w~(x_0) = 2 d(x_0, t~0)."""
    tau = torch.full((y.shape[0],), t_min, device=y.device)
    mean, _ = vpsde.marginal(tau)
    d = torch.sigmoid(disc(mean.view(-1, *[1] * (y.ndim - 1)) * y, tau).float()).clamp(EPS, 1 - EPS)
    return 2 * d
