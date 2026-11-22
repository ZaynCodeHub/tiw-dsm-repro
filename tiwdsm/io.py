"""Loading trained checkpoints for sampling / evaluation / deployment (team implementation)."""
from __future__ import annotations

import hashlib

import torch

from .edm import EDMPrecond
from .unet import build_unet


def load_ema(path: str, device="cpu"):
    """Returns (denoiser in eval mode, metadata dict). Only the EMA weights are needed for inference."""
    s = torch.load(path, map_location="cpu", weights_only=False)
    meta = s.get("meta", {})
    model = EDMPrecond(build_unet(meta.get("preset", "small")))
    model.load_state_dict(s["ema"])
    return model.to(device).eval().requires_grad_(False), meta


def file_sha256(path: str, n=12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]
