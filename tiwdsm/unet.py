"""Compact DDPM++-style U-Net (team implementation).

The design follows the DDPM++ backbone the paper uses (Song et al., 2021; Karras et
al., 2022). It has residual blocks with GroupNorm, SiLU and dropout, self-attention at
16x16, a positional noise embedding fed through a 2-layer MLP, sqrt(1/2) skip scaling,
and zero-initialised output layers. It is written from scratch and is much smaller than
the paper's 56M-parameter DDPM++ so that it trains on one free T4/P100 GPU.

Presets
-------
small : ch=64,  mult=(1,2,2,2), 2 blocks/level, attn@16          ~ 9M params  (our default)
base  : ch=128, mult=(2,2,2),   4 blocks/level, attn@16          ~ 56M params (paper-sized)
tiny  : ch=32,  mult=(1,2),     1 block/level,  attn@16          (CPU smoke tests only)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "tiny": dict(ch=32, mult=(1, 2), blocks=1, attn_res=(16,), dropout=0.0),
    "small": dict(ch=64, mult=(1, 2, 2, 2), blocks=2, attn_res=(16,), dropout=0.13),
    "base": dict(ch=128, mult=(2, 2, 2), blocks=4, attn_res=(16,), dropout=0.13),
}


def _gn(c):
    return nn.GroupNorm(min(32, c // 4) if c >= 8 else 1, c, eps=1e-6)


def _zero(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim, self.max_positions = dim, max_positions

    def forward(self, x):  # x: [B]
        half = self.dim // 2
        freqs = torch.arange(half, device=x.device, dtype=torch.float32) / (half - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.float()[:, None] * freqs[None]
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.norm0 = _gn(cin)
        self.conv0 = nn.Conv2d(cin, cout, 3, padding=1)
        self.affine = nn.Linear(emb_dim, cout)
        self.norm1 = _gn(cout)
        self.drop = nn.Dropout(dropout)
        self.conv1 = _zero(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv0(F.silu(self.norm0(x)))
        h = h + self.affine(emb)[:, :, None, None]
        h = self.conv1(self.drop(F.silu(self.norm1(h))))
        return (h + self.skip(x)) * math.sqrt(0.5)


class Attention(nn.Module):
    def __init__(self, c, heads=1):
        super().__init__()
        self.heads = heads
        self.norm = _gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = _zero(nn.Conv2d(c, c, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return (x + self.proj(out)) * math.sqrt(0.5)


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """Maps (x_in, c_noise) -> F_theta(x_in, c_noise). Preconditioning lives in edm.EDMPrecond."""

    def __init__(self, img_res=32, in_ch=3, ch=64, mult=(1, 2, 2, 2), blocks=2, attn_res=(16,), dropout=0.13):
        super().__init__()
        emb_dim = ch * 4
        self.pos = PositionalEmbedding(ch)
        self.map = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim), nn.SiLU())
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.enc = nn.ModuleList()
        skips = [ch]
        cur, res = ch, img_res
        for li, m in enumerate(mult):
            cout = ch * m
            for _ in range(blocks):
                self.enc.append(nn.ModuleList([ResBlock(cur, cout, emb_dim, dropout),
                                               Attention(cout) if res in attn_res else nn.Identity()]))
                cur = cout
                skips.append(cur)
            if li != len(mult) - 1:
                self.enc.append(Down(cur))
                res //= 2
                skips.append(cur)

        self.mid = nn.ModuleList([ResBlock(cur, cur, emb_dim, dropout), Attention(cur),
                                  ResBlock(cur, cur, emb_dim, dropout)])

        self.dec = nn.ModuleList()
        for li, m in reversed(list(enumerate(mult))):
            cout = ch * m
            for _ in range(blocks + 1):
                self.dec.append(nn.ModuleList([ResBlock(cur + skips.pop(), cout, emb_dim, dropout),
                                               Attention(cout) if res in attn_res else nn.Identity()]))
                cur = cout
            if li != 0:
                self.dec.append(Up(cur))
                res *= 2
        self.norm_out = _gn(cur)
        self.conv_out = _zero(nn.Conv2d(cur, in_ch, 3, padding=1))

    def forward(self, x, c_noise):
        emb = self.map(self.pos(c_noise))
        h = self.conv_in(x)
        hs = [h]
        for layer in self.enc:
            if isinstance(layer, Down):
                h = layer(h)
            else:
                h = layer[1](layer[0](h, emb))
            hs.append(h)
        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)
        for layer in self.dec:
            if isinstance(layer, Up):
                h = layer(h)
            else:
                h = layer[1](layer[0](torch.cat([h, hs.pop()], 1), emb))
        return self.conv_out(F.silu(self.norm_out(h)))


def build_unet(preset: str = "small") -> UNet:
    return UNet(**PRESETS[preset])
