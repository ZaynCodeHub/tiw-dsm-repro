"""2-D sanity check of our TIW-DSM implementation (team implementation).

Recreates the paper's toy setting (Sec. 3, Fig. 2):
  p_bias = 0.9 N((-2,-2), I) + 0.1 N((2,2), I)
  p_data = 0.5 N((-2,-2), I) + 0.5 N((2,2), I)
with |D_bias| = 10,000 and |D_ref| = 500 (5%). It uses exactly the same loss, sampler and
discriminator utilities as the image pipeline, so it checks the signs and scales of
the reweighting and score-correction terms end to end. It runs in a few minutes on CPU.

Output: results/toy_2d.json and report/figures/toy_2d.png
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiwdsm import vpsde  # noqa: E402
from tiwdsm.edm import EDMPrecond, sample  # noqa: E402
from tiwdsm.losses import score_loss  # noqa: E402
from tiwdsm.unet import PositionalEmbedding  # noqa: E402


class MLP(nn.Module):
    def __init__(self, out=2, hidden=256, emb=64):
        super().__init__()
        self.pos = PositionalEmbedding(emb)
        self.net = nn.Sequential(nn.Linear(2 + emb, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.pos(t)], 1))


class ToyDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(out=1)

    def forward(self, x_vp, tau):
        return self.mlp(x_vp, tau * 999.0).squeeze(1)


def mixture(n, p_minor, rng):
    minor = rng.random(n) < p_minor
    centers = np.where(minor[:, None], 2.0, -2.0)
    return (centers + rng.normal(size=(n, 2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--disc_iters", type=int, default=3000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/toy_2d.json")
    ap.add_argument("--fig", default="report/figures/toy_2d.png")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed = [run_seed(sd, a) for sd in seeds]
    keys = list(per_seed[0][0].keys())
    results = {k: {"minor_mode_fraction": float(np.mean([r[0][k] for r in per_seed])),
                   "std": float(np.std([r[0][k] for r in per_seed])),
                   "per_seed": [r[0][k] for r in per_seed]} for k in keys}
    results.update({"target": 0.5, "bias_data": 0.1, "obs_data": 0.3, "seeds": seeds,
                    "disc_iters": a.disc_iters, "score_iters": a.iters})
    for k in keys:
        print(f"{k:5s}: {results[k]['minor_mode_fraction']:.3f} +- {results[k]['std']:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    plot(per_seed[0][1], results, a.fig)


def run_seed(seed, a):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = 0.25  # bring data to roughly EDM's sigma_data = 0.5 scale
    bias = torch.from_numpy(mixture(10000, 0.1, rng)) * scale
    ref = torch.from_numpy(mixture(500, 0.5, rng)) * scale
    obs = torch.cat([bias, ref.repeat(20, 1)])

    # 1) time-dependent discriminator (ref = 1, bias = 0)
    disc = ToyDisc()
    opt = torch.optim.Adam(disc.parameters(), 1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(a.disc_iters):
        xr = ref[torch.randint(len(ref), (256,))]
        xb = bias[torch.randint(len(bias), (256,))]
        x0 = torch.cat([xr, xb]); lab = torch.cat([torch.ones(256), torch.zeros(256)])
        tau = vpsde.sample_tau(512, "cpu")
        m, s = vpsde.marginal(tau)
        xt = m[:, None] * x0 + s[:, None] * torch.randn_like(x0)
        loss = bce(disc(xt, tau), lab)
        opt.zero_grad(); loss.backward(); opt.step()
    disc.eval().requires_grad_(False)

    results, samples = {}, {}
    runs = [("ref", "ref", ref, {}), ("obs", "obs", obs, {}), ("iw", "iw", obs, {}),
            ("tiw_W", "tiw", obs, {"correction": False}), ("tiw_C", "tiw", obs, {"reweight": False}),
            ("tiw", "tiw", obs, {})]
    for key, mode, data, kw in runs:
        torch.manual_seed(seed)
        den = EDMPrecond(MLP())
        opt = torch.optim.Adam(den.parameters(), 1e-3)
        for _ in range(a.iters):
            y = data[torch.randint(len(data), (512,))]
            loss, _ = score_loss(den, y, mode=mode, disc=disc, **kw)
            opt.zero_grad(); loss.backward(); opt.step()
        den.eval()
        x = sample(den, torch.randn(20000, 2, generator=torch.Generator().manual_seed(1000 + seed)), num_steps=40) / scale
        frac = float(((x[:, 0] + x[:, 1]) > 0).float().mean())
        results[key] = frac
        samples[key] = x[:3000].numpy()
        print(f"seed {seed} {key:5s}: share of the (2,2) mode = {frac:.3f}  (target 0.5)", flush=True)
    return results, samples


def plot(samples, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.2), sharex=True, sharey=True)
    names = {"ref": "DSM(ref)", "obs": "DSM(obs)", "iw": "IW-DSM", "tiw_W": "TIW: W only",
             "tiw_C": "TIW: C only", "tiw": "TIW-DSM (W+C)"}
    for ax, mode in zip(axes, ["ref", "obs", "iw", "tiw_W", "tiw_C", "tiw"]):
        s = samples[mode]
        ax.scatter(s[:, 0], s[:, 1], s=1, alpha=0.3, color="#2F75B5")
        ax.set_title(f"{names[mode]}\nshare = {results[mode]['minor_mode_fraction']:.2f} $\\pm$ {results[mode]['std']:.2f}", fontsize=9)
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal")
    fig.suptitle("2-D sanity check: target share of the (2,2) mode = 0.50", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
