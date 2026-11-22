"""Stage 1: train the time-dependent discriminator d_phi(x_t, t) (team implementation).

The training loop is our own; its settings follow the official train_classifier.py and the
paper's Table 6. Each step draws B reference and B biased images, perturbs them to a random
VP time (importance-sampled), feeds them through the frozen ADM feature extractor and trains
the shallow head with binary cross-entropy (reference = 1, biased = 0).

After training it writes a diagnostic of how separable the two sources are at each noise
level (results/discriminator_diag.json). This is the core assumption of TIW: separability
should fall as t grows.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiwdsm import vpsde  # noqa: E402
from tiwdsm.data import Paths, load_bias_ref, to_tensor  # noqa: E402
from tiwdsm.discriminator import TimeDiscriminator, load_discriminator  # noqa: E402


@torch.no_grad()
def diagnose(disc, bias, ref, device, n=500, taus=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0), seed=0):
    g = torch.Generator().manual_seed(seed)
    rb = to_tensor(bias[torch.randperm(len(bias), generator=g)[:n].numpy()]).to(device)
    rr = to_tensor(ref[:n]).to(device)
    out = []
    for t in taus:
        tau = torch.full((len(rb),), t, device=device)
        m, s = vpsde.marginal(tau)
        pb, pr = [], []
        for x0, store in [(rb, pb), (rr, pr)]:
            tt = tau[:len(x0)]
            mm, ss = m[:len(x0)], s[:len(x0)]
            xt = mm.view(-1, 1, 1, 1) * x0 + ss.view(-1, 1, 1, 1) * torch.randn(x0.shape, generator=g).to(device)
            store.append(torch.sigmoid(disc(xt, tt).float()).cpu())
        pb, pr = torch.cat(pb), torch.cat(pr)
        acc = 0.5 * (float((pr > 0.5).float().mean()) + float((pb <= 0.5).float().mean()))
        wb = (pb / (1 - pb)).clamp(max=1e4)
        out.append({"tau": t, "sigma_edm": float(torch.sqrt(1 / m[0] ** 2 - 1)), "balanced_acc": acc,
                    "mean_d_ref": float(pr.mean()), "mean_d_bias": float(pb.mean()),
                    "w_bias_median": float(wb.median()), "w_bias_p95": float(wb.quantile(0.95))})
        print(f"tau={t:.2f}  acc={acc:.3f}  d_ref={float(pr.mean()):.3f}  d_bias={float(pb.mean()):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--backbone", default="adm", choices=["adm", "scratch"])
    ap.add_argument("--extractor", default="checkpoints/feature_extractor/32x32_classifier.pt")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128, help="per source (official: 128 + 128)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints/discriminator.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--diag_only", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bias, ref = load_bias_ref(Paths(a.root))
    ext = a.extractor if a.backbone == "adm" else None

    if not a.diag_only:
        disc = TimeDiscriminator(a.backbone, ext).to(dev)
        opt = torch.optim.Adam(disc.trainable_parameters(), lr=a.lr, weight_decay=1e-7)
        scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev == "cuda")
        bce = nn.BCEWithLogitsLoss()
        Xb, Xr = to_tensor(bias), to_tensor(ref)
        log, t0 = [], time.time()
        for it in range(1, a.iters + 1):
            xb = Xb[torch.randint(len(Xb), (a.batch,))]
            xr = Xr[torch.randint(len(Xr), (a.batch,))]  # sampling with replacement == official x20 repetition
            x0 = torch.cat([xr, xb]).to(dev)
            lab = torch.cat([torch.ones(a.batch), torch.zeros(a.batch)]).to(dev)
            tau = vpsde.sample_tau(len(x0), dev)
            m, s = vpsde.marginal(tau)
            xt = m.view(-1, 1, 1, 1) * x0 + s.view(-1, 1, 1, 1) * torch.randn_like(x0)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                logits = disc(xt, tau)
            loss = bce(logits.float(), lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if it % 100 == 0 or it == 1:
                acc = float(((logits > 0).float() == lab).float().mean())
                log.append({"iter": it, "bce": float(loss), "acc": acc, "sec": time.time() - t0})
                print(f"iter {it}/{a.iters}  bce {float(loss):.4f}  acc {acc:.3f}  {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save(disc.head_state(), a.out)
        os.makedirs("results", exist_ok=True)
        json.dump({"args": vars(a), "log": log, "train_seconds": time.time() - t0},
                  open("results/discriminator_train.json", "w"), indent=2)

    disc = load_discriminator(a.out, ext, dev)
    diag = diagnose(disc, bias, ref, dev)
    os.makedirs("results", exist_ok=True)
    json.dump(diag, open("results/discriminator_diag.json", "w"), indent=2)


if __name__ == "__main__":
    main()
