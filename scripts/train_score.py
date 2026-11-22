"""Stage 2: train the score network with DSM(obs) / DSM(ref) / IW-DSM / TIW-DSM (team implementation).

Resumable: re-running the same command continues from <outdir>/ckpt_latest.pt, so long runs
can be split across 12-hour Kaggle/Colab sessions. --time_limit_h saves and exits cleanly
before the platform kills the session.

Defaults follow the official command for CIFAR-10 LT / 5% (batch 256, lr 1e-3 with a 10 Mimg ramp-up,
EMA half-life 0.5 Mimg, 11 Mimg of training, dropout 0.13). The main deviations are the smaller U-Net,
fp16 mixed precision and x-flip instead of EDM's augmentation pipeline.

Example:
  python scripts/train_score.py --mode tiw --outdir runs/tiw --disc checkpoints/discriminator.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiwdsm.data import ArrayDataset, Paths, build_obs, infinite_loader, load_bias_ref, to_uint8  # noqa: E402
from tiwdsm.discriminator import load_discriminator  # noqa: E402
from tiwdsm.edm import EMA, EDMPrecond, sample, seeded_latents  # noqa: E402
from tiwdsm.losses import score_loss  # noqa: E402
from tiwdsm.unet import PRESETS, build_unet  # noqa: E402


def save_grid(images_uint8, path, nrow=8):
    from PIL import Image
    n = len(images_uint8); ncol = (n + nrow - 1) // nrow
    grid = np.zeros((ncol * 34 + 2, nrow * 34 + 2, 3), np.uint8) + 255
    for i, im in enumerate(images_uint8):
        r, c = divmod(i, nrow)
        grid[2 + r * 34:2 + r * 34 + 32, 2 + c * 34:2 + c * 34 + 32] = im
    Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--mode", default="tiw", choices=["obs", "ref", "iw", "tiw"])
    ap.add_argument("--no_reweight", action="store_true", help="TIW ablation: correction only")
    ap.add_argument("--no_correction", action="store_true", help="TIW ablation: reweighting only")
    ap.add_argument("--disc", default="checkpoints/discriminator.pt")
    ap.add_argument("--extractor", default="checkpoints/feature_extractor/32x32_classifier.pt")
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--batch", type=int, default=256, help="official: 256")
    ap.add_argument("--lr", type=float, default=1e-3, help="official: 1e-3")
    ap.add_argument("--lr_rampup_kimg", type=float, default=10000, help="official EDM/TIW default")
    ap.add_argument("--ema_halflife_kimg", type=float, default=500)
    ap.add_argument("--duration_mimg", type=float, default=11.0, help="official CIFAR-10 LT 5%% command uses 11 Mimg")
    ap.add_argument("--snapshot_kimg", type=float, default=500)
    ap.add_argument("--time_limit_h", type=float, default=11.3)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per step (batch is the total per step)")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = a.amp and dev == "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.outdir, exist_ok=True)

    bias, ref = load_bias_ref(Paths(a.root))
    if a.mode == "ref":
        images, tags = ref, np.ones(len(ref), np.int64)
    else:
        images, tags = build_obs(bias, ref, ref_repeat=20)
    loader = DataLoader(ArrayDataset(images, tags, flip=True), batch_size=a.batch, shuffle=True,
                        num_workers=a.workers, drop_last=True, pin_memory=dev == "cuda",
                        persistent_workers=a.workers > 0)
    batches = infinite_loader(loader)
    accum = a.accum

    disc = None
    if a.mode in ("tiw", "iw"):
        disc = load_discriminator(a.disc, a.extractor, dev)

    model = EDMPrecond(build_unet(a.preset)).to(dev)
    ema = EMA(model, a.ema_halflife_kimg)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())

    cur_nimg, train_sec = 0, 0.0
    ck = os.path.join(a.outdir, "ckpt_latest.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); ema.ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); cur_nimg, train_sec = s["cur_nimg"], s["train_sec"]
        torch.set_rng_state(s["rng_cpu"])
        print(f"resumed from {ck} at {cur_nimg / 1e6:.3f} Mimg")
    json.dump({**vars(a), "n_params": n_params}, open(os.path.join(a.outdir, "args.json"), "w"), indent=2)

    def denoise_amp(x, sigma):
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            return model(x, sigma)

    def checkpoint(final=False):
        torch.save({"model": model.state_dict(), "ema": ema.ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "cur_nimg": cur_nimg, "train_sec": train_sec,
                    "rng_cpu": torch.get_rng_state()}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        meta = {"preset": a.preset, "mode": a.mode, "no_reweight": a.no_reweight, "no_correction": a.no_correction,
                "nimg": cur_nimg, "train_hours": train_sec / 3600, "batch": a.batch, "n_params": n_params}
        torch.save({"ema": ema.ema.state_dict(), "meta": meta}, os.path.join(a.outdir, "ema_latest.pt"))
        if cur_nimg >= next_snap - 1 or final:  # keep one EMA per snapshot so models can be compared at equal Mimg
            torch.save({"ema": ema.ema.state_dict(), "meta": meta}, os.path.join(a.outdir, f"ema_{cur_nimg // 1000:06d}k.pt"))
        if final:
            torch.save({"ema": ema.ema.state_dict(), "meta": meta}, os.path.join(a.outdir, "ema_final.pt"))

    def snapshot():
        ema.ema.eval()
        z = seeded_latents(range(64), dev)
        imgs = to_uint8(sample(ema.ema, z, num_steps=18))
        save_grid(imgs, os.path.join(a.outdir, f"samples_{cur_nimg // 1000:06d}k.png"))

    logf = open(os.path.join(a.outdir, "log.jsonl"), "a")
    total = int(a.duration_mimg * 1e6)
    next_snap = (cur_nimg // int(a.snapshot_kimg * 1000) + 1) * int(a.snapshot_kimg * 1000)
    session_start, tick_t, tick_n, acc = time.time(), time.time(), cur_nimg, []
    model.train()
    while cur_nimg < total:
        y, _ = next(batches)
        y = y.to(dev, non_blocking=True)
        lr = a.lr * min(cur_nimg / max(a.lr_rampup_kimg * 1000, 1e-8), 1.0) if a.lr_rampup_kimg > 0 else a.lr
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        try:
            losses, stats = [], {}
            for yi in y.chunk(accum):  # gradient accumulation keeps the effective batch = a.batch
                li, stats = score_loss(denoise_amp, yi, mode=a.mode, disc=disc, reweight=not a.no_reweight,
                                       correction=not a.no_correction, disc_amp=use_amp)
                scaler.scale(li / accum).backward()
                losses.append(float(li))
            loss = float(np.mean(losses))
        except torch.cuda.OutOfMemoryError:
            if accum >= a.batch // 8:
                raise
            accum *= 2
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"CUDA OOM -> using {accum} micro-batches of {a.batch // accum} (effective batch unchanged)", flush=True)
            continue
        scaler.unscale_(opt)
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0, posinf=1e5, neginf=-1e5)
        scaler.step(opt); scaler.update()
        cur_nimg += a.batch
        ema.update(model, cur_nimg, a.batch)
        acc.append({"loss": loss, **stats})

        if cur_nimg - tick_n >= 50_000 or cur_nimg >= total:
            now = time.time(); dt = now - tick_t; train_sec += dt
            rec = {"kimg": cur_nimg / 1000, "lr": lr, "sec_per_kimg": dt / ((cur_nimg - tick_n) / 1000),
                   "train_hours": train_sec / 3600}
            for k in acc[0]:
                rec[k] = float(np.mean([r[k] for r in acc if k in r]))
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
            tick_t, tick_n, acc = now, cur_nimg, []
            if cur_nimg >= next_snap:
                checkpoint(); snapshot(); model.train()
                next_snap += int(a.snapshot_kimg * 1000)
            if (now - session_start) / 3600 > a.time_limit_h:
                checkpoint()
                print("time limit reached; checkpoint saved. Re-run the same command to resume.")
                return
    checkpoint(final=True); snapshot()
    print("training finished:", os.path.join(a.outdir, "ema_final.pt"))


if __name__ == "__main__":
    main()
