"""Additional experiment: sampling budget vs quality, class balance and inference time (team implementation).

Research question
  How does the number of sampling steps (NFE) and the solver order affect FID, recall,
  generated class balance and generation time? Does TIW-DSM keep its debiasing advantage
  over DSM(obs) when sampling is made cheaper?

Design
  Same frozen checkpoints, same seeds (0..N-1) for every setting, no retraining.
  Heun (2nd order, NFE = 2*steps - 1) and Euler (1st order, NFE = steps) at matched NFE.

  python scripts/steps_experiment.py --ckpt obs=runs/obs/ema_final.pt --ckpt tiw=runs/tiw/ema_final.pt
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import EvalContext, generate  # noqa: E402
from tiwdsm.edm import nfe_for, sample, seeded_latents  # noqa: E402
from tiwdsm.io import load_ema  # noqa: E402

DEFAULT_GRID = {"heun": [3, 5, 9, 13, 18, 25], "euler": [5, 9, 17, 25, 35, 49]}


@torch.no_grad()
def single_image_latency(model, steps, solver, device, repeats=5):
    z = seeded_latents([123], device)
    sample(model, z, num_steps=steps, solver=solver)  # warm-up
    ts = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        sample(model, z, num_steps=steps, solver=solver)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return 1000 * sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--heun", default=",".join(map(str, DEFAULT_GRID["heun"])))
    ap.add_argument("--euler", default=",".join(map(str, DEFAULT_GRID["euler"])))
    ap.add_argument("--out", default="results/steps_experiment.json")
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    ctx = EvalContext(a.root)
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["model"], r["solver"], r["steps"]) for r in rows}
    grid = [("heun", int(s)) for s in a.heun.split(",") if s] + [("euler", int(s)) for s in a.euler.split(",") if s]

    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        model, meta = load_ema(path, ctx.device)
        for solver, steps in grid:
            if (name, solver, steps) in done:
                continue
            images, sec = generate(model, a.num, steps, solver, a.batch, 0, ctx.device)
            r = {"model": name, "solver": solver, "steps": steps, "nfe": nfe_for(solver, steps),
                 "gen_seconds": sec, "ms_per_image_batched": 1000 * sec / a.num,
                 "ms_single_image": single_image_latency(model, steps, solver, ctx.device),
                 "device": torch.cuda.get_device_name(0) if ctx.device == "cuda" else "cpu"}
            r.update(ctx.evaluate(images))
            rows.append(r)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=2)
            print(f"{name:5s} {solver:5s} steps={steps:3d} NFE={r['nfe']:3d}  FID={r['fid']:.2f}  "
                  f"recall={r['recall']:.3f}  bias={r['bias_eq51']:.4f}  tail={r['tail_share']:.3f}  "
                  f"{r['ms_per_image_batched']:.2f} ms/img", flush=True)


if __name__ == "__main__":
    main()
