"""Generate samples from a trained model and compute every reported metric (team implementation).

  python scripts/evaluate.py --ckpt runs/tiw/ema_final.pt --name tiw --num 50000
  python scripts/evaluate.py --real bias --name data_bias      # metrics of the training data itself

Writes results/eval_<name>.json (and optionally the generated images as an .npz).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiwdsm import metrics as M  # noqa: E402
from tiwdsm.data import Paths, load_bias_ref, to_uint8  # noqa: E402
from tiwdsm.edm import nfe_for, sample, seeded_latents  # noqa: E402
from tiwdsm.io import file_sha256, load_ema  # noqa: E402


class EvalContext:
    def __init__(self, root="data", device=None, classifier="checkpoints/eval_classifier_vgg13bn.ts",
                 stats_cache="stats/cifar10_train_inception.npz"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        p = Paths(root)
        train = np.load(p.cifar_train)["images"]
        self.mu, self.sigma, self.pr_real = M.reference_stats(train, self.device, cache=stats_cache)
        self.clf = M.load_eval_classifier(classifier, self.device)
        _, ref = load_bias_ref(p)
        self.ref_probs = M.class_probs(self.clf, ref, self.device)

    def evaluate(self, images_uint8, fid=True, pr=True, n_pr=10000):
        out = {"n_images": int(len(images_uint8))}
        if fid or pr:
            feats = M.inception_features(images_uint8, self.device)
            if fid:
                out["fid"] = M.fid_from_feats(feats, self.mu, self.sigma)
            if pr:
                p, r = M.precision_recall(self.pr_real, feats[:n_pr], self.device)
                out["precision"], out["recall"], out["n_pr"] = p, r, int(min(n_pr, len(feats)))
        probs = M.class_probs(self.clf, images_uint8, self.device)
        out.update(M.distribution_metrics(probs, self.ref_probs))
        return out


@torch.no_grad()
def generate(model, num, steps=18, solver="heun", batch=500, seed_offset=0, device="cuda"):
    imgs, t_total = [], 0.0
    for i in range(0, num, batch):
        seeds = range(seed_offset + i, seed_offset + min(i + batch, num))
        z = seeded_latents(seeds, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        x = sample(model, z, num_steps=steps, solver=solver)
        if device == "cuda":
            torch.cuda.synchronize()
        t_total += time.time() - t0
        imgs.append(to_uint8(x))
    return np.concatenate(imgs), t_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--real", choices=["bias", "ref", "target"])
    ap.add_argument("--samples", help="evaluate an existing .npz (key 'images')")
    ap.add_argument("--name", required=True)
    ap.add_argument("--num", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--solver", default="heun", choices=["heun", "euler"])
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--root", default="data")
    ap.add_argument("--save_samples", action="store_true")
    ap.add_argument("--no_pr", action="store_true")
    a = ap.parse_args()
    ctx = EvalContext(a.root)
    rec = {"name": a.name}

    if a.ckpt:
        model, meta = load_ema(a.ckpt, ctx.device)
        images, sec = generate(model, a.num, a.steps, a.solver, a.batch, a.seed_offset, ctx.device)
        rec.update({"ckpt": a.ckpt, "ckpt_sha256": file_sha256(a.ckpt), "meta": meta, "steps": a.steps,
                    "solver": a.solver, "nfe": nfe_for(a.solver, a.steps), "gen_seconds": sec,
                    "ms_per_image": 1000 * sec / a.num, "seeds": [a.seed_offset, a.seed_offset + a.num - 1]})
        if a.save_samples:
            os.makedirs("samples", exist_ok=True)
            np.savez_compressed(f"samples/{a.name}.npz", images=images)
    elif a.samples:
        images = np.load(a.samples)["images"]
    else:
        bias, ref = load_bias_ref(Paths(a.root))
        images = {"bias": bias, "ref": ref, "target": np.load(Paths(a.root).cifar_test)["images"]}[a.real]
        rec["real"] = a.real

    rec.update(ctx.evaluate(images, pr=not a.no_pr))
    os.makedirs("results", exist_ok=True)
    out = f"results/eval_{a.name}.json"
    json.dump(rec, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("class_share", "meta")}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
