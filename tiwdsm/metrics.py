"""Evaluation metrics (team implementation).

* FID         : Inception-v3 pool3 features from `pytorch-fid` (Seitzer, reused library),
                which uses the same TF-ported FID weights as EDM's fid.py; reference statistics
                come from the full balanced CIFAR-10 training set (50,000 images), as in the paper.
* Precision / Recall : k-NN manifold estimate of Kynkaenniemi et al. (2019), k = 3, on the same
                Inception features (our own implementation).
* Bias        : paper Eq. 51, Bias = sum_z | E_{x~D_ref}[p(z|x)] - E_{x~p_theta}[p(z|x)] |,
                with p(z|x) from a pretrained CIFAR-10 VGG13-BN classifier (same family as the paper).
* Class histogram metrics : argmax class shares, L1 / KL distance to uniform, head/tail shares.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASS_NAMES

HEAD = [0, 1, 2, 3, 4]  # airplane..deer: the 5 most frequent classes in D_bias
TAIL = [5, 6, 7, 8, 9]  # dog..truck: the 5 rarest classes in D_bias


# ----------------------------------------------------------------------------- Inception / FID
_INCEPTION = {}


def inception(device):
    if device not in _INCEPTION:
        from pytorch_fid.inception import InceptionV3
        _INCEPTION[device] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    return _INCEPTION[device]


@torch.no_grad()
def inception_features(images_uint8: np.ndarray, device, batch=250) -> np.ndarray:
    net = inception(device)
    feats = []
    for i in range(0, len(images_uint8), batch):
        x = torch.from_numpy(images_uint8[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        feats.append(net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float64))
    return np.concatenate(feats)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    """||mu1-mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}); same formula as pytorch-fid, written to work with
    every SciPy version (newer SciPy removed the `disp` argument pytorch-fid relies on)."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = np.real(covmean)
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid_from_feats(feats, mu_ref, sigma_ref):
    mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
    return frechet_distance(mu, sigma, mu_ref, sigma_ref)


def reference_stats(train_images, device, cache="stats/cifar10_train_inception.npz", n_pr=10000, seed=0):
    """Inception mean/cov of the balanced target + a fixed 10k feature subset for precision/recall."""
    if os.path.exists(cache):
        z = np.load(cache)
        return z["mu"], z["sigma"], z["pr_feats"]
    feats = inception_features(train_images, device)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), min(n_pr, len(feats)), replace=False)
    mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, mu=mu, sigma=sigma, pr_feats=feats[idx].astype(np.float32))
    return mu, sigma, feats[idx].astype(np.float32)


# ----------------------------------------------------------------------------- precision / recall
def _kth_radius(feats, k, device, chunk=2000):
    f = torch.from_numpy(feats).to(device)
    radii = []
    for i in range(0, len(f), chunk):
        d = torch.cdist(f[i:i + chunk], f)
        radii.append(d.kthvalue(k + 1, dim=1).values)  # +1 skips the point itself
    return torch.cat(radii)


def _coverage(query, support, radii, device, chunk=2000):
    q, s = torch.from_numpy(query).to(device), torch.from_numpy(support).to(device)
    inside = []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], s)
        inside.append((d <= radii[None]).any(dim=1))
    return float(torch.cat(inside).float().mean())


def precision_recall(real_feats, gen_feats, device, k=3):
    real_feats = real_feats.astype(np.float32); gen_feats = gen_feats.astype(np.float32)
    r_real = _kth_radius(real_feats, k, device)
    r_gen = _kth_radius(gen_feats, k, device)
    precision = _coverage(gen_feats, real_feats, r_real, device)
    recall = _coverage(real_feats, gen_feats, r_gen, device)
    return precision, recall


# ----------------------------------------------------------------------------- class distribution
def load_eval_classifier(path, device):
    return torch.jit.load(path, map_location=device).eval()


@torch.no_grad()
def class_probs(clf, images_uint8, device, batch=500):
    out = []
    for i in range(0, len(images_uint8), batch):
        x = torch.from_numpy(images_uint8[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
        out.append(F.softmax(clf(x).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def distribution_metrics(probs, ref_probs=None):
    k = probs.shape[1]
    hist = np.bincount(probs.argmax(1), minlength=k) / len(probs)
    uni = np.full(k, 1 / k)
    eps = 1e-12
    m = {
        "class_share": hist.tolist(),
        "class_share_named": {c: float(h) for c, h in zip(CLASS_NAMES, hist)},
        "l1_to_uniform": float(np.abs(hist - uni).sum()),
        "kl_to_uniform": float(np.sum(hist * np.log((hist + eps) / uni))),
        "head_share": float(hist[HEAD].sum()),
        "tail_share": float(hist[TAIL].sum()),
        "max_over_min": float(hist.max() / max(hist.min(), eps)),
    }
    if ref_probs is not None:  # paper Eq. 51
        m["bias_eq51"] = float(np.abs(ref_probs.mean(0) - probs.mean(0)).sum())
    return m
