"""Data utilities (team implementation).

The official CIFAR-10 LT / 5% release contains:
  fake_data.npz : D_bias, 10,000 long-tailed images  (key "samples", uint8 NHWC)
  real_data.npz : D_ref,     500 reference images     (key "samples", uint8 NHWC)
  dataset.zip   : score-training archive = D_bias + 20 x D_ref (20,000 PNGs)

In Deliverable 2 we verified that dataset.zip is exactly D_bias plus D_ref repeated 20
times, so we rebuild D_obs from the two NPZ files here instead of decoding PNGs.
Repeating D_ref 20 times makes the two sources 1:1, which matches the paper's mixture
p_obs = 0.5 p_bias + 0.5 p_data (Appendix, Eq. 36).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_DATA_ROOT = os.environ.get("TIWDSM_DATA", "data")


@dataclass
class Paths:
    root: str = DEFAULT_DATA_ROOT

    @property
    def bias(self):
        return os.path.join(self.root, "tiw_dsm/cifar10/discriminator_training/bias_10000/fake_data.npz")

    @property
    def ref(self):
        return os.path.join(self.root, "tiw_dsm/cifar10/discriminator_training/unbias_500/real_data.npz")

    @property
    def cifar_train(self):  # balanced target (50,000) as uint8 NHWC + labels
        return os.path.join(self.root, "cifar10/cifar10_train.npz")

    @property
    def cifar_test(self):
        return os.path.join(self.root, "cifar10/cifar10_test.npz")

    @property
    def labels(self):  # recovered labels for D_bias / D_ref (EDA + analysis only)
        return os.path.join(self.root, "tiw_dsm/recovered_labels.npz")


def load_samples(path: str) -> np.ndarray:
    arr = np.load(path)["samples"]
    assert arr.ndim == 4 and arr.shape[1:] == (32, 32, 3) and arr.dtype == np.uint8, arr.shape
    return arr


def load_bias_ref(paths: Paths):
    return load_samples(paths.bias), load_samples(paths.ref)


def build_obs(bias: np.ndarray, ref: np.ndarray, ref_repeat: int = 20):
    """D_obs = D_bias U (D_ref repeated). Returns images and a source flag (1 = reference)."""
    images = np.concatenate([bias] + [ref] * ref_repeat, axis=0)
    source = np.concatenate([np.zeros(len(bias), np.int64), np.ones(len(ref) * ref_repeat, np.int64)])
    return images, source


def to_tensor(images_uint8: np.ndarray) -> torch.Tensor:
    """uint8 NHWC [0,255] -> float32 NCHW [-1,1] (same normalisation as Deliverable 2)."""
    x = torch.from_numpy(np.ascontiguousarray(images_uint8)).permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """float NCHW [-1,1] -> uint8 NHWC."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


class ArrayDataset(Dataset):
    """Images + an integer tag (source flag). Random horizontal flip optional.

    Class labels are deliberately NOT returned: TIW-DSM only knows the source of an
    image, never its class.
    """

    def __init__(self, images: np.ndarray, tags: np.ndarray | None = None, flip: bool = True):
        self.images = images
        self.tags = tags if tags is not None else np.zeros(len(images), np.int64)
        self.flip = flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.flip and np.random.rand() < 0.5:
            img = img[:, ::-1, :]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return x, int(self.tags[i])


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def sha1_rows(images: np.ndarray):
    return [hashlib.sha1(im.tobytes()).digest() for im in images]


def recover_labels(images: np.ndarray, official_images: np.ndarray, official_labels: np.ndarray) -> np.ndarray:
    """Exact pixel-hash matching to the official CIFAR-10 train set (from Deliverable 2)."""
    lookup = {h: int(l) for h, l in zip(sha1_rows(official_images), official_labels)}
    labels = np.asarray([lookup.get(h, -1) for h in sha1_rows(images)], np.int64)
    return labels


def make_synthetic(root: str, n_bias: int = 400, n_ref: int = 40, n_train: int = 1000, seed: int = 0):
    """Tiny synthetic stand-in with the same file layout. Used only for CPU smoke tests."""
    rng = np.random.default_rng(seed)
    p = Paths(root)

    def blobs(labels):
        imgs = np.zeros((len(labels), 32, 32, 3), np.uint8)
        for i, c in enumerate(labels):
            base = np.array([(c * 25) % 255, (c * 70) % 255, (255 - c * 25) % 255], np.float32)
            noise = rng.normal(0, 25, (32, 32, 3))
            imgs[i] = np.clip(base + noise, 0, 255).astype(np.uint8)
        return imgs

    probs = 0.6 ** np.arange(10); probs /= probs.sum()
    bias_l = rng.choice(10, n_bias, p=probs)
    ref_l = rng.integers(0, 10, n_ref)
    tr_l = np.repeat(np.arange(10), n_train // 10)
    te_l = np.repeat(np.arange(10), 20)
    for path, arr in [(p.bias, blobs(bias_l)), (p.ref, blobs(ref_l))]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, samples=arr)
    os.makedirs(os.path.dirname(p.cifar_train), exist_ok=True)
    np.savez(p.cifar_train, images=blobs(tr_l), labels=tr_l)
    np.savez(p.cifar_test, images=blobs(te_l), labels=te_l)
    np.savez(p.labels, bias=bias_l, ref=ref_l)
    return p
