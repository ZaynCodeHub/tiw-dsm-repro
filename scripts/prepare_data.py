"""Download and verify every input the pipeline needs (team implementation).

  1. Official CIFAR-10 LT / 5% arrays (D_bias = fake_data.npz, D_ref = real_data.npz)
  2. Balanced CIFAR-10 train (target distribution, FID reference) and test (classifier check)
  3. Exact-hash label recovery for D_bias / D_ref (analysis only, never used in training)
  4. Authors' pretrained ADM feature extractor (32x32_classifier.pt) for the discriminator
  5. Pretrained CIFAR-10 VGG13-BN classifier for the bias metric, exported to TorchScript

Usage:  python scripts/prepare_data.py            (real data, needs internet)
        python scripts/prepare_data.py --synthetic (tiny fake data for CPU smoke tests)
"""
import argparse
import io
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiwdsm.data import Paths, load_bias_ref, make_synthetic, recover_labels  # noqa: E402

OFFICIAL_IDS = {  # Google-Drive ids of the files linked from the official README (verified in Deliverable 2)
    "bias": "1J1cknM-XpeJ2YOFkTbswL3snYsrgodx-",
    "ref": "1Yp7uJLlaEPavKCisSpJxghEzZFqa4-M2",
}
FEATURE_EXTRACTOR_ID = "1agLe5TbdhXZ7Vpmx3G0Ly9v5dyAAAACc"  # checkpoints/discriminator/feature_extractor/32x32_classifier.pt
CKPT_FOLDER = "https://drive.google.com/drive/folders/1vYLH8UNlXWZarn0IOtiPuU8FvBFqJvTP"
HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def fetch_cifar(split, out):
    if os.path.exists(out):
        return
    import pyarrow.parquet as pq
    from PIL import Image
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".parquet"
    urllib.request.urlretrieve(HF + f"{split}-00000-of-00001.parquet", tmp)
    t = pq.read_table(tmp, columns=["img", "label"])
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"), np.uint8)
                     for c in t["img"].to_pylist()])
    np.savez(out, images=imgs, labels=np.asarray(t["label"].to_pylist(), np.int64))
    os.remove(tmp)


def fetch_feature_extractor(dest):
    if os.path.exists(dest):
        print("feature extractor present:", dest)
        return True
    import gdown
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        gdown.download(id=FEATURE_EXTRACTOR_ID, output=dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
            return True
    except Exception as e:  # noqa: BLE001
        print("direct download failed:", e)
    try:  # list the authors' checkpoint folder without downloading everything
        files = gdown.download_folder(url=CKPT_FOLDER, skip_download=True, quiet=True)
        match = [f for f in files if f.path.replace("\\", "/").endswith("32x32_classifier.pt")]
        if match:
            gdown.download(id=match[0].id, output=dest, quiet=False)
            return os.path.exists(dest)
    except Exception as e:  # noqa: BLE001
        print("automatic download failed:", e)
    print("\n[ACTION] Download 'discriminator/feature_extractor/32x32_classifier.pt' manually from\n"
          f"  {CKPT_FOLDER}\n  and place it at {dest}.\n"
          "  Or train the discriminator with --backbone scratch (weaker, documented fallback).\n")
    return False


class Normalised(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def forward(self, x01):  # x in [0,1]
        return self.net((x01 - self.mean) / self.std)


def export_eval_classifier(dest, test_npz, device):
    if os.path.exists(dest):
        print("eval classifier present:", dest)
    else:
        net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg13_bn", pretrained=True, trust_repo=True)
        model = Normalised(net).eval()
        ts = torch.jit.trace(model, torch.rand(2, 3, 32, 32))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        ts.save(dest)
    clf = torch.jit.load(dest, map_location=device).eval()
    z = np.load(test_npz)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(z["images"]), 500):
            x = torch.from_numpy(z["images"][i:i + 500]).to(device).permute(0, 3, 1, 2).float() / 255
            correct += int((clf(x).argmax(1).cpu().numpy() == z["labels"][i:i + 500]).sum())
    acc = correct / len(z["images"])
    print(f"eval classifier test accuracy: {acc:.4f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip_extractor", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import json

    if a.synthetic:
        p = make_synthetic(a.root)
        print("synthetic data written under", a.root)
        return

    p = Paths(a.root)
    import gdown
    for key, path in [("bias", p.bias), ("ref", p.ref)]:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            gdown.download(id=OFFICIAL_IDS[key], output=path, quiet=False)
    fetch_cifar("train", p.cifar_train)
    fetch_cifar("test", p.cifar_test)

    bias, ref = load_bias_ref(p)
    tr = np.load(p.cifar_train)
    lb, lr = recover_labels(bias, tr["images"], tr["labels"]), recover_labels(ref, tr["images"], tr["labels"])
    assert (lb >= 0).all() and (lr >= 0).all(), "some images did not match CIFAR-10"
    np.savez(p.labels, bias=lb, ref=lr)
    info = {"n_bias": len(bias), "n_ref": len(ref),
            "bias_counts": np.bincount(lb, minlength=10).tolist(),
            "ref_counts": np.bincount(lr, minlength=10).tolist()}
    print(info)

    ok = True if a.skip_extractor else fetch_feature_extractor("checkpoints/feature_extractor/32x32_classifier.pt")
    acc = export_eval_classifier("checkpoints/eval_classifier_vgg13bn.ts", p.cifar_test, device)
    info.update({"feature_extractor_available": ok, "eval_classifier_test_acc": acc})
    os.makedirs("results", exist_ok=True)
    json.dump(info, open("results/data_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
