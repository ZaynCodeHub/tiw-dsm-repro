"""Fallback CIFAR-10 classifier for the bias metric (team implementation).

Only needed if torch.hub cannot download the pretrained VGG13-BN in prepare_data.py,
or for the synthetic smoke test. Trains a small ResNet on the balanced CIFAR-10
training set and exports TorchScript to the same path the evaluation code reads.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import Normalised  # noqa: E402
from tiwdsm.data import Paths  # noqa: E402


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


def small_resnet(w=64):
    return nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                         Block(w, w, 1), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2), Block(4 * w, 8 * w, 2),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8 * w, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", default="checkpoints/eval_classifier_vgg13bn.ts")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = Paths(a.root)
    tr, te = np.load(p.cifar_train), np.load(p.cifar_test)
    X = torch.from_numpy(tr["images"]).permute(0, 3, 1, 2).float() / 255; Y = torch.from_numpy(tr["labels"])
    model = Normalised(small_resnet(a.width)).to(dev)
    opt = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = a.epochs * (len(X) // 128 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            x = X[idx].to(dev); y = Y[idx].to(dev)
            if torch.rand(()) < 0.5:
                x = x.flip(3)
            x = F.pad(x, (4, 4, 4, 4), mode="reflect")
            dx, dy = np.random.randint(0, 9, 2)
            x = x[:, :, dy:dy + 32, dx:dx + 32]
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(te["images"]).permute(0, 3, 1, 2).float().div(255)
            pred = torch.cat([model(xt[i:i + 500].to(dev)).argmax(1).cpu() for i in range(0, len(xt), 500)])
            acc = float((pred.numpy() == te["labels"]).mean())
        print(f"epoch {ep + 1}: test acc {acc:.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.jit.trace(model.cpu().eval(), torch.rand(2, 3, 32, 32)).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
