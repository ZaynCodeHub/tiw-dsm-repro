"""TIW-DSM reproduction package (team implementation).

Modules
-------
data           : loading the official CIFAR-10 LT / 5% arrays, building D_obs     [team]
unet           : compact DDPM++-style U-Net score network                         [team, design follows DDPM++/EDM]
edm            : EDM preconditioning, noise sampling and Heun/Euler samplers      [adapted from EDM]
vpsde          : EDM-sigma <-> VP-time conversion for the discriminator           [adapted from official TIW-DSM]
discriminator  : time-dependent discriminator, density ratio and its gradient     [team, adapted from official]
losses         : DSM(obs), DSM(ref), IW-DSM and TIW-DSM objectives                [team, from the paper's equations]
metrics        : FID, precision/recall, class-distribution bias metrics            [team, uses pytorch-fid Inception]
"""
import os
import sys

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
