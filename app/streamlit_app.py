"""TIW-DSM deployment: Streamlit inference app (team implementation).

Run:   streamlit run app/streamlit_app.py
Needs: checkpoints/deploy/tiw_dsm.pt  (EMA weights of our TIW-DSM model)
       checkpoints/deploy/dsm_obs.pt  (optional, for the side-by-side comparison)
       checkpoints/eval_classifier_vgg13bn.ts (optional, adds predicted class labels)

Pipeline shown live:
  Input (seed or uploaded image) -> Preprocessing (seeded Gaussian noise / resize + normalise + noise)
  -> Trained model (EDM-preconditioned U-Net + Heun/Euler ODE solver) -> Prediction (image, class)
  -> Output (grid, class histogram, timing)
"""
import io
import os
import sys
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tiwdsm import CLASS_NAMES  # noqa: E402
from tiwdsm.data import to_uint8  # noqa: E402
from tiwdsm.edm import nfe_for, sample, seeded_latents  # noqa: E402
from tiwdsm.io import file_sha256, load_ema  # noqa: E402

DEPLOY = {"TIW-DSM (ours)": "checkpoints/deploy/tiw_dsm.pt", "DSM(obs) baseline": "checkpoints/deploy/dsm_obs.pt"}
CLF = "checkpoints/eval_classifier_vgg13bn.ts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="TIW-DSM demo", layout="wide")


@st.cache_resource
def get_model(path):
    model, meta = load_ema(os.path.join(ROOT, path), DEVICE)
    return model, meta, file_sha256(os.path.join(ROOT, path))


@st.cache_resource
def get_classifier():
    p = os.path.join(ROOT, CLF)
    return torch.jit.load(p, map_location=DEVICE).eval() if os.path.exists(p) else None


def classify(images_uint8):
    clf = get_classifier()
    if clf is None:
        return None, None
    with torch.no_grad():
        x = torch.from_numpy(images_uint8).to(DEVICE).permute(0, 3, 1, 2).float() / 255
        p = torch.softmax(clf(x), 1).cpu().numpy()
    return p.argmax(1), p.max(1)


def grid(images, nrow, scale=4):
    n = len(images); rows = (n + nrow - 1) // nrow
    g = np.full((rows * 34 + 2, nrow * 34 + 2, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, nrow)
        g[2 + r * 34:34 + r * 34, 2 + c * 34:34 + c * 34] = im
    return Image.fromarray(g).resize((g.shape[1] * scale, g.shape[0] * scale), Image.NEAREST)


def run(model, z, steps, solver, x_init=None, sigma_max=80.0, traj=False):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = sample(model, z, num_steps=steps, solver=solver, x_init=x_init, sigma_max=sigma_max, return_trajectory=traj)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t0


available = {k: v for k, v in DEPLOY.items() if os.path.exists(os.path.join(ROOT, v))}
st.title("Unbiased image generation with TIW-DSM")
st.caption("Reproduction of Kim et al., *Training Unbiased Diffusion Models From Biased Dataset* (ICLR 2024). "
           "Trained on CIFAR-10 LT (100:1 imbalance) + a 5% unbiased reference set.")
if not available:
    st.error("No checkpoint found. Copy your trained EMA weights to checkpoints/deploy/tiw_dsm.pt (see README).")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(available))
    solver = st.radio("ODE solver", ["heun", "euler"], horizontal=True)
    steps = st.select_slider("Sampling steps", options=[3, 5, 9, 13, 18, 25, 35, 49], value=18 if solver == "heun" else 35)
    st.write(f"Network evaluations (NFE): **{nfe_for(solver, steps)}**")
    model, meta, sha = get_model(available[model_name])
    st.divider()
    st.subheader("Active model")
    st.json({"checkpoint": available[model_name], "sha256": sha, "objective": meta.get("mode"),
             "architecture": f"U-Net '{meta.get('preset')}' ({meta.get('n_params', 0) / 1e6:.1f}M params)",
             "trained images (M)": round(meta.get("nimg", 0) / 1e6, 2), "device": DEVICE})

tab1, tab2, tab3 = st.tabs(["Generate from a seed", "Image-to-image from your upload", "How it works"])

with tab1:
    c1, c2, c3 = st.columns([1, 1, 1])
    seed = c1.number_input("Seed (type any number, e.g. one the examiner gives)", 0, 2**31 - 1, 2026, step=1)
    n = c2.select_slider("Number of images", options=[1, 4, 8, 16, 32, 64], value=16)
    compare = c3.checkbox("Compare with the other model (same noise)", value=len(available) > 1,
                          disabled=len(available) < 2)
    show_traj = c3.checkbox("Show denoising trajectory (first image)")
    if st.button("Generate", type="primary"):
        z = seeded_latents(range(int(seed), int(seed) + n), DEVICE)
        names = [model_name] + ([k for k in available if k != model_name] if compare else [])
        cols = st.columns(len(names))
        for col, name in zip(cols, names):
            m, mt, _ = get_model(available[name])
            res, sec = run(m, z, steps, solver, traj=show_traj and name == model_name)
            x, tr = res if isinstance(res, tuple) else (res, None)
            imgs = to_uint8(x)
            with col:
                st.subheader(name)
                st.image(grid(imgs, nrow=min(8, n)))
                st.write(f"{n} image(s) in **{sec:.2f} s** ({1000 * sec / n:.0f} ms/image, NFE {nfe_for(solver, steps)}, {DEVICE})")
                pred, conf = classify(imgs)
                if pred is not None:
                    hist = np.bincount(pred, minlength=10)
                    st.bar_chart({c: int(h) for c, h in zip(CLASS_NAMES, hist)}, height=200)
                    st.caption("Predicted classes (VGG13-BN): " + ", ".join(f"{CLASS_NAMES[p]} ({c:.2f})" for p, c in zip(pred[:16], conf[:16])))
                buf = io.BytesIO(); grid(imgs, nrow=min(8, n)).save(buf, "PNG")
                st.download_button("Download PNG", buf.getvalue(), f"{name.split()[0]}_seed{seed}.png", key=f"dl{name}")
            if tr is not None:
                st.write("Denoised estimate $\\hat x_0$ at each step (first image):")
                st.image(grid(to_uint8(torch.cat([t[:1] for t in tr])), nrow=len(tr), scale=3))

with tab2:
    st.write("Upload **any** image. It is centre-cropped, resized to 32x32, scaled to [-1, 1], noised to the "
             "chosen level and then denoised by the trained model (SDEdit). Higher strength lets the "
             "model change more.")
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    strength = st.slider("Strength = starting noise level sigma", 0.2, 20.0, 2.0, 0.1)
    seed2 = st.number_input("Noise seed", 0, 2**31 - 1, 0, key="seed2")
    if up is not None:
        img = Image.open(up).convert("RGB")
        s = min(img.size)
        img = img.crop(((img.width - s) // 2, (img.height - s) // 2, (img.width + s) // 2, (img.height + s) // 2))
        small = np.asarray(img.resize((32, 32), Image.BICUBIC), np.uint8)
        x0 = torch.from_numpy(small).permute(2, 0, 1)[None].float().to(DEVICE) / 127.5 - 1
        z = seeded_latents([int(seed2)], DEVICE)
        (x, tr), sec = run(model, z, steps, solver, x_init=x0, sigma_max=float(strength), traj=True)
        noisy = to_uint8((x0 + strength * z).clamp(-1, 1))
        out = to_uint8(x)
        c1, c2, c3 = st.columns(3)
        c1.image(Image.fromarray(small).resize((192, 192), Image.NEAREST), caption="1. preprocessed input (32x32)")
        c2.image(Image.fromarray(noisy[0]).resize((192, 192), Image.NEAREST), caption=f"2. noised, sigma={strength}")
        c3.image(Image.fromarray(out[0]).resize((192, 192), Image.NEAREST), caption=f"3. model output ({sec:.2f} s)")
        pi, _ = classify(small[None]); po, co = classify(out)
        if pi is not None:
            st.write(f"Classifier: input looks like **{CLASS_NAMES[pi[0]]}**, output looks like **{CLASS_NAMES[po[0]]}** ({co[0]:.2f}).")
        st.image(grid(to_uint8(torch.cat([t[:1] for t in tr])), nrow=len(tr), scale=3), caption="denoising trajectory")

with tab3:
    st.markdown("""
**Pipeline:** input (seed or image) → preprocessing (seeded N(0, I) noise × σ_max, or resize + [-1,1] + noise)
→ trained model (EDM-preconditioned U-Net trained with TIW-DSM) → deterministic ODE solve (Heun/Euler)
→ prediction (image in [-1,1] → uint8; class predicted by a separate CIFAR-10 classifier) → output.

**Why TIW-DSM?** The training set is 64% airplanes + automobiles. A normal diffusion model copies that bias.
TIW-DSM uses a time-dependent discriminator (trained once, *not needed at inference*) to reweight training
samples and correct the score toward the unbiased distribution.

**Proof the output comes from the model:** change the seed or upload a new image, and the result changes.
The checkpoint hash is shown in the sidebar, and the trajectory view shows every intermediate network prediction.
""")
