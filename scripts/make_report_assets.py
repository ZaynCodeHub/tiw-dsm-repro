"""Turn results/*.json and runs/*/log.jsonl into report tables, figures and number macros.

Nothing here is typed by hand. Every number in the report's results sections comes from the
JSON files written by the evaluation scripts. Missing results stay visible as "pending".

  python scripts/make_report_assets.py --runs obs=runs/obs tiw=runs/tiw
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiwdsm import CLASS_NAMES  # noqa: E402

FIG = "report/figures"
GEN = "report/generated"
LABEL = {"obs": "DSM(obs)", "tiw": "TIW-DSM", "iw": "IW-DSM", "ref": "DSM(ref)",
         "tiwW": "TIW (W only)", "tiwC": "TIW (C only)"}
COLORS = {"obs": "#E8871E", "tiw": "#2F75B5", "iw": "#8E6BB8", "ref": "#5F6B75", "tiwW": "#18A999", "tiwC": "#C0392B"}
PAPER = {"ref": 16.47, "obs": 12.99, "iw": 15.79, "tiw": 11.51, "tiwW": 13.27, "tiwC": 11.62}
PENDING = r"\pending"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(v, nd=2):
    return PENDING if v is None else f"{v:.{nd}f}"


def macro(name, value):
    return f"\\def\\{name}{{{value}}}\n"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="name=dir pairs, e.g. obs=runs/obs tiw=runs/tiw")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True); os.makedirs(GEN, exist_ok=True)
    plt = plt_setup()
    runs = dict(r.split("=", 1) for r in a.runs)
    M = ""  # number macros

    # ---------------------------------------------------------------- data / setup info
    info = load("results/data_info.json") or {}
    M += macro("resClfAcc", fmt(100 * info["eval_classifier_test_acc"], 1) if "eval_classifier_test_acc" in info else PENDING)

    # ---------------------------------------------------------------- discriminator
    dtrain, diag = load("results/discriminator_train.json"), load("results/discriminator_diag.json")
    if dtrain:
        M += macro("resDiscMinutes", fmt(dtrain["train_seconds"] / 60, 0))
        M += macro("resDiscFinalAcc", fmt(100 * np.mean([r["acc"] for r in dtrain["log"][-10:]]), 1))
        M += macro("resDiscBackbone", dtrain["args"]["backbone"].upper())
    if diag:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        t = [r["tau"] for r in diag]
        ax[0].plot(t, [r["balanced_acc"] for r in diag], "o-", color="#2F75B5")
        ax[0].axhline(0.5, ls="--", color="grey"); ax[0].set_xlabel("VP time $t$"); ax[0].set_ylabel("balanced accuracy")
        ax[0].set_title("Separability of $D_{ref}$ vs $D_{bias}$")
        ax[1].plot(t, [r["mean_d_ref"] for r in diag], "o-", label="mean $d$ on $D_{ref}$", color="#18A999")
        ax[1].plot(t, [r["mean_d_bias"] for r in diag], "s-", label="mean $d$ on $D_{bias}$", color="#E8871E")
        ax[1].set_xlabel("VP time $t$"); ax[1].set_ylabel("$d_\\phi(x_t,t)$"); ax[1].legend(); ax[1].set_title("Discriminator output")
        fig.tight_layout(); fig.savefig(f"{FIG}/disc_diag.png", dpi=170); plt.close(fig)
        rows = "".join(f"{r['tau']:.2f} & {r['sigma_edm']:.3g} & {100 * r['balanced_acc']:.1f} & {r['mean_d_ref']:.3f} & "
                       f"{r['mean_d_bias']:.3f} & {r['w_bias_median']:.3f} \\\\\n" for r in diag)
        open(f"{GEN}/disc_table.tex", "w").write(rows)
        M += macro("resDiscAccLow", fmt(100 * diag[0]["balanced_acc"], 1))
        M += macro("resDiscAccHigh", fmt(100 * diag[-1]["balanced_acc"], 1))

    # ---------------------------------------------------------------- training curves
    if runs:
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))
        for name, d in runs.items():
            lp = os.path.join(d, "log.jsonl")
            if not os.path.exists(lp):
                continue
            L = [json.loads(l) for l in open(lp)]
            k = [r["kimg"] / 1000 for r in L]
            ax[0].plot(k, [r["loss"] for r in L], label=LABEL.get(name, name), color=COLORS.get(name))
            if "w_mean" in L[0]:
                ax[1].plot(k, [r["w_mean"] for r in L], label=f"{LABEL.get(name, name)} mean $\\tilde w$", color=COLORS.get(name))
                ax[1].plot(k, [r["w_std"] for r in L], ls="--", label=f"std $\\tilde w$", color=COLORS.get(name), alpha=0.6)
            ax[2].plot(k, [r["sec_per_kimg"] for r in L], label=LABEL.get(name, name), color=COLORS.get(name))
            args = load(os.path.join(d, "args.json")) or {}
            M += macro(f"resHours{name}", fmt(L[-1]["train_hours"], 1))
            M += macro(f"resMimg{name}", fmt(L[-1]["kimg"] / 1000, 2))
            M += macro(f"resSecKimg{name}", fmt(float(np.median([r["sec_per_kimg"] for r in L])), 1))
            if "n_params" in args:
                M += macro("resParams", fmt(args["n_params"] / 1e6, 1))
            grids = sorted(glob.glob(os.path.join(d, "samples_*k.png")))
            if grids:
                shutil.copy(grids[-1], f"{FIG}/samples_{name}.png")
        ax[0].set_yscale("log"); ax[0].set_title("training loss"); ax[0].set_xlabel("Mimg"); ax[0].legend()
        ax[1].set_title("TIW weight $\\tilde w = 2d$"); ax[1].set_xlabel("Mimg"); ax[1].legend(fontsize=7)
        ax[2].set_title("seconds per kimg"); ax[2].set_xlabel("Mimg")
        fig.tight_layout(); fig.savefig(f"{FIG}/training_curves.png", dpi=170); plt.close(fig)

    # ---------------------------------------------------------------- main results
    order = ["ref", "obs", "iw", "tiwW", "tiwC", "tiw"]
    ev = {n: load(f"results/eval_{n}.json") for n in order}
    real = {n: load(f"results/eval_data_{n}.json") for n in ["bias", "ref", "target"]}
    rows = ""
    for n in order:
        e = ev[n]
        if e is None and n not in ("obs", "tiw"):
            continue
        g = lambda k, nd=2, s=1: fmt(e[k] * s if e and k in e else None, nd)  # noqa: E731
        rows += (f"{LABEL[n]} & {fmt(PAPER.get(n))} & {g('fid')} & {g('precision', 3)} & {g('recall', 3)} & "
                 f"{g('bias_eq51', 3)} & {g('tail_share', 1, 100)} & {g('l1_to_uniform', 3)} \\\\\n")
        if e:
            M += macro(f"resFID{n}", g("fid")); M += macro(f"resRecall{n}", g("recall", 3))
            M += macro(f"resBias{n}", g("bias_eq51", 3)); M += macro(f"resTail{n}", g("tail_share", 1, 100))
            M += macro(f"resLone{n}", g("l1_to_uniform", 3)); M += macro(f"resNimg{n}", f"{e['n_images']:,}")
            M += macro(f"resPrec{n}", g("precision", 3))
    open(f"{GEN}/main_rows.tex", "w").write(rows)
    rrows = ""
    for n, lab in [("target", "CIFAR-10 test (balanced, 10k)"), ("bias", "$D_{bias}$ itself (10k)"), ("ref", "$D_{ref}$ itself (500)")]:
        e = real[n]
        g = lambda k, nd=2, s=1: fmt(e[k] * s if e and k in e else None, nd)  # noqa: E731
        rrows += (f"{lab} & -- & {g('fid')} & {g('precision', 3)} & {g('recall', 3)} & {g('bias_eq51', 3)} & "
                  f"{g('tail_share', 1, 100)} & {g('l1_to_uniform', 3)} \\\\\n")
    open(f"{GEN}/real_rows.tex", "w").write(rrows)
    if real["target"]:
        M += macro("resFIDtestfloor", fmt(real["target"]["fid"]))
    if ev["obs"] and ev["tiw"]:
        M += macro("resFIDgain", fmt(ev["obs"]["fid"] - ev["tiw"]["fid"]))
        M += macro("resBiasRed", fmt(100 * (1 - ev["tiw"]["bias_eq51"] / ev["obs"]["bias_eq51"]), 1))
        M += macro("resMsImg", fmt(ev["tiw"].get("ms_per_image", float("nan")), 2))

    # class-share figure + table
    labels = load("results/data_info.json")
    fig, ax = plt.subplots(figsize=(10, 3.2)) if (ev["obs"] or ev["tiw"]) else (None, None)
    x = np.arange(10); w = 0.2
    bars = [("Target (uniform)", np.full(10, 0.1), "#BBBBBB")]
    if labels and "bias_counts" in labels:
        bc = np.asarray(labels["bias_counts"], float); bars.append(("$D_{bias}$ (exact labels)", bc / bc.sum(), "#5F6B75"))
    for n in ["obs", "tiw"]:
        if ev[n]:
            bars.append((LABEL[n] + " (classifier)", np.asarray(ev[n]["class_share"]), COLORS[n]))
    if fig is not None:
        for i, (lab, v, c) in enumerate(bars):
            ax.bar(x + (i - (len(bars) - 1) / 2) * w, v, w, label=lab, color=c)
        ax.set_xticks(x, CLASS_NAMES, rotation=20); ax.set_ylabel("share of images"); ax.legend(fontsize=8, ncol=2)
        ax.set_title("Class proportions: target, biased training data and generated samples")
        fig.tight_layout(); fig.savefig(f"{FIG}/class_share.png", dpi=170); plt.close(fig)
    crow = ""
    for i, c in enumerate(CLASS_NAMES):
        cells = [c.capitalize(), "10.0"]
        cells.append(f"{100 * labels['bias_counts'][i] / sum(labels['bias_counts']):.1f}" if labels and "bias_counts" in labels else PENDING)
        for n in ["obs", "tiw"]:
            cells.append(f"{100 * ev[n]['class_share'][i]:.1f}" if ev[n] else PENDING)
        crow += " & ".join(cells) + " \\\\\n"
    open(f"{GEN}/class_rows.tex", "w").write(crow)

    # ---------------------------------------------------------------- steps experiment
    st = load("results/steps_experiment.json")
    if st:
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.1))
        for model in sorted({r["model"] for r in st}):
            for solver, ls in [("heun", "-"), ("euler", ":")]:
                R = sorted([r for r in st if r["model"] == model and r["solver"] == solver], key=lambda r: r["nfe"])
                if not R:
                    continue
                nfe = [r["nfe"] for r in R]
                kw = dict(ls=ls, marker="o" if solver == "heun" else "s", ms=3, color=COLORS.get(model),
                          label=f"{LABEL.get(model, model)}, {solver}")
                axs[0].plot(nfe, [r["fid"] for r in R], **kw)
                axs[1].plot(nfe, [r["recall"] for r in R], **kw)
                axs[2].plot(nfe, [r["bias_eq51"] for r in R], **kw)
                axs[3].plot(nfe, [r["ms_per_image_batched"] for r in R], **kw)
        for axx, t in zip(axs, ["FID-10k (lower better)", "Recall (higher better)", "Bias, Eq. 51 (lower better)", "ms / image (batched)"]):
            axx.set_xscale("log"); axx.set_xlabel("NFE"); axx.set_title(t, fontsize=9)
        axs[0].set_yscale("log"); axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{FIG}/steps_experiment.png", dpi=170); plt.close(fig)
        srows = ""
        for r in sorted(st, key=lambda r: (r["model"] != "obs", r["solver"], r["nfe"])):
            srows += (f"{LABEL.get(r['model'], r['model'])} & {r['solver'].capitalize()} & {r['steps']} & {r['nfe']} & "
                      f"{r['fid']:.2f} & {r['recall']:.3f} & {r['bias_eq51']:.3f} & {100 * r['tail_share']:.1f} & "
                      f"{r['ms_per_image_batched']:.2f} & {r['ms_single_image']:.0f} \\\\\n")
        open(f"{GEN}/steps_rows.tex", "w").write(srows)
        M += macro("resStepsDevice", st[0].get("device", "GPU").replace("_", " "))
        M += macro("resStepsN", f"{st[0]['n_images']:,}")
        # findings used in text
        for model in ["obs", "tiw"]:
            H = {r["steps"]: r for r in st if r["model"] == model and r["solver"] == "heun"}
            if 18 in H:
                M += macro(f"resStepsFIDref{model}", fmt(H[18]["fid"]))
                M += macro(f"resStepsBiasref{model}", fmt(H[18]["bias_eq51"], 3))
        R = [r for r in st if r["model"] == "tiw" and r["solver"] == "heun"]
        if R:
            best = min(r["fid"] for r in R)
            ok = sorted([r for r in R if r["fid"] <= 1.10 * best], key=lambda r: r["nfe"])
            if ok:
                M += macro("resStepsKneeNFE", str(ok[0]["nfe"]))
                M += macro("resStepsKneeFID", fmt(ok[0]["fid"]))

    # ---------------------------------------------------------------- toy
    toy = load("results/toy_2d.json")
    if toy:
        for k in ["ref", "obs", "iw", "tiw_W", "tiw_C", "tiw"]:
            if k in toy:
                M += macro("resToy" + k.replace("_", ""), fmt(toy[k]["minor_mode_fraction"], 3))
                M += macro("resToyStd" + k.replace("_", ""), fmt(toy[k].get("std", 0.0), 3))

    open(f"{GEN}/numbers.tex", "w").write("% auto-generated by scripts/make_report_assets.py\n" + M)
    print(f"wrote {GEN}/numbers.tex with {M.count(chr(10))} values; figures in {FIG}/")


if __name__ == "__main__":
    main()
