"""
Resumen de experimentos y figuras para el informe.

    python -m src.analyze --runs runs --out results

Genera:
  results/summary.csv, results/summary.md     tabla de todas las ablaciones (valid / test / Bolivia)
  results/fig_ablation_iou.png                IoU test vs Bolivia por experimento
  results/<run>/confusion_{test,bolivia}.png  matrices de confusión
  results/<run>/curves.png                    curvas de entrenamiento
  results/<run>/errors_{split}/*.png          paneles VV | VH | S2 RGB | etiqueta | predicción | errores
                                              para los chips con más falsos positivos y falsos negativos
                                              (material para el análisis de fallos del SAR)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch

from .dataset import IGNORE, S1_CLIP, hand_split
from .engine import eval_loader, get_device, load_ckpt, predict_logits

WATER, LAND, NODATA = "#2b6cb0", "#e8e2d0", "#9e9e9e"


# ----------------------------------------------------------------------------- #
def summarize(runs_dir, out_dir):
    rows = []
    for mf in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        name = os.path.basename(os.path.dirname(mf))
        m = json.load(open(mf))
        row = {"experimento": name, "umbral": m["threshold"]}
        for sp in ["valid", "test", "bolivia"]:
            for k in ["iou_total", "iou_mean_chip", "f1", "precision", "recall", "omission", "commission"]:
                row[f"{sp}_{k}"] = m[sp][k]
        row["brecha_test_bolivia"] = m["gap_test_minus_bolivia"]
        rows.append(row)
    if not rows:
        print("No hay corridas con metrics.json"); return None
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    cols = ["experimento", "test_iou_total", "test_f1", "test_precision", "test_recall",
            "bolivia_iou_total", "bolivia_f1", "brecha_test_bolivia"]
    short = df[cols].copy()
    short.columns = ["Experimento", "IoU test", "F1 test", "P test", "R test",
                     "IoU Bolivia", "F1 Bolivia", "Brecha IoU"]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(short.to_markdown(index=False, floatfmt=".3f"))
    print(short.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(df)), 4))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["test_iou_total"], w, label="Test (eventos vistos)", color="#2b6cb0")
    ax.bar(x + w / 2, df["bolivia_iou_total"], w, label="Bolivia (evento no visto)", color="#dd8452")
    ax.set_xticks(x, df["experimento"], rotation=30, ha="right")
    ax.set_ylabel("IoU clase agua"); ax.set_ylim(0, 1); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_ablation_iou.png"), dpi=150)
    plt.close(fig)
    return df


# ----------------------------------------------------------------------------- #
def plot_confusion(conf, title, path):
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]], dtype=float)
    norm = m / m.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norm[i, j]:.2%}\n({int(m[i, j]):,})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks([0, 1], ["No agua", "Agua"]); ax.set_yticks([0, 1], ["No agua", "Agua"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(hist_csv, path):
    h = pd.read_csv(hist_csv)
    h["step"] = np.arange(1, len(h) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 3.2))
    ax1.plot(h["step"], h["train_loss"], color="#555", label="pérdida train")
    ax1.set_xlabel("época (acumulada)"); ax1.set_ylabel("pérdida")
    ax2 = ax1.twinx()
    ax2.plot(h["step"], h["val_iou"], color="#2b6cb0", label="IoU valid")
    ax2.set_ylabel("IoU valid"); ax2.set_ylim(0, 1)
    for ph in h["phase"].unique()[1:]:
        ax1.axvline(h.loc[h["phase"] == ph, "step"].min() - 0.5, ls="--", color="#aaa")
    fig.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _db(img_norm_band, mean, std):
    """Revierte la normalización para mostrar dB."""
    v = img_norm_band.numpy() * std + mean
    return v * (S1_CLIP[1] - S1_CLIP[0]) + S1_CLIP[0]


def _s2_rgb(path):
    if not path or not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        rgb = src.read([4, 3, 2]).astype(np.float32) / 10000.0
    rgb = np.clip(rgb / np.percentile(rgb, 98).clip(min=1e-3), 0, 1)
    return rgb.transpose(1, 2, 0)


def error_panels(cfg, model, device, split, thr, out_dir, k=6):
    from .dataset import S1_MEAN, S1_STD
    os.makedirs(out_dir, exist_ok=True)
    preds = predict_logits(model, eval_loader(cfg, split, 0), device)
    s2_paths = {s.chip_id: s.s1.replace("S1Hand", "S2Hand")
                for s in hand_split(cfg["data_root"], split)}

    stats = []
    for cid, lg, m, img in preds:
        p = torch.sigmoid(lg[0]) > thr
        v = m != IGNORE
        fp = int((p & (m == 0) & v).sum()); fn = int((~p & (m == 1) & v).sum())
        stats.append((cid, fp, fn, lg, m, img))
    worst_fp = sorted(stats, key=lambda s: -s[1])[:k]
    worst_fn = sorted(stats, key=lambda s: -s[2])[:k]

    from matplotlib.colors import ListedColormap
    lab_cmap = ListedColormap([LAND, WATER, NODATA])
    err_cmap = ListedColormap(["#ffffff", "#d62728", "#1f77b4", "#2ca02c", NODATA])

    for kind, group in [("FP", worst_fp), ("FN", worst_fn)]:
        for cid, fp, fn, lg, m, img in group:
            p = (torch.sigmoid(lg[0]) > thr).numpy()
            mm = m.numpy()
            lab = np.where(mm == IGNORE, 2, mm)
            err = np.zeros_like(mm)  # 0 TN, 1 FP, 2 FN, 3 TP, 4 nodata
            err[(p == 1) & (mm == 0)] = 1
            err[(p == 0) & (mm == 1)] = 2
            err[(p == 1) & (mm == 1)] = 3
            err[mm == IGNORE] = 4
            rgb = _s2_rgb(s2_paths.get(cid))
            panels = [("VV (dB)", _db(img[0], S1_MEAN[0], S1_STD[0]), "gray"),
                      ("VH (dB)", _db(img[1], S1_MEAN[1], S1_STD[1]), "gray")]
            n = 5 + (rgb is not None)
            fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.9))
            for ax, (t, a, cm) in zip(axs, panels):
                ax.imshow(a, cmap=cm, vmin=-30, vmax=0); ax.set_title(t, fontsize=8)
            i = 2
            if rgb is not None:
                axs[i].imshow(rgb); axs[i].set_title("S2 RGB (contexto)", fontsize=8); i += 1
            axs[i].imshow(lab, cmap=lab_cmap, vmin=0, vmax=2, interpolation="nearest")
            axs[i].set_title("Etiqueta", fontsize=8)
            axs[i + 1].imshow(p, cmap=ListedColormap([LAND, WATER]), vmin=0, vmax=1,
                              interpolation="nearest")
            axs[i + 1].set_title("Predicción", fontsize=8)
            axs[i + 2].imshow(err, cmap=err_cmap, vmin=0, vmax=4, interpolation="nearest")
            axs[i + 2].set_title("Error: FP rojo, FN azul, TP verde", fontsize=7)
            for ax in axs:
                ax.axis("off")
            fig.suptitle(f"{cid}  |  FP={fp:,}  FN={fn:,}", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{kind}_{cid}.png"), dpi=120)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--panels-for", nargs="*", default=None,
                    help="corridas para generar paneles de error (por defecto: la mejor en test)")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    df = summarize(args.runs, args.out)
    if df is None:
        return
    device = get_device()
    targets = args.panels_for or [df.sort_values("test_iou_total").iloc[-1]["experimento"]]

    for run in sorted(os.listdir(args.runs)):
        rd = os.path.join(args.runs, run)
        if not os.path.exists(os.path.join(rd, "metrics.json")):
            continue
        od = os.path.join(args.out, run); os.makedirs(od, exist_ok=True)
        m = json.load(open(os.path.join(rd, "metrics.json")))
        for sp in ["test", "bolivia"]:
            plot_confusion(m[sp]["confusion"], f"{run} – {sp}", os.path.join(od, f"confusion_{sp}.png"))
        plot_curves(os.path.join(rd, "history.csv"), os.path.join(od, "curves.png"))
        if run in targets:
            model, ck = load_ckpt(os.path.join(rd, "best.pt"), device)
            cfg = ck["cfg"]
            if args.data_root:
                cfg["data_root"] = args.data_root
            for sp in ["test", "bolivia"]:
                error_panels(cfg, model, device, sp, m["threshold"], os.path.join(od, f"errors_{sp}"))
    print(f"Figuras guardadas en {args.out}/")


if __name__ == "__main__":
    main()
