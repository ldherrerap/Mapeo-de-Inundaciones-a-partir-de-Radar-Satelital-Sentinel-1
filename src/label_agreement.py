"""
¿Qué tan ruidosa es la etiqueta débil? Compara S1OtsuLabelHand (umbral de Otsu) con
LabelHand (experto) en los mismos 446 chips dorados. Sirve de evidencia para la ablación (a).

    python -m src.label_agreement --data-root data/sen1floods11/v1.1
"""
import argparse
import glob
import os

import pandas as pd

from .dataset import IGNORE, read_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/sen1floods11/v1.1")
    ap.add_argument("--out", default="results/label_agreement.csv")
    a = ap.parse_args()
    base = os.path.join(a.data_root, "data", "flood_events", "HandLabeled")
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "S1OtsuLabelHand", "*.tif"))):
        cid = "_".join(os.path.basename(p).split("_")[:2])
        gold = read_label(os.path.join(base, "LabelHand", f"{cid}_LabelHand.tif"))
        weak = read_label(p)
        v = (gold != IGNORE) & (weak != IGNORE)
        g, w = gold[v] == 1, weak[v] == 1
        rows.append({"chip_id": cid, "event": cid.split("_")[0],
                     "tp": int((g & w).sum()), "fp": int((~g & w).sum()),
                     "fn": int((g & ~w).sum()), "tn": int((~g & ~w).sum())})
    if not rows:
        raise SystemExit("No se encontró S1OtsuLabelHand; corre scripts/download_data.sh")

    ev = pd.DataFrame(rows).groupby("event")[["tp", "fp", "fn", "tn"]].sum()
    ev.loc["TOTAL"] = ev.sum()
    n = ev[["tp", "fp", "fn", "tn"]].sum(axis=1)
    ev["iou_agua"] = ev.tp / (ev.tp + ev.fp + ev.fn).clip(lower=1)
    ev["acuerdo_pixel"] = (ev.tp + ev.tn) / n
    ev["ruido_pixel"] = 1 - ev["acuerdo_pixel"]
    ev["omision"] = ev.fn / (ev.tp + ev.fn).clip(lower=1)
    ev["comision"] = ev.fp / (ev.tp + ev.fp).clip(lower=1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ev.to_csv(a.out)
    print("Etiqueta débil (Otsu S1) vs etiqueta dorada, por evento:")
    print(ev[["iou_agua", "acuerdo_pixel", "ruido_pixel", "omision", "comision"]].round(3).to_string())


if __name__ == "__main__":
    main()
