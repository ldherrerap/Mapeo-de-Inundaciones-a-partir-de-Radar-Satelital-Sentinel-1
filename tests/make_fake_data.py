"""
Crea un mini Sen1Floods11 sintético con la MISMA estructura de carpetas y nombres que v1.1.
Sirve para probar el pipeline completo sin descargar datos:

    python tests/make_fake_data.py --out data/fake --size 128
    QUICK=1 DATA_ROOT=data/fake bash run_all.sh
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin


def write(path, arr, dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=dtype, crs="EPSG:4326",
                       transform=from_origin(0, 0, 1e-4, 1e-4)) as dst:
        dst.write(arr.astype(dtype))


def chip(rng, size):
    yy, xx = np.mgrid[:size, :size]
    cx, cy, r = rng.uniform(0, size, 3) * [1, 1, 0.4]
    water = ((xx - cx) ** 2 + (yy - cy) ** 2) < (r + 8) ** 2
    vv = np.where(water, -20, -9) + rng.normal(0, 2.5, (size, size))
    vh = np.where(water, -27, -16) + rng.normal(0, 2.5, (size, size))
    s1 = np.stack([vv, vh]).astype(np.float32)
    s1[:, :3, :3] = np.nan  # simula NaN del SAR
    lab = water.astype(np.int16)
    lab[-6:, :] = -1  # simula "sin datos"
    s2 = rng.integers(200, 1800, (13, size, size)).astype(np.uint16)
    s2[:, water] //= 4
    return s1, lab, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    hand = os.path.join(a.out, "data", "flood_events", "HandLabeled")
    weak = os.path.join(a.out, "data", "flood_events", "WeaklyLabeled")
    splits = {"train": [], "valid": [], "test": [], "bolivia": []}
    cid = 1000
    for ev, n_tr, n_va, n_te in [("Ghana", 6, 2, 2), ("India", 6, 2, 2), ("Spain", 4, 1, 1)]:
        for sp, n in [("train", n_tr), ("valid", n_va), ("test", n_te)]:
            for _ in range(n):
                cid += 1; splits[sp].append(f"{ev}_{cid}")
    for _ in range(4):
        cid += 1; splits["bolivia"].append(f"Bolivia_{cid}")

    for sp, ids in splits.items():
        for c in ids:
            s1, lab, s2 = chip(rng, a.size)
            write(f"{hand}/S1Hand/{c}_S1Hand.tif", s1, "float32")
            write(f"{hand}/LabelHand/{c}_LabelHand.tif", lab, "int16")
            write(f"{hand}/S2Hand/{c}_S2Hand.tif", s2, "uint16")
            otsu = np.clip(lab, 0, 1); otsu[rng.random(lab.shape) < 0.05] ^= 1
            write(f"{hand}/S1OtsuLabelHand/{c}_S1OtsuLabelHand.tif", otsu, "int16")
        os.makedirs(os.path.join(a.out, "splits", "flood_handlabeled"), exist_ok=True)
        with open(os.path.join(a.out, "splits", "flood_handlabeled", f"flood_{sp}_data.csv"), "w") as f:
            for c in ids:
                f.write(f"{c}_S1Hand.tif,{c}_LabelHand.tif\n")

    for ev in ["Ghana", "India", "Spain", "Bolivia"]:
        for _ in range(8):
            cid += 1; c = f"{ev}_{cid}"
            s1, lab, _ = chip(rng, a.size)
            noisy = lab.copy(); noisy[rng.random(lab.shape) < 0.05] ^= 1  # etiqueta ruidosa
            write(f"{weak}/S1Weak/{c}_S1Weak.tif", s1, "float32")
            write(f"{weak}/S1OtsuLabelWeak/{c}_S1OtsuLabelWeak.tif", np.clip(noisy, 0, 1), "int16")
            write(f"{weak}/S2IndexLabelWeak/{c}_S2IndexLabelWeak.tif", np.clip(lab, 0, 1), "int16")
    print(f"Datos sintéticos en {a.out}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
