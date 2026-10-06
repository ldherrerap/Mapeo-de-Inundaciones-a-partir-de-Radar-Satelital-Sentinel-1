"""
Carga de Sen1Floods11 (v1.1) para segmentación de agua con U-Net.

Estructura esperada (tal como queda tras scripts/download_data.sh):

data/sen1floods11/v1.1/
├── data/flood_events/HandLabeled/{S1Hand, LabelHand, S2Hand}/EVENT_CHIP_<LAYER>.tif
├── data/flood_events/WeaklyLabeled/{S1Weak, S1OtsuLabelWeak, S2IndexLabelWeak}/...
└── splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv

Etiquetas: -1 = sin datos (se ignora), 0 = no agua, 1 = agua.
"""
from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

IGNORE = 255  # índice ignorado en pérdidas y métricas

# Normalización del notebook oficial (Train.ipynb de Cloud to Street):
# dB recortado a [-50, 1], escalado a [0, 1] y luego estandarizado.
S1_CLIP = (-50.0, 1.0)
S1_MEAN = np.array([0.6851, 0.5235], dtype=np.float32)  # VV, VH
S1_STD = np.array([0.0820, 0.1102], dtype=np.float32)

# Bandas de Sentinel-2 usadas en el bono de fusión (índices 0-based del GeoTIFF de 13 bandas)
S2_BANDS = [1, 2, 3, 7, 11, 12]  # B2, B3, B4, B8, B11, B12
S2_SCALE = 10000.0


@dataclass
class Sample:
    s1: str
    label: str
    s2: str | None = None
    weak: bool = False  # True si la etiqueta es débil (umbralización automática)

    @property
    def event(self) -> str:
        return os.path.basename(self.s1).split("_")[0]

    @property
    def chip_id(self) -> str:
        return "_".join(os.path.basename(self.s1).split("_")[:2])


# ----------------------------------------------------------------------------- #
# Construcción de listas de muestras
# ----------------------------------------------------------------------------- #
def _hand_dirs(root: str):
    base = os.path.join(root, "data", "flood_events", "HandLabeled")
    return (os.path.join(base, "S1Hand"), os.path.join(base, "LabelHand"),
            os.path.join(base, "S2Hand"))


def _read_split_csv(path: str) -> list[str]:
    """Devuelve los IDs EVENT_CHIP listados en un CSV oficial de partición."""
    ids = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            name = os.path.basename(row[0].strip())
            ids.append("_".join(name.split("_")[:2]))
    return ids


def hand_split(root: str, split: str, use_s2: bool = False) -> list[Sample]:
    """split ∈ {train, valid, test, bolivia} usando la partición oficial.

    Si el CSV de Bolivia no existe, se construye a partir del prefijo del nombre.
    """
    s1_dir, lab_dir, s2_dir = _hand_dirs(root)
    split_dir = os.path.join(root, "splits", "flood_handlabeled")
    csv_path = os.path.join(split_dir, f"flood_{split}_data.csv")

    if os.path.exists(csv_path):
        ids = _read_split_csv(csv_path)
    elif split == "bolivia":
        ids = ["_".join(os.path.basename(p).split("_")[:2])
               for p in sorted(glob.glob(os.path.join(s1_dir, "Bolivia_*_S1Hand.tif")))]
    else:
        raise FileNotFoundError(f"No se encontró la partición oficial: {csv_path}")

    samples = []
    for cid in ids:
        s = Sample(
            s1=os.path.join(s1_dir, f"{cid}_S1Hand.tif"),
            label=os.path.join(lab_dir, f"{cid}_LabelHand.tif"),
            s2=os.path.join(s2_dir, f"{cid}_S2Hand.tif") if use_s2 else None,
        )
        if os.path.exists(s.s1) and os.path.exists(s.label):
            samples.append(s)
    if not samples:
        raise RuntimeError(f"Partición '{split}' vacía. ¿Descargaste los datos en {root}?")
    return samples


def weak_samples(root: str, source: str = "S1OtsuLabelWeak",
                 exclude_events: tuple[str, ...] = ("Bolivia",),
                 max_chips: int | None = None, seed: int = 42) -> list[Sample]:
    """Chips con etiqueta débil. source ∈ {S1OtsuLabelWeak, S2IndexLabelWeak}.

    Se excluye Bolivia para que el test geográfico siga siendo independiente.
    """
    base = os.path.join(root, "data", "flood_events", "WeaklyLabeled")
    s1_dir, lab_dir = os.path.join(base, "S1Weak"), os.path.join(base, source)
    samples = []
    for p in sorted(glob.glob(os.path.join(s1_dir, "*_S1Weak.tif"))):
        cid = "_".join(os.path.basename(p).split("_")[:2])
        if cid.split("_")[0] in exclude_events:
            continue
        lab = os.path.join(lab_dir, f"{cid}_{source}.tif")
        if os.path.exists(lab):
            samples.append(Sample(s1=p, label=lab, weak=True))
    if max_chips:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_chips]
    return samples


# ----------------------------------------------------------------------------- #
# Lectura y normalización
# ----------------------------------------------------------------------------- #
def read_s1(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        x = src.read([1, 2]).astype(np.float32)  # (2, H, W): VV, VH en dB
    x = np.nan_to_num(x, nan=S1_CLIP[0])
    x = np.clip(x, *S1_CLIP)
    x = (x - S1_CLIP[0]) / (S1_CLIP[1] - S1_CLIP[0])
    x = (x - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
    return x


def read_s2(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        x = src.read([b + 1 for b in S2_BANDS]).astype(np.float32)
    x = np.nan_to_num(x) / S2_SCALE
    return np.clip(x, 0, 1) * 2 - 1  # aprox. centrado en 0


def read_label(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        y = src.read(1).astype(np.int64)
    out = np.full_like(y, IGNORE)
    out[y == 0] = 0
    out[y == 1] = 1
    return out


def s2_cloud_fraction(path: str, blue_thresh: float = 0.2) -> float:
    """Heurística simple de nubes: fracción de píxeles con B2 (azul) muy brillante."""
    with rasterio.open(path) as src:
        blue = src.read(2).astype(np.float32) / S2_SCALE
    return float((blue > blue_thresh).mean())


# ----------------------------------------------------------------------------- #
# Dataset de PyTorch
# ----------------------------------------------------------------------------- #
class FloodDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool = False, crop: int | None = 256,
                 use_s2: bool = False):
        self.samples = samples
        self.train = train
        self.crop = crop
        self.use_s2 = use_s2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = read_s1(s.s1)
        if self.use_s2:
            x = np.concatenate([x, read_s2(s.s2)], axis=0)
        y = read_label(s.label)

        if self.train:
            x, y = self._augment(x, y)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "mask": torch.from_numpy(np.ascontiguousarray(y)),
            "weak": torch.tensor(float(s.weak)),
            "id": s.chip_id,
        }

    def _augment(self, x, y):
        # Recorte aleatorio (reduce memoria y actúa como aumento)
        if self.crop and self.crop < x.shape[-1]:
            H, W = y.shape
            for _ in range(10):  # intenta evitar recortes 100 % sin datos
                r, c = np.random.randint(0, H - self.crop + 1), np.random.randint(0, W - self.crop + 1)
                yc = y[r:r + self.crop, c:c + self.crop]
                if (yc != IGNORE).mean() > 0.1:
                    break
            x = x[:, r:r + self.crop, c:c + self.crop]
            y = yc
        # Volteos y rotaciones de 90° (el SAR no tiene orientación "arriba" privilegiada)
        if np.random.rand() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        if np.random.rand() < 0.5:
            x, y = x[:, ::-1, :], y[::-1, :]
        k = np.random.randint(4)
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return x, y
