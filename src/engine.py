"""Utilidades compartidas: construir modelo, loaders, bucle de evaluación."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FloodDataset, hand_split, s2_cloud_fraction
from .metrics import SegMeter
from .unet import UNet


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def in_channels(cfg) -> int:
    from .dataset import S2_BANDS
    return 2 + (len(S2_BANDS) if cfg.get("use_s2", False) else 0)


def build_model(cfg) -> UNet:
    m = cfg["model"]
    return UNet(in_channels=in_channels(cfg), base_ch=m.get("base_ch", 32),
                depth=m.get("depth", 4), skip_mode=m.get("skip_mode", "concat"),
                dropout=m.get("dropout", 0.1))


def filter_cloudy(samples, max_cloud: float):
    """Para el bono SAR+óptico: se descartan chips con S2 nublado."""
    return [s for s in samples if s2_cloud_fraction(s.s2) <= max_cloud]


def eval_loader(cfg, split: str, num_workers: int = 2) -> DataLoader:
    use_s2 = cfg.get("use_s2", False)
    need_s2 = use_s2 or cfg.get("s2_subset", False)  # s2_subset: SAR-only sobre los mismos chips sin nubes
    samples = hand_split(cfg["data_root"], split, use_s2=need_s2)
    if need_s2:
        samples = filter_cloudy(samples, cfg.get("s2_max_cloud", 0.1))
    ds = FloodDataset(samples, train=False, crop=None, use_s2=use_s2)
    return DataLoader(ds, batch_size=4, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Devuelve lista de (id, logits[H,W], mask[H,W], image[C,H,W]) en CPU."""
    model.eval()
    out = []
    for b in loader:
        logits = model(b["image"].to(device)).float().cpu()
        for i, cid in enumerate(b["id"]):
            out.append((cid, logits[i], b["mask"][i], b["image"][i]))
    return out


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    meter = SegMeter(threshold)
    for b in loader:
        logits = model(b["image"].to(device)).float().cpu()
        meter.update(logits, b["mask"], b["id"])
    return meter


def tune_threshold(preds, grid=np.linspace(0.2, 0.8, 13)) -> float:
    """Elige el umbral que maximiza el IoU total en validación (nunca en test)."""
    best_t, best_iou = 0.5, -1
    for t in grid:
        meter = SegMeter(float(t))
        for cid, lg, m, _ in preds:
            meter.update(lg[None], m[None], [cid])
        iou = meter.compute()["iou_total"]
        if iou > best_iou:
            best_t, best_iou = float(t), iou
    return best_t


def save_ckpt(path, model, cfg, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, **(extra or {})}, path)


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck
