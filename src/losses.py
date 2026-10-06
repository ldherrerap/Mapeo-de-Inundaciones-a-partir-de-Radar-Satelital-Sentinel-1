"""
Pérdidas para segmentación binaria con píxeles ignorados (etiqueta -1 -> 255).

  bce    : BCE con pos_weight (compensa que el agua es minoritaria)
  dice   : Dice suave (optimiza directamente el solapamiento)
  combo  : alpha * BCE + (1 - alpha) * Dice

weights (opcional, por muestra) permite bajar el peso de los chips con etiqueta débil.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import IGNORE


def _flatten(logits, target):
    logits = logits.squeeze(1)
    valid = target != IGNORE
    return logits, target.float().clamp(max=1), valid.float()


class SegLoss(nn.Module):
    def __init__(self, kind: str = "combo", pos_weight: float = 3.0, alpha: float = 0.5,
                 smooth: float = 1.0):
        super().__init__()
        assert kind in {"bce", "dice", "combo"}
        self.kind, self.alpha, self.smooth = kind, alpha, smooth
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def bce(self, logits, target, valid, w):
        l = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                               pos_weight=self.pos_weight)
        per_sample = (l * valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)
        return (per_sample * w).sum() / w.sum().clamp(min=1e-6)

    def dice(self, logits, target, valid, w):
        p = torch.sigmoid(logits) * valid
        t = target * valid
        inter = (p * t).sum((1, 2))
        denom = p.sum((1, 2)) + t.sum((1, 2))
        d = 1 - (2 * inter + self.smooth) / (denom + self.smooth)
        return (d * w).sum() / w.sum().clamp(min=1e-6)

    def forward(self, logits, target, weights=None):
        logits, target, valid = _flatten(logits, target)
        w = torch.ones(logits.shape[0], device=logits.device) if weights is None else weights
        if self.kind == "bce":
            return self.bce(logits, target, valid, w)
        if self.kind == "dice":
            return self.dice(logits, target, valid, w)
        return self.alpha * self.bce(logits, target, valid, w) + \
            (1 - self.alpha) * self.dice(logits, target, valid, w)
