"""
Métricas para la clase "agua" ignorando píxeles sin datos.

Se acumulan TP/FP/FN/TN y se reportan:
  - IoU total (todos los píxeles del conjunto juntos, como en Bonafilia et al. 2020)
  - IoU medio por chip (promedio de IoU de cada chip; los chips sin agua ni predicción se omiten)
  - Precisión, recall, F1
  - Tasa de omisión  = FN / (TP + FN)   (agua real no detectada)
  - Tasa de comisión = FP / (TP + FP)   (agua predicha que no lo es)
  - Matriz de confusión 2x2
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset import IGNORE


class SegMeter:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0
        self.per_chip = []  # (id, iou, tp, fp, fn, water_frac)

    @torch.no_grad()
    def update(self, logits, target, ids=None):
        pred = (torch.sigmoid(logits.squeeze(1)) > self.threshold)
        valid = target != IGNORE
        t = target == 1
        tp = (pred & t & valid).sum((1, 2))
        fp = (pred & ~t & valid).sum((1, 2))
        fn = (~pred & t & valid).sum((1, 2))
        tn = (~pred & ~t & valid).sum((1, 2))
        self.tp += int(tp.sum()); self.fp += int(fp.sum())
        self.fn += int(fn.sum()); self.tn += int(tn.sum())
        n_valid = valid.sum((1, 2)).clamp(min=1)
        for i in range(pred.shape[0]):
            union = int(tp[i] + fp[i] + fn[i])
            iou = float(tp[i]) / union if union > 0 else float("nan")
            cid = ids[i] if ids is not None else str(len(self.per_chip))
            self.per_chip.append((cid, iou, int(tp[i]), int(fp[i]), int(fn[i]),
                                  float((t[i] & valid[i]).sum() / n_valid[i])))

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        chip_ious = [c[1] for c in self.per_chip if not np.isnan(c[1])]
        return {
            "iou_total": tp / (tp + fp + fn + eps),
            "iou_mean_chip": float(np.mean(chip_ious)) if chip_ious else float("nan"),
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + eps),
            "omission": fn / (tp + fn + eps),
            "commission": fp / (tp + fp + eps),
            "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "n_chips": len(self.per_chip),
        }
