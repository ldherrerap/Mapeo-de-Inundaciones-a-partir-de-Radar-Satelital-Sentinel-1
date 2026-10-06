"""
Entrenamiento de la U-Net en Sen1Floods11.

Uso:
    python -m src.train --config configs/e04_gold_weak_mixed.yaml
    python -m src.train --config configs/e01_gold_bce.yaml --epochs 2   # prueba rápida

Estrategias de supervisión (cfg.weak.mode):
    none      : solo los chips dorados de la partición train
    weak_only : solo los chips débiles (¿cuánto valen las etiquetas automáticas por sí solas?)
    mixed    : dorados + débiles en el mismo batch; la pérdida de los débiles se pondera por weak.weight
               y el muestreador garantiza una fracción gold_ratio de chips dorados por batch
    pretrain : pre-entrena con los débiles weak.pretrain_epochs épocas y luego ajusta solo con dorados

Al terminar se evalúa automáticamente en valid, test (en distribución) y bolivia (geográfico).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .dataset import FloodDataset, hand_split, weak_samples
from .engine import (build_model, eval_loader, evaluate, filter_cloudy, get_device, load_ckpt,
                     predict_logits, save_ckpt, set_seed, tune_threshold)
from .losses import SegLoss
from .unet import count_params


def load_cfg(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides.data_root:
        cfg["data_root"] = overrides.data_root
    if overrides.epochs is not None:
        cfg["train"]["epochs"] = overrides.epochs
        if cfg.get("weak", {}).get("mode") == "pretrain":
            cfg["weak"]["pretrain_epochs"] = max(1, overrides.epochs // 2)
    if overrides.out_dir:
        cfg["out_dir"] = overrides.out_dir
    return cfg


def make_train_loader(cfg, phase: str):
    """phase ∈ {gold, weak, mixed}"""
    t, w = cfg["train"], cfg.get("weak", {})
    use_s2 = cfg.get("use_s2", False)
    need_s2 = use_s2 or cfg.get("s2_subset", False)
    gold = hand_split(cfg["data_root"], "train", use_s2=need_s2)
    if need_s2:
        gold = filter_cloudy(gold, cfg.get("s2_max_cloud", 0.1))
    kw = dict(train=True, crop=t.get("crop", 256), use_s2=use_s2)

    if phase == "gold":
        ds = FloodDataset(gold, **kw)
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                          num_workers=t.get("num_workers", 2)), len(gold), 0

    weak = weak_samples(cfg["data_root"], source=w.get("source", "S1OtsuLabelWeak"),
                        max_chips=w.get("max_chips"), seed=cfg.get("seed", 42))
    if phase == "weak":
        ds = FloodDataset(weak, **kw)
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                          num_workers=t.get("num_workers", 2)), 0, len(weak)

    # mixed: muestreo ponderado para que ~gold_ratio de cada batch sea dorado
    ds = ConcatDataset([FloodDataset(gold, **kw), FloodDataset(weak, **kw)])
    r = w.get("gold_ratio", 0.5)
    sw = [r / len(gold)] * len(gold) + [(1 - r) / len(weak)] * len(weak)
    steps = w.get("steps_per_epoch", math.ceil(len(gold) / t["batch_size"]) * 2)
    sampler = WeightedRandomSampler(sw, num_samples=steps * t["batch_size"], replacement=True)
    return DataLoader(ds, batch_size=t["batch_size"], sampler=sampler, drop_last=True,
                      num_workers=t.get("num_workers", 2)), len(gold), len(weak)


def run_phase(cfg, model, loader, val_loader, device, epochs, run_dir, tag, history):
    t = cfg["train"]
    w_weak = cfg.get("weak", {}).get("weight", 1.0)
    crit = SegLoss(**cfg["loss"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=epochs * len(loader),
                                                pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, bad = -1.0, 0
    patience = t.get("patience", 15)

    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in loader:
            x, y = b["image"].to(device), b["mask"].to(device)
            weights = torch.where(b["weak"] > 0, w_weak, 1.0).to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = crit(logits.float(), y, weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)

        m = evaluate(model, val_loader, device).compute()
        row = {"phase": tag, "epoch": ep, "train_loss": tot / max(n, 1),
               "val_iou": m["iou_total"], "val_f1": m["f1"], "lr": sched.get_last_lr()[0],
               "secs": time.time() - t0}
        history.append(row)
        print(f"[{tag}] ep {ep:03d} loss={row['train_loss']:.4f} val_IoU={m['iou_total']:.4f} "
              f"F1={m['f1']:.4f} ({row['secs']:.0f}s)", flush=True)

        if m["iou_total"] > best:
            best, bad = m["iou_total"], 0
            save_ckpt(os.path.join(run_dir, f"best_{tag}.pt"), model, cfg, {"epoch": ep, "val_iou": best})
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stopping en época {ep}")
                break
    # recargar el mejor modelo de esta fase
    best_model, _ = load_ckpt(os.path.join(run_dir, f"best_{tag}.pt"), device)
    model.load_state_dict(best_model.state_dict())
    return best


def final_eval(cfg, model, device, run_dir):
    nw = cfg["train"].get("num_workers", 2)
    val_preds = predict_logits(model, eval_loader(cfg, "valid", nw), device)
    thr = tune_threshold(val_preds) if cfg.get("tune_threshold", True) else 0.5
    results = {"threshold": thr}
    for split in ["valid", "test", "bolivia"]:
        meter = evaluate(model, eval_loader(cfg, split, nw), device, threshold=thr)
        res = meter.compute()
        results[split] = res
        with open(os.path.join(run_dir, f"per_chip_{split}.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["chip_id", "iou", "tp", "fp", "fn", "water_frac"])
            wr.writerows(meter.per_chip)
        print(f"  {split:8s} IoU={res['iou_total']:.4f}  F1={res['f1']:.4f}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"omis={res['omission']:.3f}  comis={res['commission']:.3f}")
    gap = results["test"]["iou_total"] - results["bolivia"]["iou_total"]
    results["gap_test_minus_bolivia"] = gap
    print(f"  brecha IoU (test - Bolivia) = {gap:+.4f}")
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="sobrescribe train.epochs")
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    run_dir = os.path.join(cfg.get("out_dir", "runs"), cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model = build_model(cfg).to(device)
    print(f"== {cfg['name']} | device={device} | params={count_params(model)/1e6:.2f}M")
    val_loader = eval_loader(cfg, "valid", cfg["train"].get("num_workers", 2))
    history = []
    mode = cfg.get("weak", {}).get("mode", "none")

    if mode == "pretrain":
        loader, _, nw = make_train_loader(cfg, "weak")
        print(f"   pre-entrenamiento con {nw} chips débiles")
        run_phase(cfg, model, loader, val_loader, device, cfg["weak"]["pretrain_epochs"],
                  run_dir, "pretrain", history)
        loader, ng, _ = make_train_loader(cfg, "gold")
        print(f"   ajuste fino con {ng} chips dorados")
        run_phase(cfg, model, loader, val_loader, device, cfg["train"]["epochs"], run_dir,
                  "finetune", history)
    else:
        phase = {"mixed": "mixed", "weak_only": "weak"}.get(mode, "gold")
        loader, ng, nw = make_train_loader(cfg, phase)
        print(f"   entrenamiento '{phase}': {ng} dorados + {nw} débiles")
        run_phase(cfg, model, loader, val_loader, device, cfg["train"]["epochs"], run_dir,
                  "main", history)

    save_ckpt(os.path.join(run_dir, "best.pt"), model, cfg)
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        wr.writeheader(); wr.writerows(history)
    final_eval(cfg, model, device, run_dir)


if __name__ == "__main__":
    main()
