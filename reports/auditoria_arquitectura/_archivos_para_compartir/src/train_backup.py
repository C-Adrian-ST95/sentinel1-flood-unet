"""Entrenamiento corto de una U-Net sobre etiquetas doradas de Sen1Floods11.

Desde la carpeta raíz del proyecto:
python -m src.train --root raw/sen1floods11 --epochs 3
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import Sen1Floods11Dataset
from src.metrics import WaterMetrics, masked_bce
from src.model import UNet


def number(value):
    return "n/d" if value is None else f"{value:.4f}"


class Trainer:
    def __init__(self, model, device, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def epoch(self, loader, training):
        self.model.train(training)
        stats = WaterMetrics()
        weighted_loss = 0.0
        valid_total = 0
        for x, y, valid in loader:
            x, y, valid = (t.to(self.device) for t in (x, y, valid))
            count = int(valid.sum().item())
            if count == 0:
                continue
            with torch.set_grad_enabled(training):
                logits = self.model(x)
                loss = masked_bce(logits, y, valid)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
            weighted_loss += loss.item() * count
            valid_total += count
            stats.update(logits.detach(), y, valid)
        if not valid_total:
            raise RuntimeError("No se encontraron píxeles válidos para entrenar/evaluar")
        return {"loss": weighted_loss / valid_total, **stats.summary()}

    def fit(self, train_loader, valid_loader, epochs, output, base, crop_size, seed):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        history = []
        best_iou = -1.0
        for epoch in range(1, epochs + 1):
            tr = self.epoch(train_loader, training=True)
            va = self.epoch(valid_loader, training=False)
            history.append({"epoch": epoch, "train": tr, "valid": va})
            print(
                f"Época {epoch}/{epochs} | "
                f"Train BCE={tr['loss']:.4f} | Valid BCE={va['loss']:.4f} | "
                f"IoU={number(va['iou_global'])} | F1={number(va['f1_agua'])}"
            )
            iou = va["iou_global"]
            if epoch == 1 or (iou is not None and iou > best_iou):
                best_iou = -1.0 if iou is None else iou
                torch.save({
                    "state_dict": self.model.state_dict(),
                    "base": base,
                    "crop_size_train": crop_size,
                    "normalizacion_db": [-50.0, 5.0],
                    "seed": seed,
                    "best_epoch": epoch,
                    "validation_metrics": va,
                }, output)
                print(f"  Mejor modelo guardado: {output}")
        with output.with_suffix(".history.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2, allow_nan=False)
        return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("raw/sen1floods11"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit-train", type=int, default=40)
    p.add_argument("--limit-valid", type=int, default=10)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs/unet_prueba_best.pt"))
    args = p.parse_args()
    if min(args.epochs, args.limit_train, args.limit_valid, args.crop_size, args.batch_size, args.base) <= 0:
        p.error("Épocas, límites, crop, batch y base deben ser positivos")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    tr = Sen1Floods11Dataset(args.root, "train", args.limit_train, args.crop_size, args.seed)
    va = Sen1Floods11Dataset(args.root, "valid", args.limit_valid, args.crop_size, args.seed)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet(in_channels=2, out_channels=1, base=args.base)
    print("Parámetros:", sum(p.numel() for p in model.parameters()))
    Trainer(model, device, args.lr).fit(
        train_loader, valid_loader, args.epochs, args.output,
        args.base, args.crop_size, args.seed
    )


if __name__ == "__main__":
    main()
