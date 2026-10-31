"""Entrenamiento corto de una U-Net sobre etiquetas doradas de Sen1Floods11.

Desde la carpeta raíz del proyecto:
python -m src.train --root raw/sen1floods11 --epochs 3
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Sen1Floods11Dataset
from .metrics import WaterMetrics, segmentation_loss
from .model import UNet


def number(value):
    return "n/d" if value is None else f"{value:.4f}"


class Trainer:
    def __init__(self, model, device, lr=1e-3, loss_name="bce"):
        self.model = model.to(device)
        self.device = device
        self.loss_name = loss_name
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
                loss = segmentation_loss(logits, y, valid, self.loss_name)
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
        entrenamiento_inicio = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoca_inicio = time.perf_counter()
            tr = self.epoch(train_loader, training=True)
            va = self.epoch(valid_loader, training=False)
            duracion_epoca = time.perf_counter() - epoca_inicio
            history.append({"epoch": epoch, "train": tr, "valid": va, "duration_seconds": duracion_epoca})
            print(
                f"Época {epoch}/{epochs} | "
                f"Train loss={tr['loss']:.4f} | Valid loss={va['loss']:.4f} | "
                f"IoU={number(va['iou_global'])} | F1={number(va['f1_agua'])} | "
                f"Tiempo={duracion_epoca:.2f} s",
                flush=True,
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
                    "loss_name": self.loss_name,
                    "best_epoch": epoch,
                    "validation_metrics": va,
                }, output)
                print(f"  Mejor modelo guardado: {output}")
            # Persistir historial tras cada epoca; sobrevive a interrupciones.
            with output.with_suffix(".history.json").open("w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2, allow_nan=False)
        duracion_total = time.perf_counter() - entrenamiento_inicio
        print(
            f"Entrenamiento completado | Tiempo total={duracion_total:.2f} s "
            f"({duracion_total / 60:.2f} min)",
            flush=True,
        )
        return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("raw/sen1floods11"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--loss", choices=("bce", "bce_dice"), default="bce")
    p.add_argument("--limit-train", type=int, default=40)
    p.add_argument("--limit-valid", type=int, default=10)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs/unet_prueba_best.pt"))
    args = p.parse_args()
    if min(args.epochs, args.crop_size, args.batch_size, args.base) <= 0:
        p.error("Epocas, crop, batch y base deben ser positivos")
    if args.limit_train < 0 or args.limit_valid < 0:
        p.error("Los limites de train y valid deben ser >= 0 (0 = todos)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    tr = Sen1Floods11Dataset(args.root, "train", None if args.limit_train == 0 else args.limit_train, args.crop_size, args.seed)
    va = Sen1Floods11Dataset(args.root, "valid", None if args.limit_valid == 0 else args.limit_valid, args.crop_size, args.seed)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet(in_channels=2, out_channels=1, base=args.base)
    print("Parámetros:", sum(p.numel() for p in model.parameters()))
    print("Funcion de perdida:", args.loss)
    Trainer(model, device, args.lr, args.loss).fit(
        train_loader, valid_loader, args.epochs, args.output,
        args.base, args.crop_size, args.seed
    )


if __name__ == "__main__":
    main()
