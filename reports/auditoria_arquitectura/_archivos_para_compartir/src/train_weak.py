"""Entrena M3 (BCE) o M4 (BCE + Dice) con gold+weak de Sen1Floods11.

Ejecutar desde la carpeta raíz:
python -m src.train_weak --root data/raw/sen1floods11 --loss bce --epochs 2 \
  --weak-limit 64 --batch-size 4 --output outputs/prueba_m3/best.pt
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import Sen1Floods11Dataset
from .weak_dataset import Sen1WeakDataset, BalancedGoldWeakBatchSampler
from .metrics import WaterMetrics, segmentation_loss
from .model import UNet


def json_dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class WeakTrainer:
    def __init__(self, model, device, loss_name="bce", lr=3e-4, amp=False):
        self.model = model.to(device)
        self.device = device
        self.loss_name = loss_name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.amp_requested = bool(amp)
        if self.amp_requested and device.type == "cuda":
            capability = torch.cuda.get_device_capability()
            # En GPUs anteriores a Ampere, FP16 puede producir NaN durante el
            # entrenamiento de esta U-Net aunque la pérdida se calcule en FP32.
            self.amp = capability >= (8, 0)
            if not self.amp:
                print(
                    f"Aviso: AMP solicitado, pero la GPU tiene capacidad "
                    f"{capability[0]}.{capability[1]}; se usará FP32 para evitar NaN.",
                    flush=True,
                )
        else:
            self.amp = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def run_epoch(self, loader, training):
        self.model.train(training)
        metricas = WaterMetrics() if not training else None
        total_loss = 0.0
        total_valid = 0
        for x, y, valid in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            valid = valid.to(self.device, non_blocking=True)
            count = int(valid.sum().item())
            if count == 0:
                continue
            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type="cuda", enabled=self.amp):
                    logits = self.model(x)
                    # Calcula la pérdida en FP32 aun con autocast para estabilidad.
                    loss = segmentation_loss(logits.float(), y, valid, self.loss_name)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            total_loss += float(loss.detach().item()) * count
            total_valid += count
            if metricas is not None:
                metricas.update(logits.detach(), y, valid)
        if total_valid == 0:
            raise RuntimeError("No existen píxeles válidos")
        return {"loss": total_loss / total_valid,
                **(metricas.summary() if metricas is not None else {})}

    def fit(self, train_loader, valid_loader, sampler, args, counts):
        best_path = args.output
        best_path.parent.mkdir(parents=True, exist_ok=True)
        last_path = best_path.with_name("last.pt")
        history_path = best_path.with_suffix(".history.json")
        timing_path = best_path.with_suffix(".timing.json")
        history = []
        best_iou, best_epoch, first_epoch = -1.0, None, 1
        if args.resume:
            if last_path.exists():
                data = torch.load(last_path, map_location="cpu", weights_only=True)
                if (data.get("loss_name") != args.loss or
                    data.get("counts") != counts or
                    data.get("base") != args.base):
                    raise ValueError("Configuración distinta a la guardada en last.pt")
                self.model.load_state_dict(data["state_dict"])
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
                if self.amp and data.get("scaler_state_dict"):
                    self.scaler.load_state_dict(data["scaler_state_dict"])
                first_epoch = data["epoch"] + 1
                best_iou, best_epoch = data["best_iou"], data["best_epoch"]
                history = json.loads(history_path.read_text(encoding="utf-8"))
                print(f"Reanudando desde época {first_epoch}; mejor IoU={best_iou:.4f}")
            else:
                print("No existe last.pt: se iniciará desde cero")
        start = time.perf_counter()
        epochs_done = 0
        for epoch in range(first_epoch, args.epochs + 1):
            sampler.set_epoch(epoch)
            t0 = time.perf_counter()
            tr = self.run_epoch(train_loader, True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_train = time.perf_counter() - t0
            t1 = time.perf_counter()
            val = self.run_epoch(valid_loader, False)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_valid = time.perf_counter() - t1
            iou = val["iou_global"]
            improved = iou is not None and iou > best_iou
            if improved:
                best_iou, best_epoch = iou, epoch
                torch.save({
                    "state_dict": self.model.state_dict(), "base": args.base,
                    "crop_size_train": args.crop_size, "normalizacion_db": [-50.0, 5.0],
                    "seed": args.seed, "loss_name": args.loss, "best_epoch": epoch,
                    "validation_metrics": val, "training_counts": counts,
                    "model_family": "gold_weak", "gold_fraction": 0.5,
                }, best_path)
            history.append({
                "epoch": epoch, "train": tr, "valid": val,
                "tiempo_train_s": round(t_train, 2),
                "tiempo_valid_s": round(t_valid, 2),
                "tiempo_epoca_s": round(t_train + t_valid, 2),
            })
            json_dump(history_path, history)
            torch.save({
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch, "best_iou": best_iou, "best_epoch": best_epoch,
                "loss_name": args.loss, "counts": counts, "base": args.base,
                "seed": args.seed, "crop_size": args.crop_size,
            }, last_path)
            epochs_done += 1
            print(f"Época {epoch}/{args.epochs} | Train={tr['loss']:.4f} | "
                  f"Valid={val['loss']:.4f} | IoU={iou:.4f} | "
                  f"F1={val['f1_agua']:.4f} | "
                  f"Train={t_train:.1f}s | Valid={t_valid:.1f}s "
                  f"{'[MEJOR]' if improved else ''}", flush=True)
            if args.patience > 0 and best_epoch is not None and epoch - best_epoch >= args.patience:
                print("Early stopping: sin mejora del IoU de validación", flush=True)
                break
        json_dump(timing_path, {
            "tiempo_sesion_s": round(time.perf_counter() - start, 2),
            "epocas_ejecutadas_en_sesion": epochs_done,
            "mejor_epoca": best_epoch, "mejor_iou_valid": best_iou,
            "tiempo_epocas_acumulado_s": round(sum(h["tiempo_epoca_s"] for h in history), 2),
            "cuentas": counts,
        })
        return history


def main():
    p = argparse.ArgumentParser(description="M3/M4: etiquetas doradas + débiles")
    p.add_argument("--root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--loss", choices=("bce", "bce_dice"), required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--weak-limit", type=int, default=0, help="0 = todos los débiles admitidos")
    p.add_argument("--gold-limit", type=int, default=0, help="0 = 252 del train oficial")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5, help="0 desactiva early stopping")
    p.add_argument("--amp", action="store_true", help="Mixed precision en CUDA")
    p.add_argument("--resume", action="store_true", help="Cargar last.pt para continuar")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        p.error("batch-size debe ser PAR y >= 2")
    if min(args.epochs, args.crop_size, args.base) <= 0 or args.weak_limit < 0 or args.gold_limit < 0:
        p.error("Parámetros inválidos")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Pérdida: {args.loss} | AMP: {args.amp}", flush=True)
    gold = Sen1Floods11Dataset(args.root, "train", None if args.gold_limit == 0 else args.gold_limit,
                               args.crop_size, args.seed)
    valid = Sen1Floods11Dataset(args.root, "valid", None, args.crop_size, args.seed)
    weak = Sen1WeakDataset(args.root, args.crop_size,
                           None if args.weak_limit == 0 else args.weak_limit,
                           args.seed, args.output.parent / "auditoria_weak.json")
    counts = {"gold": len(gold), "weak": len(weak), "valid_gold": len(valid)}
    sampler = BalancedGoldWeakBatchSampler(len(gold), len(weak), args.batch_size, args.seed)
    data = ConcatDataset([gold, weak])
    train_loader = DataLoader(data, batch_sampler=sampler,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = UNet(in_channels=2, out_channels=1, base=args.base)
    print(f"Chips: {counts} | Lotes por época: {len(train_loader)} | "
          f"Parámetros: {sum(p.numel() for p in model.parameters())}", flush=True)
    WeakTrainer(model, device, args.loss, args.lr, args.amp).fit(
        train_loader, val_loader, sampler, args, counts)


if __name__ == "__main__":
    main()
