"""Entrena M3 y M4 consecutivos con gold+weak, evalúa VALID y compara.

Uso desde raíz (GPU Colab recomendado):
python -m src.run_m3_m4 --root data/raw/sen1floods11 \
  --epochs 12 --batch-size 8 --crop-size 512 --amp

Para continuar en Colab: añadir --run-dir RUTA_EXISTENTE --resume.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def run(cmd, title):
    print(f"\n{'=' * 60}\n{title}\n{' '.join(map(str, cmd))}\n{'=' * 60}", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"{title} terminó en {(time.perf_counter() - start) / 60:.2f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description="Experimentos M3/M4 Sen1Floods11")
    p.add_argument("--root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/experimentos_weak"))
    p.add_argument("--run-dir", type=Path, default=None, help="Carpeta existente si reanudas")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--weak-limit", type=int, default=0, help="0=usar todos los débiles admisibles")
    p.add_argument("--gold-limit", type=int, default=0, help="0=252 chips dorados de train")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Solo pruebas breves")
    p.add_argument("--no-eval", action="store_true")
    args = p.parse_args()
    if not args.root.exists():
        p.error(f"Ruta de datos no existente: {args.root}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error("No se detectó GPU: para prueba corta puedes añadir --allow-cpu")
    if args.resume and args.run_dir is None:
        p.error("Para reanudar indica --run-dir CARPETA --resume")

    run_dir = (args.run_dir if args.run_dir is not None else
               args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items() if k not in ("resume", "run_dir")}
    if args.resume:
        if config_path.exists():
            old = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("batch_size", "crop_size", "weak_limit", "gold_limit", "lr", "seed", "base"):
                if old.get(key) != config.get(key):
                    p.error(f"Al reanudar no cambies {key}: anterior={old.get(key)}, nuevo={config.get(key)}")
    else:
        if config_path.exists():
            p.error(f"Carpeta existente: {run_dir}. Usa --resume para continuar.")
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"RESULTADOS: {run_dir}", flush=True)

    models = (("M3_Weak_BCE", "bce"), ("M4_Weak_BCE_Dice", "bce_dice"))
    checkpoints = []
    for name, loss in models:
        out = run_dir / name
        out.mkdir(exist_ok=True)
        ckpt = out / "best.pt"
        done = out / "best.timing.json"
        if args.resume and done.exists() and ckpt.exists():
            print(f"{name}: ya terminó. Se conserva checkpoint.", flush=True)
        else:
            cmd = [sys.executable, "-m", "src.train_weak", "--root", str(args.root),
                   "--loss", loss, "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size), "--crop-size", str(args.crop_size),
                   "--weak-limit", str(args.weak_limit), "--gold-limit", str(args.gold_limit),
                   "--lr", str(args.lr), "--seed", str(args.seed), "--base", str(args.base),
                   "--num-workers", str(args.num_workers), "--patience", str(args.patience),
                   "--output", str(ckpt)]
            if args.amp:
                cmd.append("--amp")
            if args.resume:
                cmd.append("--resume")
            run(cmd, f"ENTRENANDO {name}")
        checkpoints.append((name, ckpt))

    rows = []
    for name, ckpt in checkpoints:
        meta = torch.load(ckpt, map_location="cpu", weights_only=True)
        stats = meta["validation_metrics"]
        timings = json.loads(ckpt.with_suffix(".timing.json").read_text(encoding="utf-8"))
        rows.append({
            "modelo": name, "mejor_epoca": meta["best_epoch"],
            "iou_global": stats["iou_global"],
            "iou_media_chips": stats["iou_media_chips"],
            "f1_agua": stats["f1_agua"],
            "precision_agua": stats["precision_agua"],
            "recall_agua": stats["recall_agua"],
            "tiempo_epocas_s": timings["tiempo_epocas_acumulado_s"],
        })
    with (run_dir / "comparacion_valid.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, ckpt in checkpoints:
        history = json.loads(ckpt.with_suffix(".history.json").read_text(encoding="utf-8"))
        plt.plot([e["epoch"] for e in history],
                 [e["valid"]["iou_global"] for e in history], marker="o", label=name)
    plt.title("M3 y M4: IoU validación (89 chips dorados)")
    plt.xlabel("Época")
    plt.ylabel("IoU global")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(run_dir / "comparacion_iou_valid.png", dpi=150)
    plt.close()

    if not args.no_eval:
        for name, ckpt in checkpoints:
            cmd = [sys.executable, "-m", "src.test", "--root", str(args.root),
                   "--checkpoint", str(ckpt), "--split", "valid", "--limit", "0",
                   "--crop-size", str(args.crop_size), "--output-dir", str(run_dir / name / "evaluacion")]
            run(cmd, f"EVALUANDO {name} EN VALID")
    print(f"\nFINALIZADO: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
