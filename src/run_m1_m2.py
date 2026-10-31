"""Entrena M1 (BCE) y M2 (BCE+Dice) en secuencia y evalua validacion.

Uso desde la raiz del proyecto:
    python -m src.run_m1_m2 --epochs 15 --batch-size 4
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


def ejecutar(cmd, etiqueta):
    print("\n" + "=" * 65, flush=True)
    print(etiqueta, flush=True)
    print(" ".join(str(x) for x in cmd), flush=True)
    print("=" * 65, flush=True)
    inicio = time.perf_counter()
    subprocess.run(cmd, check=True)
    duracion = time.perf_counter() - inicio
    print(f"{etiqueta} completado en {duracion:.2f} s ({duracion / 60:.2f} min)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Entrenar M1 y M2 sin mezclar test ni Bolivia")
    p.add_argument("--root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/experimentos_gold"))
    p.add_argument("--run-dir", type=Path, help="Carpeta exacta para esta ejecución")
    p.add_argument("--no-eval", action="store_true", help="No ejecutar test.py sobre valid")
    args = p.parse_args()

    if min(args.epochs, args.batch_size, args.crop_size, args.base) <= 0:
        p.error("epochs, batch-size, crop-size y base deben ser positivos")
    if not (args.root / "splits/flood_handlabeled/flood_train_data.csv").exists():
        p.error(f"No se encuentra el dataset: {args.root}")

    if not torch.cuda.is_available():
        p.error("CUDA no disponible: se cancela para no iniciar dos entrenamientos largos en CPU")

    carpeta = args.run_dir or args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    if carpeta.exists():
        p.error(f"La carpeta de ejecución ya existe: {carpeta}")
    carpeta.mkdir(parents=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"RESULTADOS: {carpeta}", flush=True)
    configuracion = {"root": str(args.root), "epochs": args.epochs,
                     "batch_size": args.batch_size, "crop_size": args.crop_size,
                     "lr": args.lr, "seed": args.seed, "base": args.base,
                     "train": "split oficial completo", "valid": "split oficial completo",
                     "test": "reservado", "bolivia": "reservado"}
    (carpeta / "config.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    inicio_experimento = time.perf_counter()
    modelos = [("M1_BCE", "bce"), ("M2_BCE_Dice", "bce_dice")]
    checkpoints = []
    for nombre, perdida in modelos:
        modelo_dir = carpeta / nombre
        modelo_dir.mkdir()
        ckpt = modelo_dir / "best.pt"
        ejecutar([
            sys.executable, "-m", "src.train", "--root", str(args.root),
            "--epochs", str(args.epochs), "--limit-train", "0", "--limit-valid", "0",
            "--crop-size", str(args.crop_size), "--batch-size", str(args.batch_size),
            "--lr", str(args.lr), "--seed", str(args.seed), "--base", str(args.base),
            "--loss", perdida, "--output", str(ckpt)
        ], f"ENTRENANDO {nombre}")
        checkpoints.append((nombre, ckpt))

    registros = []
    for nombre, ckpt in checkpoints:
        data = torch.load(ckpt, map_location="cpu", weights_only=True)
        m = data["validation_metrics"]
        registros.append({
            "modelo": nombre, "mejor_epoca": data["best_epoch"],
            "iou_global": m["iou_global"], "iou_media_chips": m["iou_media_chips"],
            "f1_agua": m["f1_agua"], "precision_agua": m["precision_agua"],
            "recall_agua": m["recall_agua"]
        })
    with (carpeta / "comparacion_valid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registros[0]))
        writer.writeheader()
        writer.writerows(registros)

    # IoU comparable entre ambos modelos, por epoca.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for nombre, ckpt in checkpoints:
        history_path = ckpt.with_suffix(".history.json")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        xs = [item["epoch"] for item in history]
        ys = [item["valid"]["iou_global"] for item in history]
        plt.plot(xs, ys, marker="o", markersize=3, label=nombre)
    plt.title("IoU global en validacion (sin test)")
    plt.xlabel("Epoca")
    plt.ylabel("IoU")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(carpeta / "comparacion_iou_valid.png", dpi=150)
    plt.close()

    if not args.no_eval:
        for nombre, ckpt in checkpoints:
            try:
                ejecutar([
                    sys.executable, "-m", "src.test", "--root", str(args.root),
                    "--checkpoint", str(ckpt), "--split", "valid",
                    "--limit", "0", "--crop-size", str(args.crop_size),
                    "--output-dir", str(carpeta / nombre / "evaluacion")
                ], f"EVALUANDO {nombre} EN VALID")
            except subprocess.CalledProcessError:
                print(f"Aviso: fallo la visualizacion de {nombre}; pesos y comparacion_valid.csv estan guardados", flush=True)

    duracion_experimento = time.perf_counter() - inicio_experimento
    print(
        f"\nFINALIZADO. Tiempo total M1+M2={duracion_experimento:.2f} s "
        f"({duracion_experimento / 60:.2f} min). Revisa: {carpeta}",
        flush=True,
    )


if __name__ == "__main__":
    main()
