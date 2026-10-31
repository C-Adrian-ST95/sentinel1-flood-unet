"""Evaluar checkpoints M1/M2 ya entrenados sobre test y Bolivia, sin reentrenar.

Ejemplo:
python -m src.compare_bolivia --run outputs/experimentos_gold/run_20261029_005856_233967
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    rows = []
    for nombre in ("M1_BCE", "M2_BCE_Dice"):
        checkpoint = args.run / nombre / "best.pt"
        if not checkpoint.is_file():
            p.error(f"No se encuentra: {checkpoint}")
        resultados = {}
        for split in ("test", "bolivia"):
            out = args.run / nombre / split
            command = [
                sys.executable, "-m", "src.test", "--root", str(args.root),
                "--checkpoint", str(checkpoint), "--split", split,
                "--limit", "0", "--crop-size", str(args.crop_size),
                "--batch-size", str(args.batch_size), "--output-dir", str(out),
            ]
            print(f"\nEVALUANDO {nombre} | {split.upper()}", flush=True)
            subprocess.run(command, check=True)
            resultados[split] = json.loads((out / f"metricas_{split}.json").read_text(encoding="utf-8"))

        test = resultados["test"]
        bolivia = resultados["bolivia"]
        rows.append({
            "modelo": nombre,
            "iou_test": test["iou_global"],
            "iou_bolivia": bolivia["iou_global"],
            "brecha_iou_test_menos_bolivia": test["iou_global"] - bolivia["iou_global"],
            "f1_test": test["f1_agua"],
            "f1_bolivia": bolivia["f1_agua"],
            "precision_bolivia": bolivia["precision_agua"],
            "recall_bolivia": bolivia["recall_agua"],
            "chips_test": test["chips"],
            "chips_bolivia": bolivia["chips"],
        })

    path = args.run / "comparacion_geografica.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    xs = range(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in xs], [r["iou_test"] for r in rows], width, label="Test interno (90)")
    ax.bar([x + width / 2 for x in xs], [r["iou_bolivia"] for r in rows], width, label="Bolivia (15)")
    ax.set_xticks(list(xs), [r["modelo"] for r in rows])
    ax.set_ylabel("IoU global (agua)")
    ax.set_ylim(0, 1)
    ax.set_title("Generalización geográfica: Test vs Bolivia")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figure = args.run / "comparacion_geografica.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    print("\nRESULTADOS DE GENERALIZACIÓN:")
    for r in rows:
        print(f"  {r['modelo']}: IoU test={r['iou_test']:.4f} | "
              f"IoU Bolivia={r['iou_bolivia']:.4f} | "
              f"Brecha test-Bolivia={r['brecha_iou_test_menos_bolivia']:+.4f}")
    print("Guardados:", path, "y", figure)


if __name__ == "__main__":
    main()
