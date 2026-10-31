"""Evaluacion y dos visualizaciones de U-Net en eventos distintos de Sen1Floods11.

Desde la raiz del proyecto:
python -m src.test --root data/raw/sen1floods11 --split valid --limit 0
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Guardar figuras sin interfaz grafica.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from .dataset import Sen1Floods11Dataset
from .metrics import WaterMetrics
from .model import UNet


def contraste(banda):
    """Ajusta contraste solo para mostrar la imagen (no cambia el modelo)."""
    b = np.ma.asarray(banda).astype(np.float32).filled(np.nan)
    valores = b[np.isfinite(b) & (b > -1000)]
    if valores.size == 0:
        return np.zeros(b.shape, dtype=np.float32)
    bajo, alto = np.percentile(valores, [2, 98])
    return np.clip((np.nan_to_num(b, nan=bajo) - bajo) / max(alto - bajo, 1e-6), 0, 1)


class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader):
        metricas = WaterMetrics()
        for x, y, valid in loader:
            logits = self.model(x.to(self.device))
            metricas.update(logits, y.to(self.device), valid.to(self.device))
        return metricas.summary()

    @torch.no_grad()
    def save_example(self, dataset, filename, index):
        x, y, valid = dataset[index]
        logits = self.model(x.unsqueeze(0).to(self.device))
        pred = torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5
        real = y[0].numpy() >= 0.5
        area = valid[0].numpy().astype(bool)
        image_path, _ = dataset.pairs[index]

        with rasterio.open(image_path) as src:
            vv, vh = src.read([1, 2], masked=True)
        if vv.shape != pred.shape:
            raise ValueError("Usa --crop-size 512 para comparar los chips originales completos.")

        # Busca RGB de Sentinel-2 en la misma ubicacion.
        s2_path = image_path.parent.parent / "S2Hand" / image_path.name.replace("_S1Hand.tif", "_S2Hand.tif")
        if s2_path.exists():
            with rasterio.open(s2_path) as src:
                rgb = src.read([4, 3, 2], masked=True)
            rgb_img = np.stack([contraste(b) for b in rgb], axis=-1)
            titulo_rgb = "Sentinel-2 RGB (optica)"
        else:
            rgb_img = None
            titulo_rgb = "Sentinel-2 no descargado"

        clases = ListedColormap(["#e5e7eb", "#0284c7", "#111827"])
        ref = np.where(area, real.astype(np.uint8), 2)
        salida = np.where(area, pred.astype(np.uint8), 2)
        errores = np.zeros(real.shape, dtype=np.uint8)
        errores[real & pred & area] = 1   # TP verde
        errores[~real & pred & area] = 2  # FP rojo
        errores[real & ~pred & area] = 3  # FN amarillo
        errores[~area] = 4               # NoData negro
        colores_error = ListedColormap(["#e5e7eb", "#16a34a", "#ef4444", "#facc15", "#111827"])

        tp = np.sum(real & pred & area)
        fp = np.sum(~real & pred & area)
        fn = np.sum(real & ~pred & area)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
        valor_iou = "n/d" if iou is None else f"{iou:.3f}"

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(contraste(vv), cmap="gray")
        axes[0, 0].set_title("Sentinel-1 VV (radar)")
        axes[0, 1].imshow(contraste(vh), cmap="gray")
        axes[0, 1].set_title("Sentinel-1 VH (radar)")
        if rgb_img is not None:
            axes[0, 2].imshow(rgb_img)
        else:
            axes[0, 2].text(0.5, 0.5, "Sin imagen S2", ha="center", va="center", transform=axes[0, 2].transAxes)
        axes[0, 2].set_title(titulo_rgb)
        axes[1, 0].imshow(ref, cmap=clases, vmin=0, vmax=2)
        axes[1, 0].set_title("Referencia: agua azul")
        axes[1, 1].imshow(salida, cmap=clases, vmin=0, vmax=2)
        axes[1, 1].set_title("Prediccion U-Net: agua azul")
        axes[1, 2].imshow(errores, cmap=colores_error, vmin=0, vmax=4)
        axes[1, 2].set_title("TP verde | FP rojo | FN amarillo")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"{image_path.stem}   |   IoU del chip: {valor_iou}", fontsize=12)
        fig.tight_layout()
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/unet_prueba_best.pt"))
    p.add_argument("--split", choices=["valid", "test", "bolivia"], default="valid")
    p.add_argument("--limit", type=int, default=0, help="0 = todos los chips del split")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    if args.split == "test":
        print("ATENCION: test es el conjunto reservado para evaluacion final.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = UNet(in_channels=2, out_channels=1, base=checkpoint["base"])
    model.load_state_dict(checkpoint["state_dict"])
    ds = Sen1Floods11Dataset(
        args.root, args.split, None if args.limit == 0 else args.limit,
        args.crop_size, checkpoint.get("seed", 42)
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    evaluator = Evaluator(model, device)
    stats = evaluator.evaluate(loader)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_file = args.output_dir / f"metricas_{args.split}.json"
    json_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # Elegir dos imagenes de paises distintos, no siempre dataset[0].
    indices = []
    paises = set()
    for i, (ruta, _) in enumerate(ds.pairs):
        pais = ruta.stem.split("_", 1)[0]
        if pais not in paises:
            paises.add(pais)
            indices.append(i)
        if len(indices) == 2:
            break
    if len(indices) < 2 and len(ds) >= 2:
        indices = [0, 1]
    if len(indices) < 2:
        raise RuntimeError("El split seleccionado no contiene al menos dos chips para generar ejemplos.")

    for i in indices:
        chip = ds.pairs[i][0].stem.replace("_S1Hand", "")
        png = args.output_dir / f"prediccion_{args.split}_{chip}.png"
        evaluator.save_example(ds, png, i)
        print("Figura:", png)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Metricas:", json_file)


if __name__ == "__main__":
    main()
