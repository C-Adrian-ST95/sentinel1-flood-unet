"""Carga local de Sen1Floods11 v1.1 usando los splits oficiales.

Solo chips dorados para la primera prueba; -1 (NoData) se ignora.
"""
import csv
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset


class Sen1Floods11Dataset(Dataset):
    """Pares VV/VH y máscara manual; lectura por ventana, sin cargar todo a RAM."""

    def __init__(self, root, split="train", limit=None, crop_size=256, seed=42):
        self.root = Path(root)
        if split not in {"train", "valid", "test", "bolivia"}:
            raise ValueError("split debe ser train, valid, test o bolivia")
        self.split = split
        self.crop_size = crop_size
        self.random_crop = split == "train"
        self.random = random.Random(seed)

        base = self.root / "data/flood_events/HandLabeled"
        img_dir = base / "S1Hand"
        mask_dir = base / "LabelHand"
        csv_path = self.root / "splits/flood_handlabeled" / f"flood_{split}_data.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"No existe el CSV del split: {csv_path}")
        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Falta S1Hand o LabelHand dentro de {base}")

        # CSV oficial: ruta a imagen SAR, ruta a máscara; las rutas pueden ser largas.
        pairs = []
        missing = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                img_name = row[0].strip().replace("\\", "/").split("/")[-1]
                mask_name = row[1].strip().replace("\\", "/").split("/")[-1]
                if not (img_name.endswith("_S1Hand.tif") and mask_name.endswith("_LabelHand.tif")):
                    # Permite una fila de cabecera, pero no mezcla capas incompatibles.
                    if "image" in img_name.lower() or "sar" in img_name.lower():
                        continue
                    raise ValueError(f"Columnas inesperadas en {csv_path}: {row[:2]}")
                if img_name.removesuffix("_S1Hand.tif") != mask_name.removesuffix("_LabelHand.tif"):
                    raise ValueError(f"Chip desparejado: {img_name} / {mask_name}")
                image = img_dir / img_name
                mask = mask_dir / mask_name
                if image.is_file() and mask.is_file():
                    pairs.append((image, mask))
                else:
                    missing.append((str(image), str(mask)))
        if missing:
            raise FileNotFoundError(
                f"Faltan {len(missing)} parejas del CSV {split}. Primera pareja: {missing[0]}"
            )
        if not pairs:
            raise ValueError(f"El CSV {csv_path} no tiene pares utilizables")

        self.random.shuffle(pairs)
        self.pairs = pairs if limit is None else pairs[:limit]
        if not self.pairs:
            raise ValueError("El límite debe ser mayor que cero")
        print(f"[{split}] {len(self.pairs)} imágenes seleccionadas de {len(pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with rasterio.open(image_path) as sar, rasterio.open(mask_path) as reference:
            if sar.count != 2 or reference.count != 1:
                raise ValueError(f"Bandas inesperadas en {image_path.name}")
            if (sar.width, sar.height) != (reference.width, reference.height):
                raise ValueError(f"Dimensiones distintas en {image_path.name}")
            # También verificamos la georreferenciación para no comparar grillas distintas.
            if sar.crs != reference.crs or not sar.transform.almost_equals(reference.transform):
                raise ValueError(f"Grillas geográficas distintas en {image_path.name}")

            size = self.crop_size
            if size is None:
                window = None  # Chip entero, normalmente 512 x 512.
            else:
                if size > min(sar.height, sar.width) or size <= 0:
                    raise ValueError(f"crop_size={size} no cabe en {image_path.name}")
                if self.random_crop:
                    r = random.randint(0, sar.height - size)
                    c = random.randint(0, sar.width - size)
                else:
                    r = (sar.height - size) // 2
                    c = (sar.width - size) // 2
                window = Window(c, r, size, size)

            image = sar.read([1, 2], window=window, masked=True)
            mask = reference.read(1, window=window, masked=True)

        x = image.astype(np.float32).filled(np.nan)
        y = mask.filled(-1).astype(np.int16)
        valid = np.isin(y, [0, 1]) & np.isfinite(x).all(axis=0)
        valid &= (x > -1000).all(axis=0)  # Descarta valores sentinel extremos.

        # Normalización inicial fija para VV y VH en dB; documentar su rango.
        x = np.nan_to_num(x, nan=-50.0, neginf=-50.0, posinf=5.0)
        x = (np.clip(x, -50.0, 5.0) + 50.0) / 55.0
        x[:, ~valid] = 0.0
        y = np.where(valid, y, 0).astype(np.float32)[None]
        valid = valid.astype(np.float32)[None]

        return (torch.from_numpy(x.copy()),
                torch.from_numpy(y.copy()),
                torch.from_numpy(valid.copy()))
