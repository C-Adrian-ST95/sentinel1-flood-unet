"""Sen1Floods11: pares de etiquetas débiles S1Otsu, con auditoría espacial.

NO utiliza regiones de evaluación (Bolivia, valid, test) para entrenar.
Misma normalización VV/VH que src.dataset.Sen1Floods11Dataset.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
import torch
from torch.utils.data import Dataset, Sampler

from .dataset import Sen1Floods11Dataset


def bounds_wgs84(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"GeoTIFF sin CRS: {path}")
        if src.crs.to_epsg() == 4326:
            return tuple(src.bounds)
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def overlap(a, b):
    """Solapamiento de área positivo de dos bounding boxes (sin incluir bordes)."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


class Sen1WeakDataset(Dataset):
    """Chips S1Weak + S1OtsuLabelWeak, excluyendo regiones retenidas."""

    def __init__(self, root, crop_size=512, limit=None, seed=42, audit_path=None):
        self.root = Path(root)
        self.crop_size = crop_size
        self.rng = random.Random(seed)
        weak_root = self.root / "data/flood_events/WeaklyLabeled"
        image_dir = weak_root / "S1Weak"
        mask_dir = weak_root / "S1OtsuLabelWeak"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError("Faltan S1Weak / S1OtsuLabelWeak. Descarga con --mode weak")

        # Recopilar regiones doradas reservadas: no usar en el entrenamiento.
        holdout = []
        for split in ("valid", "test", "bolivia"):
            ref = Sen1Floods11Dataset(self.root, split, limit=None, crop_size=crop_size, seed=seed)
            holdout += [(path.stem.split("_", 1)[0].casefold(), bounds_wgs84(path))
                        for path, _ in ref.pairs]

        images = sorted(image_dir.glob("*_S1Weak.tif"))
        if not images:
            raise FileNotFoundError(f"No hay chips S1Weak en {image_dir}")
        pairs, ignored = [], {"bolivia": 0, "solapamiento_holdout": 0}
        missing = []
        for image in images:
            prefix = image.name.removesuffix("_S1Weak.tif")
            mask = mask_dir / f"{prefix}_S1OtsuLabelWeak.tif"
            if not mask.is_file():
                missing.append(mask.name)
                continue
            country = prefix.split("_", 1)[0].casefold()
            if country == "bolivia":
                ignored["bolivia"] += 1
                continue
            bbox = bounds_wgs84(image)
            if any(overlap(bbox, ref_bbox) for ref_country, ref_bbox in holdout
                   if ref_country == country):
                ignored["solapamiento_holdout"] += 1
                continue
            pairs.append((image, mask))
        if missing:
            raise FileNotFoundError(
                f"Faltan {len(missing)} máscaras débiles. Primera: {missing[0]}"
            )
        if not pairs:
            raise ValueError("Tras excluir holdouts, no quedan chips débiles")
        self.rng.shuffle(pairs)
        allowed = len(pairs)
        self.pairs = pairs[:limit] if limit is not None else pairs
        summary = {
            "archivos_S1Weak": len(images), "pares_debiles_admisibles": allowed,
            "pares_usados": len(self.pairs), "exclusiones": ignored,
            "criterio": "Excluir Bolivia y bbox que intersecta valid/test/Bolivia",
            "fuente_etiqueta": "S1OtsuLabelWeak",
            "advertencia": "El filtrado espacial no garantiza independencia entre eventos cercanos",
        }
        if audit_path is not None:
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[weak] {len(self.pairs)} parejas seleccionadas de {len(images)} "
              f"(excluidas Bolivia={ignored['bolivia']}, overlap={ignored['solapamiento_holdout']})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        with rasterio.open(image_path) as sar, rasterio.open(mask_path) as reference:
            if sar.count != 2 or reference.count != 1:
                raise ValueError(f"Bandas inesperadas: {image_path.name}")
            if (sar.height, sar.width) != (reference.height, reference.width):
                raise ValueError(f"Dimensiones distintas: {image_path.name}")
            if sar.crs != reference.crs or not sar.transform.almost_equals(reference.transform):
                raise ValueError(f"Grillas distintas: {image_path.name}")
            size = self.crop_size
            if size is None:
                window = None
            else:
                if size <= 0 or size > min(sar.height, sar.width):
                    raise ValueError(f"crop-size={size} no cabe en {image_path.name}")
                row = random.randint(0, sar.height - size)
                col = random.randint(0, sar.width - size)
                window = Window(col, row, size, size)
            image = sar.read([1, 2], window=window, masked=True)
            mask = reference.read(1, window=window, masked=True)
        x = image.astype(np.float32).filled(np.nan)
        y = mask.filled(-1).astype(np.int16)
        valid = np.isin(y, [0, 1]) & np.isfinite(x).all(axis=0)
        valid &= (x > -1000).all(axis=0)
        x = np.nan_to_num(x, nan=-50.0, neginf=-50.0, posinf=5.0)
        x = (np.clip(x, -50.0, 5.0) + 50.0) / 55.0
        x[:, ~valid] = 0.0
        y = np.where(valid, y, 0).astype(np.float32)[None]
        valid = valid.astype(np.float32)[None]
        return (torch.from_numpy(x.copy()),
                torch.from_numpy(y.copy()),
                torch.from_numpy(valid.copy()))


class BalancedGoldWeakBatchSampler(Sampler):
    """Una pasada por TODAS las imágenes débiles en cada época.

    Cada lote combina aprox. 50% doradas (con reemplazo) y 50% débiles.
    Por ello cada muestra débil se usa exactamente una vez por época;
    las doradas se repiten para no quedar ignoradas por el gran desbalance.
    """

    def __init__(self, gold_count, weak_count, batch_size=8, seed=42):
        if gold_count < 1 or weak_count < 1:
            raise ValueError("Se requieren muestras doradas y débiles")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch-size debe ser PAR y >= 2")
        self.gold_count = gold_count
        self.weak_count = weak_count
        self.weak_per_batch = batch_size // 2
        self.seed = seed
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.current_epoch)
        weak_indices = list(range(self.gold_count, self.gold_count + self.weak_count))
        rng.shuffle(weak_indices)
        for pos in range(0, len(weak_indices), self.weak_per_batch):
            weak_batch = weak_indices[pos:pos + self.weak_per_batch]
            gold_batch = [rng.randrange(self.gold_count) for _ in weak_batch]
            batch = gold_batch + weak_batch
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(self.weak_count / self.weak_per_batch)
