#!/usr/bin/env python3
"""
Descarga selectiva de Sen1Floods11 v1.1 (datos para la Pregunta 2).

Instalar:
    python -m pip install google-cloud-storage

Uso:
    python scripts/descargar_sen1floods11.py --dest data/raw/sen1floods11 --dry-run
    python scripts/descargar_sen1floods11.py --dest data/raw/sen1floods11 --workers 6
    python scripts/descargar_sen1floods11.py --dest data/raw/sen1floods11 --optical

No realiza entrenamiento. No carga los GeoTIFF completos en la memoria RAM.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

try:
    from google.cloud import storage
except ImportError:
    sys.exit("Falta una dependencia: instala con 'python -m pip install google-cloud-storage'.")

BUCKET = "sen1floods11"
VERSION = "v1.1"
DATA = "data/flood_events"

# Se descargan todas las imágenes SAR y las máscaras para la comparación
# doradas vs. débiles. S1OtsuLabelHand y S2IndexLabelWeak se incluyen
# para analizar la calidad de las etiquetas, no son datos extra de entrada.
GROUPS = {
    "gold": [
        f"{DATA}/HandLabeled/S1Hand/",
        f"{DATA}/HandLabeled/LabelHand/",
        f"{DATA}/HandLabeled/S1OtsuLabelHand/",
    ],
    "weak": [
        f"{DATA}/WeaklyLabeled/S1Weak/",
        f"{DATA}/WeaklyLabeled/S1OtsuLabelWeak/",
        f"{DATA}/WeaklyLabeled/S2IndexLabelWeak/",
    ],
    "split": ["splits/flood_handlabeled/"],
}
META_URL = (
    "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/"
    "master/Sen1Floods11_Metadata.geojson"
)


def parse_args():
    p = argparse.ArgumentParser(description="Sen1Floods11 v1.1: SAR VV/VH, máscaras y splits")
    p.add_argument("--dest", type=Path, default=Path("data/raw/sen1floods11"),
                   help="Carpeta destino dentro de tu proyecto (por defecto data/raw/sen1floods11)")
    p.add_argument("--workers", type=int, default=6, help="Descargas paralelas (recomendado 4-8)")
    p.add_argument("--mode", choices=("all", "gold", "weak", "split"), default="all",
                   help="all descarga lo obligatorio; gold y weak permiten hacerlo por etapas")
    p.add_argument("--optical", action="store_true",
                   help="Incluye S2Hand, opcional para el bono de fusión con Sentinel-2")
    p.add_argument("--dry-run", action="store_true", help="Solo lista archivos/tamaños, sin descargarlos")
    p.add_argument("--verify-md5", action="store_true",
                   help="Comprueba MD5 cuando el bucket lo proporciona; consume tiempo de disco")
    return p.parse_args()


def local_path(dest: Path, blob_name: str) -> Path:
    # Conservamos la estructura relativa a v1.1 dentro de data/raw/sen1floods11.
    prefix = f"{VERSION}/"
    if not blob_name.startswith(prefix):
        raise ValueError(f"Objeto fuera del prefijo esperado: {blob_name}")
    relative = blob_name[len(prefix):]
    if not relative or ".." in Path(relative).parts:
        raise ValueError(f"Ruta no segura: {blob_name}")
    return dest / Path(relative)


def md5_matches(path: Path, remote_md5: str | None) -> bool:
    if not remote_md5:
        return True
    digest = hashlib.md5()  # checksum de integridad del archivo; no se usa para seguridad
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest() == base64.b64decode(remote_md5)


def download_blob(blob, target: Path, verify_md5: bool) -> str:
    size = int(blob.size or 0)
    if target.is_file() and target.stat().st_size == size:
        if (not verify_md5) or md5_matches(target, blob.md5_hash):
            return "exists"

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".part")
    last_error = None
    for attempt in range(1, 4):
        try:
            # .part evita dejar un GeoTIFF aparentemente completo si se interrumpe.
            if temp.exists():
                temp.unlink()
            blob.download_to_filename(str(temp), timeout=180, checksum="auto")
            if temp.stat().st_size != size:
                raise IOError(f"Tamaño incorrecto: {temp.stat().st_size} vs {size}")
            if verify_md5 and not md5_matches(temp, blob.md5_hash):
                raise IOError("Checksum MD5 distinto del origen")
            os.replace(temp, target)
            return "downloaded"
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    if temp.exists():
        temp.unlink()
    raise RuntimeError(f"No se descargó {blob.name}: {last_error}")


def download_metadata(dest: Path) -> str:
    dst = dest / "Sen1Floods11_Metadata.geojson"
    if dst.is_file() and dst.stat().st_size > 100:
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with urlopen(META_URL, timeout=60) as resp, tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return "downloaded"


def check_pair(dest: Path, image_layer: str, mask_layer: str, folder: str) -> None:
    base = dest / "data" / "flood_events" / folder
    image_dir, mask_dir = base / image_layer, base / mask_layer
    if not image_dir.is_dir() or not mask_dir.is_dir():
        print(f"  No se revisan pares {image_layer}/{mask_layer}: no están ambas carpetas")
        return
    # Nombre: EVENT_CHIPID_LAYER.tif. Se conserva el identificador completo.
    img_ids = {p.name[:-len(f'_{image_layer}.tif')] for p in image_dir.glob(f'*_{image_layer}.tif')}
    lab_ids = {p.name[:-len(f'_{mask_layer}.tif')] for p in mask_dir.glob(f'*_{mask_layer}.tif')}
    print(f"  {folder}: {image_layer}={len(img_ids)} | {mask_layer}={len(lab_ids)} | "
          f"parejas={len(img_ids & lab_ids)} | sin máscara={len(img_ids-lab_ids)} | "
          f"sin imagen={len(lab_ids-img_ids)}")
    if img_ids != lab_ids:
        print("  ATENCIÓN: revisa los archivos faltantes; el emparejamiento no está completo.")


def main():
    a = parse_args()
    if not 1 <= a.workers <= 16:
        sys.exit("--workers debe estar entre 1 y 16")

    categories = ["gold", "weak", "split"] if a.mode == "all" else [a.mode]
    prefixes = [f"{VERSION}/{p}" for cat in categories for p in GROUPS[cat]]
    if a.optical and a.mode in ("all", "gold"):
        prefixes.append(f"{VERSION}/{DATA}/HandLabeled/S2Hand/")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)
    blobs = []
    print(f"Origen: gs://{BUCKET}/{VERSION}/")
    print("Consultando objetos del bucket público (todavía sin descargar)...", flush=True)
    try:
        for prefix in prefixes:
            found = [b for b in bucket.list_blobs(prefix=prefix)
                     if b.name.lower().endswith((".tif", ".tiff", ".csv"))]
            count_bytes = sum(int(b.size or 0) for b in found)
            print(f"  {prefix}: {len(found)} archivos; {count_bytes / (1024**3):.3f} GiB")
            if not found:
                print("  ADVERTENCIA: prefijo vacío; comprobar acceso o ruta.")
            blobs.extend(found)
    except Exception as exc:
        sys.exit(f"Error al listar bucket: {exc}. Verifica Internet y acceso público al bucket.")

    if not blobs or any(not any(b.name.startswith(p) for b in blobs) for p in prefixes):
        sys.exit("No se encontraron todos los grupos solicitados. No continuar con una descarga incompleta.")

    # Guardamos inventario local para que se pueda auditar la descarga.
    a.dest.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"name": b.name, "size": int(b.size or 0), "md5_base64": b.md5_hash}
        for b in blobs
    ]
    print(f"TOTAL: {len(blobs)} archivos; {sum(x['size'] for x in manifest) / (1024**3):.2f} GiB")
    if a.dry_run:
        print("Modo de prueba: no se ha descargado ningún archivo.")
        return

    manifest_path = a.dest / "manifest_download.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    downloaded = existing = failures = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {
            pool.submit(download_blob, b, local_path(a.dest, b.name), a.verify_md5): b.name
            for b in blobs
        }
        for index, fut in enumerate(as_completed(futures), 1):
            try:
                state = fut.result()
                downloaded += state == "downloaded"
                existing += state == "exists"
            except Exception as exc:
                failures += 1
                print(f"ERROR: {futures[fut]}: {exc}")
            if index % 100 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] nuevos={downloaded}; "
                      f"ya completos={existing}; errores={failures}", flush=True)

    if a.mode in ("all", "gold", "split"):
        try:
            print("Metadatos de eventos:", download_metadata(a.dest))
        except Exception as exc:
            failures += 1
            print(f"No se descargaron los metadatos: {exc}")

    print("\nVerificación de parejas por nombre:")
    if a.mode in ("all", "gold"):
        check_pair(a.dest, "S1Hand", "LabelHand", "HandLabeled")
    if a.mode in ("all", "weak"):
        check_pair(a.dest, "S1Weak", "S1OtsuLabelWeak", "WeaklyLabeled")
        check_pair(a.dest, "S1Weak", "S2IndexLabelWeak", "WeaklyLabeled")

    if failures:
        sys.exit(f"Finalizado con {failures} errores. Ejecuta el mismo comando otra vez para reintentar.")
    print("DESCARGA COMPLETA. Datos guardados en:", a.dest.resolve())
    print("Recuerda que los datos de Lima 2017 (Sentinel-1 + EMSN038) se obtienen por separado.")


if __name__ == "__main__":
    main()
