"""Ejecuta la pregunta 2 desde los datos locales, con una sola orden.

Desde la raíz: python -m src.run_all --mode full
Para reutilizar checkpoints ya entrenados: python -m src.run_all --mode reuse
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
MODELS = {
    "gold": ("M1_BCE", "M2_BCE_Dice"),
    "weak": ("M3_Weak_BCE", "M4_Weak_BCE_Dice"),
}


def project_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (PROJECT / value).resolve()


def complete_run(run: Path, group: str) -> bool:
    return all(
        (run / name / "best.pt").is_file() and
        (group != "weak" or (run / name / "best.timing.json").is_file())
        for name in MODELS[group]
    )


def latest_run(parent: Path, group: str) -> Path:
    choices = sorted(
        (p for p in parent.glob("run_*") if p.is_dir() and complete_run(p, group)),
        key=lambda p: (p.stat().st_mtime, p.name), reverse=True,
    )
    if not choices:
        raise ValueError(f"No hay una ejecución {group} completa en {parent}")
    return choices[0]


def command(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(arg) for arg in args)]


def execute(label: str, argv: list[str], dry_run: bool) -> None:
    print(f"\n[{label}] {' '.join(argv)}", flush=True)
    if not dry_run:
        subprocess.run(argv, cwd=PROJECT, check=True)


def preflight(args: argparse.Namespace, data: Path, sar: Path, gdb: Path,
              reference: Path | None) -> None:
    required = [
        data / "data/flood_events/HandLabeled/S1Hand",
        data / "data/flood_events/HandLabeled/LabelHand",
        *(data / f"splits/flood_handlabeled/flood_{split}_data.csv"
          for split in ("train", "valid", "test", "bolivia")),
    ]
    if args.mode == "full":
        required.extend([
            data / "data/flood_events/WeaklyLabeled/S1Weak",
            data / "data/flood_events/WeaklyLabeled/S1OtsuLabelWeak",
        ])
    absent = [p for p in required if not p.exists()]
    if absent:
        raise ValueError("Faltan insumos Sen1Floods11:\n" + "\n".join(map(str, absent)))
    if not sar.is_file() or sar.stat().st_size == 0:
        raise ValueError(f"Falta el GeoTIFF SAR procesado de Lima: {sar}")
    if not gdb.is_dir():
        raise ValueError(f"Falta la geodatabase EMSN038: {gdb}")
    import rasterio
    with rasterio.open(sar) as scene:
        if scene.count != 2 or scene.crs is None:
            raise ValueError("El SAR de Lima requiere dos bandas VV/VH y CRS")
        if scene.width < args.chip_size or scene.height < args.chip_size:
            raise ValueError("El SAR es menor que el tamaño de chip elegido")
        if reference is not None:
            if not reference.is_file():
                raise ValueError(f"No existe la máscara de referencia: {reference}")
            with rasterio.open(reference) as mask:
                if (mask.count != 1 or mask.crs != scene.crs or
                    mask.width != scene.width or mask.height != scene.height or
                    not mask.transform.almost_equals(scene.transform)):
                    raise ValueError("La máscara de referencia no comparte la grilla SAR")
                if mask.nodata == 0:
                    raise ValueError("La máscara declara nodata=0, que significa no agua")
    if args.mode == "full":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("El entrenamiento completo requiere CUDA; usa --mode reuse para los pesos existentes")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(gold: Path, weak: Path, lima: Path, destination: Path,
                  reference: Path | None) -> None:
    rows = []
    for group, run in (("gold", gold), ("weak", weak)):
        for model in MODELS[group]:
            folder = run / model
            valid = read_json(folder / "evaluacion/metricas_valid.json")
            test = read_json(folder / "test/metricas_test.json")
            bolivia = read_json(folder / "bolivia/metricas_bolivia.json")
            lima_stats = read_json(lima / model / "metricas_lima.json")
            lima_iou = lima_stats["iou_global"]
            rows.append({
                "modelo": model,
                "checkpoint": str(folder / "best.pt"),
                "iou_valid": valid["iou_global"],
                "iou_test": test["iou_global"],
                "iou_media_chips_test": test["iou_media_chips"],
                "f1_test": test["f1_agua"],
                "precision_test": test["precision_agua"],
                "recall_test": test["recall_agua"],
                "tp_test": test["tp"], "fp_test": test["fp"],
                "fn_test": test["fn"], "tn_test": test["tn"],
                "iou_bolivia": bolivia["iou_global"],
                "iou_media_chips_bolivia": bolivia["iou_media_chips"],
                "brecha_test_menos_bolivia": (
                    test["iou_global"] - bolivia["iou_global"]
                    if test["iou_global"] is not None and bolivia["iou_global"] is not None
                    else None
                ),
                "iou_lima": lima_iou,
                "iou_media_chips_lima": lima_stats["iou_media_chips"],
                "f1_lima": lima_stats["f1_agua"],
                "brecha_test_menos_lima": (
                    test["iou_global"] - lima_iou
                    if test["iou_global"] is not None and lima_iou is not None else None
                ),
                "referencia_lima": lima_stats["referencia"],
            })
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "resumen_modelos.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "gold_run": str(gold), "weak_run": str(weak), "lima_output": str(lima),
        "referencia_lima": str(reference) if reference else None,
        "interpretacion_lima": (
            "Métricas de Lima condicionadas a la validez de la máscara suministrada; "
            "EMSN038 y SAR tienen diferencias temporales y semánticas."
            if reference else
            "Lima solo cualitativo: EMSN038 no aporta automáticamente una máscara "
            "binaria completa de agua/no agua para el 17/03/2017."
        ),
        "resultados": rows,
    }
    (destination / "resumen_ejecucion.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\nResumen final: {destination / 'resumen_modelos.csv'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("full", "reuse"), default="reuse",
                   help="full entrena M1-M4; reuse emplea checkpoints completos existentes")
    p.add_argument("--data-root", type=Path, default=Path("data/raw/sen1floods11"))
    p.add_argument("--sar", type=Path, default=Path("data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif"))
    p.add_argument("--gdb", type=Path, default=Path("data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb"))
    p.add_argument("--reference-mask", type=Path,
                   help="Máscara 0/1/-1 alineada y verificada; habilita IoU/F1 en Lima")
    p.add_argument("--gold-run", type=Path, help="Run M1/M2 existente; solo en mode reuse")
    p.add_argument("--weak-run", type=Path, help="Run M3/M4 existente; solo en mode reuse")
    p.add_argument("--gold-epochs", type=int, default=15)
    p.add_argument("--weak-epochs", type=int, default=12)
    p.add_argument("--gold-batch-size", type=int, default=4)
    p.add_argument("--weak-batch-size", type=int, default=8)
    p.add_argument("--chip-size", type=int, default=512)
    p.add_argument("--amp", action="store_true", help="AMP para M3/M4 al entrenar")
    p.add_argument("--dry-run", action="store_true", help="Validar insumos y mostrar etapas sin ejecutarlas")
    a = p.parse_args()
    if a.mode == "full" and (a.gold_run or a.weak_run):
        p.error("--gold-run y --weak-run solo se usan con --mode reuse")
    if min(a.gold_epochs, a.weak_epochs, a.gold_batch_size, a.weak_batch_size) < 1:
        p.error("Épocas y tamaños de lote deben ser positivos")
    if a.weak_batch_size < 2 or a.weak_batch_size % 2:
        p.error("--weak-batch-size debe ser par y >= 2")
    if a.chip_size != 512:
        p.error("El flujo completo usa chips originales de 512 px; src.test no admite ejemplos visuales recortados")

    data = project_path(a.data_root)
    sar = project_path(a.sar)
    gdb = project_path(a.gdb)
    reference = project_path(a.reference_mask) if a.reference_mask else None
    try:
        preflight(a, data, sar, gdb, reference)
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        if a.mode == "full":
            gold = PROJECT / "outputs/experimentos_gold" / timestamp
            weak = PROJECT / "outputs/experimentos_weak" / timestamp
        else:
            gold = project_path(a.gold_run) if a.gold_run else latest_run(
                PROJECT / "outputs/experimentos_gold", "gold"
            )
            weak = project_path(a.weak_run) if a.weak_run else latest_run(
                PROJECT / "outputs/experimentos_weak", "weak"
            )
            for path, group in ((gold, "gold"), (weak, "weak")):
                if not complete_run(path, group):
                    raise ValueError(f"Faltan checkpoints del grupo {group} en {path}")
        lima = PROJECT / "outputs/lima_2017" / timestamp
        summary = PROJECT / "outputs/pipeline" / timestamp
        print(f"Raíz: {PROJECT}\nModo: {a.mode}\nGold: {gold}\nWeak: {weak}")
        if a.mode == "full":
            execute("M1/M2", command("src.run_m1_m2", "--root", data,
                    "--run-dir", gold, "--epochs", a.gold_epochs,
                    "--batch-size", a.gold_batch_size, "--crop-size", a.chip_size), a.dry_run)
            execute("M3/M4", command("src.run_m3_m4", "--root", data,
                    "--run-dir", weak, "--epochs", a.weak_epochs,
                    "--batch-size", a.weak_batch_size, "--crop-size", a.chip_size,
                    *(["--amp"] if a.amp else [])), a.dry_run)
        if not a.dry_run:
            for path, group in ((gold, "gold"), (weak, "weak")):
                if not complete_run(path, group):
                    raise ValueError(f"No se completó {group}: {path}")
        for path, group in ((gold, "gold"), (weak, "weak")):
            for name in MODELS[group]:
                valid_path = path / name / "evaluacion/metricas_valid.json"
                if not valid_path.is_file():
                    execute(f"VALID {name}", command("src.test", "--root", data,
                            "--checkpoint", path / name / "best.pt", "--split", "valid",
                            "--crop-size", a.chip_size, "--output-dir", valid_path.parent), a.dry_run)
        for path, group in ((gold, "gold"), (weak, "weak")):
            execute(f"TEST/BOLIVIA {group}", command("src.compare_bolivia", "--run", path,
                    "--root", data, "--crop-size", a.chip_size, "--grupo", group), a.dry_run)
        execute("INSPECCIÓN EMSN038", command("src.lima.inspeccionar_emsn038",
                "--gdb", gdb, "--sar", sar, "--output", summary / "inspeccion_emsn038.json"), a.dry_run)
        lima_cmd = command("src.lima.evaluar", "--sar", sar, "--gold-run", gold,
                           "--weak-run", weak, "--output-dir", lima, "--chip-size", a.chip_size)
        if reference:
            lima_cmd += ["--reference-mask", str(reference)]
        execute("LIMA M1-M4", lima_cmd, a.dry_run)
        execute("DIAGNÓSTICO EMSN038", command("src.lima.diagnosticar_lima_emsn038",
                "--gdb", gdb, "--sar", sar, "--predicciones", lima,
                "--output-dir", summary / "diagnostico_emsn038"), a.dry_run)
        execute("MAPAS COMPARATIVOS", command("src.lima.visualizar_modelos_emsn038",
                "--gdb", gdb, "--sar", sar, "--predicciones", lima,
                "--output-dir", summary / "comparacion_visual"), a.dry_run)
        if not a.dry_run:
            write_summary(gold, weak, lima, summary, reference)
        else:
            print("\nSimulación completa: no se ejecutaron entrenamientos ni evaluaciones.")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"ERROR del flujo: {exc}\n")


if __name__ == "__main__":
    main()
