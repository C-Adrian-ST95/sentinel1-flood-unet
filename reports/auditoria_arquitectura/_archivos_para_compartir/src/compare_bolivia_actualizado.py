"""Compara modelos U-Net ya entrenados contra test (90) y Bolivia (15).

Reconoce automáticamente ejecuciones gold M1/M2 y weak M3/M4.

Ejemplos (desde la raiz del proyecto):
    python -m src.compare_bolivia --root data/raw/sen1floods11 \
        --run outputs/experimentos_gold/run_20261029_013146_547224
    python -m src.compare_bolivia --root data/raw/sen1floods11 \
        --run outputs/experimentos_weak/run_20261029_035455_467350

No vuelve a entrenar ni modifica los checkpoints.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


GRUPOS = {
    'gold': ('M1_BCE', 'M2_BCE_Dice'),
    'weak': ('M3_Weak_BCE', 'M4_Weak_BCE_Dice'),
}


def detectar_modelos(run: Path, parser: argparse.ArgumentParser, grupo: str):
    if not run.is_dir():
        parser.error(f'No existe la carpeta de ejecucion: {run}')

    if grupo != 'auto':
        nombres = GRUPOS[grupo]
    else:
        grupos_completos = [nombres for nombres in GRUPOS.values()
                            if all((run / nombre / 'best.pt').is_file() for nombre in nombres)]
        if len(grupos_completos) == 1:
            nombres = grupos_completos[0]
        elif len(grupos_completos) > 1:
            parser.error('La carpeta contiene ambos grupos. Usa --grupo gold o --grupo weak.')
        else:
            encontrados = [str(p.relative_to(run)) for p in run.glob('*/best.pt')]
            parser.error('No se encontro un par completo de checkpoints M1/M2 o M3/M4.\n'
                         f'Checkpoints encontrados: {encontrados or "ninguno"}.\n'
                         'Comprueba que el entrenamiento terminara y exista best.pt en cada modelo.')

    for nombre in nombres:
        checkpoint = run / nombre / 'best.pt'
        if not checkpoint.is_file():
            parser.error(f'No se encuentra: {checkpoint}')
    return nombres


def main():
    p = argparse.ArgumentParser(description='Evaluacion geografica test vs Bolivia para M1/M2 o M3/M4')
    p.add_argument('--run', type=Path, required=True, help='Carpeta run_... con subcarpetas de modelos')
    p.add_argument('--root', type=Path, default=Path('data/raw/sen1floods11'))
    p.add_argument('--crop-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grupo', choices=['auto', 'gold', 'weak'], default='auto')
    args = p.parse_args()

    nombres = detectar_modelos(args.run, p, args.grupo)
    print('Modelos detectados:', ', '.join(nombres), flush=True)
    rows = []

    for nombre in nombres:
        checkpoint = args.run / nombre / 'best.pt'
        resultados = {}
        for split in ('test', 'bolivia'):
            out = args.run / nombre / split
            command = [
                sys.executable, '-m', 'src.test',
                '--root', str(args.root),
                '--checkpoint', str(checkpoint),
                '--split', split,
                '--limit', '0',
                '--crop-size', str(args.crop_size),
                '--batch-size', str(args.batch_size),
                '--output-dir', str(out),
            ]
            print(f'\nEVALUANDO {nombre} | {split.upper()}', flush=True)
            subprocess.run(command, check=True)
            metrics_path = out / f'metricas_{split}.json'
            resultados[split] = json.loads(metrics_path.read_text(encoding='utf-8'))

        test = resultados['test']
        bolivia = resultados['bolivia']
        rows.append({
            'modelo': nombre,
            'iou_test': test['iou_global'],
            'iou_bolivia': bolivia['iou_global'],
            'brecha_iou_test_menos_bolivia': test['iou_global'] - bolivia['iou_global'],
            'f1_test': test['f1_agua'],
            'f1_bolivia': bolivia['f1_agua'],
            'precision_bolivia': bolivia['precision_agua'],
            'recall_bolivia': bolivia['recall_agua'],
            'chips_test': test['chips'],
            'chips_bolivia': bolivia['chips'],
        })

    path = args.run / 'comparacion_geografica.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    width = 0.36
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in xs], [r['iou_test'] for r in rows], width,
           label='Test interno (90)')
    ax.bar([x + width / 2 for x in xs], [r['iou_bolivia'] for r in rows], width,
           label='Bolivia (15)')
    ax.set_xticks(xs, [r['modelo'] for r in rows])
    ax.set_ylabel('IoU global (agua)')
    ax.set_ylim(0, 1)
    ax.set_title('Generalizacion geografica: Test vs Bolivia')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    figure = args.run / 'comparacion_geografica.png'
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    print('\nRESULTADOS DE GENERALIZACION:')
    for row in rows:
        print(f"  {row['modelo']}: IoU test={row['iou_test']:.4f} | "
              f"IoU Bolivia={row['iou_bolivia']:.4f} | "
              f"Brecha test-Bolivia={row['brecha_iou_test_menos_bolivia']:+.4f}")
    print('Guardados:', path, 'y', figure)


if __name__ == '__main__':
    main()
