"""Diagnóstico cartográfico EMSN038 vs predicciones U-Net de Lima 2017.

NO calcula IoU/F1: Potential_Maximum_flood es extensión POTENCIAL afectada,
no una etiqueta de agua observada el 17/03/2017. Footprints no prueba que
toda el área no marcada haya sido comprobada como no inundada.

Ejecutar desde la raíz del proyecto:
python -m src.lima.diagnosticar_lima_emsn038 --gdb ... --sar ... --predicciones ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

MODEL_NAMES = ('M1_BCE', 'M2_BCE_Dice', 'M3_Weak_BCE', 'M4_Weak_BCE_Dice')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gdb', type=Path, required=True)
    p.add_argument('--sar', type=Path, required=True)
    p.add_argument('--predicciones', type=Path,
                   default=Path('outputs/lima_2017/evaluacion_20261030'),
                   help='Carpeta con M1_BCE/mascara_predicha_lima.tif, etc.')
    p.add_argument('--output-dir', type=Path,
                   default=Path('reports/lima_2017/diagnostico_emsn038'))
    a = p.parse_args()
    for path in (a.gdb, a.sar):
        if not path.exists():
            p.error(f'No existe: {path}')
    return a


def read_layer(gdb, layer, crs):
    try:
        gdf = gpd.read_file(gdb, layer=layer, engine='pyogrio')
    except (ImportError, TypeError):
        gdf = gpd.read_file(gdb, layer=layer)
    if gdf.crs is None:
        raise ValueError(f'La capa {layer} no tiene CRS')
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    return gdf.to_crs(crs)


def make_mask(gdf, ref, dtype='uint8'):
    return rasterize(
        ((geom, 1) for geom in gdf.geometry), out_shape=(ref.height, ref.width),
        transform=ref.transform, fill=0, dtype=dtype, all_touched=False,
    ).astype(bool)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(a.sar) as ref:
        if ref.count != 2:
            raise ValueError('Se requieren dos bandas SAR VV/VH')
        raw = ref.read((1, 2), masked=True)
        sar_valid = (~np.ma.getmaskarray(raw).any(axis=0)) & (np.isfinite(raw.filled(np.nan)).all(axis=0))
        vv = raw[0].filled(np.nan)
        potential = read_layer(a.gdb, 'Potential_Maximum_flood', ref.crs)
        footprints = read_layer(a.gdb, 'Footprints', ref.crs)
        sources = potential['source'].fillna('(sin valor)').value_counts().to_dict() if 'source' in potential else {}
        potential_mask = make_mask(potential, ref)

        # Los polígonos con source='river' pueden representar el cauce habitual,
        # que NO debe confundirse automáticamente con terreno inundado.
        groups = {}
        if 'source' in potential:
            for source, subset in potential.groupby('source', dropna=False):
                groups[str(source)] = make_mask(subset, ref)

        dates = pd.to_datetime(footprints['SDV'], errors='coerce', utc=True) if 'SDV' in footprints else pd.Series(pd.NaT, index=footprints.index)
        post_event = footprints.loc[(dates.dt.date >= pd.Timestamp('2017-03-18').date()) & (dates.dt.date <= pd.Timestamp('2017-03-21').date())].copy()
        post_mask = make_mask(post_event, ref) if len(post_event) else np.zeros_like(sar_valid, dtype=bool)
        covered_valid = post_mask & sar_valid
        potential_valid = potential_mask & sar_valid
        potential_post = potential_valid & post_mask
        background_post = covered_valid & (~potential_mask)

        summary = {
            'escena_sar': str(a.sar), 'crs': str(ref.crs),
            'ancho': ref.width, 'alto': ref.height,
            'pixeles_sar_validos': int(sar_valid.sum()),
            'layer_potencial': 'Potential_Maximum_flood',
            'n_poligonos_potenciales': len(potential),
            'source_potencial': {str(k): int(v) for k, v in sources.items()},
            'pixeles_potencial_sobre_sar_valido': int(potential_valid.sum()),
            'footprints_todos': len(footprints),
            'footprints_18_21_marzo_2017': len(post_event),
            'fechas_footprints': [str(x) for x in dates.dt.date.dropna().unique()],
            'pixeles_sar_validos_bajo_footprints_postevento': int(covered_valid.sum()),
            'pixeles_potencial_bajo_footprints_postevento': int(potential_post.sum()),
            'pixeles_en_footprints_fuera_potencial_sin_etiqueta_negativa_verificada': int(background_post.sum()),
            'aclaracion': (
                'NO es evaluación IoU/F1. Potential_Maximum_flood no equivale a agua observada; '
                'los footprints ópticos no demuestran por sí mismos una clasificación negativa '
                'exhaustiva. Fuera del polígono potencial = desconocido para la clase agua.'
            ),
        }
        for source, mask in groups.items():
            summary['pixeles_fuente_' + source] = int((mask & sar_valid).sum())

        rows = []
        pred_preview = None
        for name in MODEL_NAMES:
            path = a.predicciones / name / 'mascara_predicha_lima.tif'
            if not path.exists():
                print(f'AVISO: no existe predicción para {name}: {path}')
                continue
            with rasterio.open(path) as pred_ds:
                if (pred_ds.crs != ref.crs or pred_ds.width != ref.width or
                        pred_ds.height != ref.height or
                        not pred_ds.transform.almost_equals(ref.transform)):
                    raise ValueError(f'Predicción {name} no coincide con la cuadrícula SAR')
                mask = pred_ds.read(1)
                unique = set(np.unique(mask).tolist())
                if not unique <= {0, 1, 255}:
                    raise ValueError(f'Predicción {name}: valores inesperados {sorted(unique)}')
                predicted_water = (mask == 1) & sar_valid
            intersect_potential = predicted_water & potential_valid
            # Solapamiento espacial descriptivo, no TP ni recall.
            row = {
                'modelo': name,
                'pixeles_agua_predicha_total': int(predicted_water.sum()),
                'pixeles_agua_predicha_en_potencial': int(intersect_potential.sum()),
                'fraccion_predicha_en_zona_potencial': (
                    float(intersect_potential.sum()/predicted_water.sum()) if predicted_water.any() else None
                ),
                'fraccion_zona_potencial_marcada_como_agua': (
                    float(intersect_potential.sum()/potential_valid.sum()) if potential_valid.any() else None
                ),
                'pixeles_agua_predicha_en_footprints_postevento': int((predicted_water & post_mask).sum()),
                'nota': 'SOLAPAMIENTO ESPACIAL DESCRIPTIVO; no IoU, precision, recall ni F1',
            }
            for source, smask in groups.items():
                row['predichos_dentro_de_fuente_' + source] = int((predicted_water & smask).sum())
            rows.append(row)
            if pred_preview is None:
                pred_preview = predicted_water

        # Mostrar la imagen en escala más pequeña para el informe y auditoría visual.
        step = max(1, int(np.ceil(max(ref.height, ref.width)/1100)))
        vv_show = vv[::step, ::step]
        valid_show = sar_valid[::step, ::step]
        p_show = potential_valid[::step, ::step]
        foot_show = covered_valid[::step, ::step]
        lo, hi = np.nanpercentile(vv_show[valid_show], [3, 97])
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        for ax in axs:
            ax.imshow(np.ma.masked_where(~valid_show, vv_show), cmap='gray', vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
        axs[0].set_title('Sentinel-1 VV · 17/03/2017')
        axs[1].imshow(np.ma.masked_where(~p_show, p_show), cmap='autumn', vmin=0, vmax=1, alpha=0.55)
        axs[1].set_title('Extensión POTENCIAL · EMSN038')
        axs[2].imshow(np.ma.masked_where(~foot_show, foot_show), cmap='Blues', vmin=0, vmax=1, alpha=.4)
        axs[2].set_title('Footprints ópticos 18/21 marzo · NO son etiquetas')
        fig.suptitle('EMSN038 · diagnóstico espacial (NO es IoU de agua observada)')
        fig.savefig(a.output_dir/'diagnostico_espacial.png', dpi=150)
        plt.close(fig)

        (a.output_dir/'diagnostico_emsn038.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
        if rows:
            with (a.output_dir/'solapamiento_modelos.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        print('\n=== DIAGNÓSTICO DE CAPAS ===')
        print('Potencial máximo:', len(potential), '| fuentes:', sources)
        print('Footprints posteriores:', len(post_event), '/', len(footprints))
        print('SAR válido:', f'{int(sar_valid.sum()):,}')
        print('Zona potencial dentro de SAR válido:', f'{int(potential_valid.sum()):,}')
        print('Zona potencial en footprints posteriores:', f'{int(potential_post.sum()):,}')
        for r in rows:
            print(r['modelo'], 'predichos en potencial:', f'{r["pixeles_agua_predicha_en_potencial"]:,}',
                  'fracción de agua predicha ubicada allí:',
                  f'{r["fraccion_predicha_en_zona_potencial"]:.3f}' if r['fraccion_predicha_en_zona_potencial'] is not None else 'indefinido')
        print('\nAVISO: estos son solapamientos descriptivos, NO métricas de segmentación del agua.')
        print('Archivos:', a.output_dir)


if __name__ == '__main__':
    main()
