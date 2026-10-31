"""Compara predicciones M1-M4 sobre Sentinel-1 con la capa potencial EMSN038.

Las figuras son diagnósticos cualitativos: los polígonos máximos potenciales NO son
etiquetas de agua observada el día del SAR y NO se calcula IoU.

Ejemplo:
python -m src.lima.visualizar_modelos_emsn038 \
 --sar data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif \
 --gdb data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb \
 --predicciones outputs/lima_2017/evaluacion_20261030 \
 --output-dir reports/lima_2017/comparacion_visual
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.plot import plotting_extent

MODELOS = (
    'M1_BCE', 'M2_BCE_Dice', 'M3_Weak_BCE', 'M4_Weak_BCE_Dice'
)


def ventana_zoom(sar, geometria, lado_m):
    centro = geometria.representative_point()
    mitad = lado_m / 2
    w = from_bounds(centro.x-mitad, centro.y-mitad,
                    centro.x+mitad, centro.y+mitad,
                    transform=sar.transform)
    col0 = max(0, int(np.floor(w.col_off)))
    row0 = max(0, int(np.floor(w.row_off)))
    col1 = min(sar.width, int(np.ceil(w.col_off+w.width)))
    row1 = min(sar.height, int(np.ceil(w.row_off+w.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError('La ventana de zoom cae fuera de la escena SAR')
    return Window(col0, row0, col1-col0, row1-row0)


def leer_escena(sar, archivos, ventana, max_lado):
    vv = sar.read(1, window=ventana, masked=True)
    vv = np.ma.filled(vv.astype('float32'), np.nan)
    paso = max(1, int(np.ceil(max(vv.shape) / max_lado)))
    transform = sar.window_transform(ventana)
    extent = plotting_extent(vv, transform)
    predicciones = {}
    for nombre, ruta in archivos.items():
        with rasterio.open(ruta) as ds:
            if ds.count != 1 or ds.crs != sar.crs or (ds.height, ds.width) != (sar.height, sar.width) or not ds.transform.almost_equals(sar.transform):
                raise ValueError(f'{nombre}: máscara y SAR no comparten EXACTAMENTE la misma cuadrícula')
            predicciones[nombre] = (ds.read(1, window=ventana) == 1)[::paso, ::paso]
    return vv[::paso, ::paso], predicciones, extent


def dibujar_comparativa(sar, rutas, pot, rios, ventana, salida, titulo, max_lado):
    vv, preds, extent = leer_escena(sar, rutas, ventana, max_lado)
    valid = np.isfinite(vv)
    lo, hi = np.percentile(vv[valid], (2, 98)) if np.any(valid) else (-30, 5)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    for ax, (nombre, mascara) in zip(axes.flat, preds.items()):
        ax.imshow(vv, extent=extent, cmap='gray', origin='upper', vmin=lo, vmax=hi,
                  interpolation='nearest')
        rgba = np.zeros((*mascara.shape, 4), dtype=np.float32)
        rgba[mascara & valid] = (0.0, 0.55, 1.0, 0.69)
        ax.imshow(rgba, extent=extent, origin='upper', interpolation='nearest')
        if len(pot):
            pot.boundary.plot(ax=ax, color='#ffe600', linewidth=1.65, zorder=5)
        if len(rios):
            rios.boundary.plot(ax=ax, color='#fa842b', linewidth=0.85, zorder=6)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_title(nombre.replace('_', ' '), fontsize=13)
        ax.set_xlabel('Este UTM 18S (m)'); ax.set_ylabel('Norte UTM 18S (m)')
        ax.ticklabel_format(style='plain', useOffset=False)
    fig.suptitle(titulo, fontsize=16)
    fig.legend(handles=[
        Patch(facecolor='#008cff', label='Agua predicha por U-Net'),
        Line2D([0],[0], color='#ffe600', linewidth=2, label='Potencial máximo EMSN038 (NO agua observada)'),
        Line2D([0],[0], color='#fa842b', linewidth=2, label='Polígono fuente «river»'),
    ], loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=11)
    fig.tight_layout(rect=(0, .085, 1, .94), h_pad=3.5, w_pad=2.0)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('Figura guardada:', salida)


def main():
    p = argparse.ArgumentParser(description='Visualiza 4 modelos vs extensión POTENCIAL EMSN038, sin calcular IoU')
    p.add_argument('--sar', required=True)
    p.add_argument('--gdb', required=True)
    p.add_argument('--predicciones', required=True, help='Carpeta con M1_BCE/.../mascara_predicha_lima.tif, etc.')
    p.add_argument('--output-dir', default='reports/lima_2017/comparacion_visual')
    p.add_argument('--zoom-m', type=float, default=6500., help='Lado de la ventana de zoom, en metros')
    args = p.parse_args()

    raiz = Path(args.predicciones)
    rutas = {n: raiz / n / 'mascara_predicha_lima.tif' for n in MODELOS}
    for nombre, ruta in rutas.items():
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta {nombre}: {ruta}')

    capas = gpd.read_file(args.gdb, layer='Potential_Maximum_flood')
    if 'source' not in capas:
        raise ValueError('La capa Potential_Maximum_flood no tiene el campo source')

    salida = Path(args.output_dir)
    with rasterio.open(args.sar) as sar:
        if sar.count < 1 or sar.crs is None:
            raise ValueError('GeoTIFF SAR sin banda VV o sin CRS')
        capas = capas.to_crs(sar.crs)
        source = capas['source'].fillna('').astype(str).str.strip().str.lower()
        pot = capas[source == 'maximun height'].copy()  # Grafía literal del archivo EMSN038
        rios = capas[source == 'river'].copy()
        if pot.empty:
            raise ValueError('No se encontraron polígonos source="maximun height"')
        print(f'Extensiones potenciales: {len(pot)}; ríos: {len(rios)}')
        dibujar_comparativa(sar, rutas, pot, rios, Window(0,0,sar.width,sar.height),
                            salida / '01_comparacion_general.png',
                            'Lima 2017: predicciones M1–M4 vs EMSN038 (diagnóstico, NO IoU)', 1450)
        mayor = pot.iloc[pot.geometry.area.to_numpy().argmax()].geometry
        dibujar_comparativa(sar, rutas, pot, rios, ventana_zoom(sar, mayor, args.zoom_m),
                            salida / '02_comparacion_zoom.png',
                            'Zoom del sector de mayor extensión potencial (NO IoU)', 1800)
    print('IMPORTANTE: esta visualización NO es una evaluación de agua observada.')


if __name__ == '__main__':
    main()
