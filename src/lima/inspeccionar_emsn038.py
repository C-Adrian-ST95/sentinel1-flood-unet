"""Inspecciona capas de EMSN038 SIN modificar la geodatabase.

Ejecutar desde la raiz del proyecto:
    python -m src.lima.inspeccionar_emsn038 --gdb ... --sar ...
Genera un resumen JSON con nombres de capas, campos y valores de muestra.
No supone que los poligonos sean inundacion ni define automaticamente la cobertura.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import defaultdict
from itertools import islice
from pathlib import Path


def simplify(value, max_len=100):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    return text[:max_len] + ('...' if len(text) > max_len else '')


def overlaps(b1, b2):
    if b1 is None or b2 is None:
        return None
    return b1[0] <= b2[2] and b1[2] >= b2[0] and b1[1] <= b2[3] and b1[3] >= b2[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gdb', required=True, type=Path, help='Carpeta .gdb de EMSN038')
    p.add_argument('--sar', required=True, type=Path, help='GeoTIFF SAR Lima 17/03/2017')
    p.add_argument('--max-samples', type=int, default=200,
                   help='Maximo de entidades a revisar por capa para valores de campos')
    p.add_argument('--output', default='reports/lima_2017/inspeccion_emsn038.json', type=Path)
    args = p.parse_args()
    if not args.gdb.exists():
        p.error(f'No existe la geodatabase: {args.gdb}')
    if not args.sar.is_file():
        p.error(f'No existe el GeoTIFF: {args.sar}')
    if args.max_samples < 1:
        p.error('--max-samples debe ser >= 1')

    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as e:
        p.error(f'Falta rasterio en el .venv: {e}')
    try:
        fiona = importlib.import_module('fiona')
    except ImportError:
        fiona = None
    if fiona is None:
        try:
            import pyogrio
        except ImportError:
            p.error('Para abrir la geodatabase instala fiona o pyogrio en el .venv')
        backend = 'pyogrio'
    else:
        backend = 'fiona'

    with rasterio.open(args.sar) as raster:
        raster_crs = str(raster.crs)
        raster_bbox = tuple(raster.bounds)
        raster_info = {
            'ruta': str(args.sar), 'crs': raster_crs,
            'bbox': list(raster_bbox), 'resolucion_m': list(raster.res),
            'ancho': raster.width, 'alto': raster.height,
        }

    try:
        layer_names = (fiona.listlayers(str(args.gdb)) if fiona is not None
                       else pyogrio.list_layers(str(args.gdb))[:, 0].tolist())
    except Exception as exc:
        raise SystemExit('No se pudo abrir la geodatabase. Comprueba que Fiona/GDAL soporte OpenFileGDB y que la ruta sea correcta.\n' + str(exc))
    print(f'\nEMSN038: {len(layer_names)} capas | SAR CRS: {raster_crs} | lector: {backend}')
    print('AVISO: los valores de atributos proceden de una muestra, no necesariamente de toda la capa.\n')

    layers = []
    for layer_name in layer_names:
        report = {'nombre': layer_name}
        try:
            if fiona is not None:
                with fiona.open(str(args.gdb), layer=layer_name) as ds:
                    report['tipo_geometria'] = ds.schema.get('geometry')
                    report['cantidad_entidades'] = len(ds)
                    crs = ds.crs_wkt or ds.crs
                    report['crs'] = str(ds.crs) if ds.crs else ('WKT disponible' if ds.crs_wkt else None)
                    report['campos'] = {k: str(v) for k, v in ds.schema.get('properties', {}).items()}
                    bbox = tuple(ds.bounds) if ds.bounds is not None else None
                    samples = (feature.get('properties') or {}
                               for feature in islice(ds, args.max_samples))
                    sample_properties = [dict(props) for props in samples]
            else:
                info = pyogrio.read_info(str(args.gdb), layer=layer_name)
                report['tipo_geometria'] = info['geometry_type']
                report['cantidad_entidades'] = int(info['features'])
                crs = info['crs']
                report['crs'] = str(crs) if crs else None
                report['campos'] = dict(zip(info['fields'].tolist(), info['dtypes'].tolist()))
                bbox = tuple(info['total_bounds']) if info['total_bounds'] is not None else None
                frame = pyogrio.read_dataframe(str(args.gdb), layer=layer_name,
                                              max_features=args.max_samples, read_geometry=False)
                sample_properties = frame.to_dict('records')
            report['bbox_origen'] = list(bbox) if bbox is not None else None
            try:
                report['bbox_en_crs_sar'] = list(transform_bounds(crs, raster_crs, *bbox)) if crs and bbox else None
                report['bbox_intersecta_sar'] = overlaps(report['bbox_en_crs_sar'], raster_bbox)
            except Exception as exc:
                report['bbox_intersecta_sar'] = None
                report['nota_crs'] = str(exc)

            unique = defaultdict(list)
            seen = defaultdict(set)
            for props in sample_properties:
                for key, value in props.items():
                    k = str(key)
                    v = simplify(value)
                    token = str(v)
                    if token not in seen[k] and len(unique[k]) < 8:
                        unique[k].append(v)
                        seen[k].add(token)
            report['entidades_muestreadas'] = len(sample_properties)
            report['valores_ejemplo'] = dict(unique)
        except Exception as exc:
            report['error'] = repr(exc)
        layers.append(report)
        print(f"CAPA: {layer_name}")
        if report.get('error'):
            print('  ERROR:', report['error'])
            continue
        print(f"  Entidades: {report['cantidad_entidades']} | Geometria: {report['tipo_geometria']} | Intersecta extension SAR: {report.get('bbox_intersecta_sar')}")
        print(f"  CRS: {report.get('crs')}")
        fields = list(report['campos'])
        print('  Campos:', ', '.join(fields[:25]) + (' ...' if len(fields) > 25 else ''))
        for key in fields[:25]:
            examples = report['valores_ejemplo'].get(key, [])
            if examples:
                print(f'    {key}: {examples[:6]}')
        print()

    summary = {
        'observacion': 'Una capa de poligonos de inundacion sola NO define las zonas negativas. Debe verificarse una capa de cobertura cartografiada. Potential_Maximum_flood indica maximo potencial, no necesariamente inundacion observada.',
        'lector': backend,
        'gdb': str(args.gdb), 'sar': raster_info,
        'capas': layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'RESUMEN GUARDADO: {args.output}')
    print('Potential_Maximum_flood NO debe usarse como verdad terreno sin verificar su significado.')
    print('Comparte este JSON para seleccionar flood-layer, flood-filter y coverage-layer de forma fundamentada.')


if __name__ == '__main__':
    main()
