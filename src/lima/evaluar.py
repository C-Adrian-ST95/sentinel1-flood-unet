"""Inferencia georreferenciada de M1-M4 en Lima 2017 con una U-Net ya entrenada.

Predice sin etiquetas. Calcula IoU/F1 SOLO con referencia binaria verificada o
poligonos de inundacion + huella de cobertura cartografiada.

Ejemplos en LEEME_LIMA.md. No modifica src/model.py ni los checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import torch

from ..model import UNet


CHECKPOINTS = {
    'M1_BCE': 'gold',
    'M2_BCE_Dice': 'gold',
    'M3_Weak_BCE': 'weak',
    'M4_Weak_BCE_Dice': 'weak',
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sar', type=Path, required=True, help='GeoTIFF: banda 1 VV_dB y banda 2 VH_dB')
    p.add_argument('--gold-run', type=Path, required=True, help='Carpeta del run M1/M2')
    p.add_argument('--weak-run', type=Path, required=True, help='Carpeta del run M3/M4')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/lima_2017/evaluacion'))
    p.add_argument('--chip-size', type=int, default=512)
    p.add_argument('--stride', type=int, default=384, help='Paso de ventana; menor que chip-size para solapamiento')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--reference-mask', type=Path, help='Raster alineado: 0=no agua, 1=agua, -1/255=sin evaluar')
    p.add_argument('--gdb', type=Path, help='GDB EMSN038 que contiene geometrías de inundación')
    p.add_argument('--flood-layer', type=str, help='Capa GDB con polígonos SOLO inundados, revisada en inspección')
    p.add_argument('--flood-filter', type=str, help='Filtro literal campo=valor de la capa de inundación (si tiene otras clases)')
    p.add_argument('--coverage-layer', type=str, help='Capa GDB que define el área donde se cartografió agua Y no agua')
    p.add_argument('--coverage-filter', type=str, help='Filtro opcional campo=valor para huella de cobertura')
    p.add_argument('--assume-outside-dry', action='store_true', help='RIESGO: considerar TODO SAR válido fuera de inundación como no agua. Úsalo SOLO si cobertura completa confirmada.')
    p.add_argument('--all-touched', action='store_true', help='Rasterizar si cualquier parte de píxel toca el polígono')
    p.add_argument('--max-fig-px', type=int, default=1600, help='Límite de resolución de imágenes de vista previa')
    args = p.parse_args()
    if not args.sar.is_file():
        p.error(f'Falta el GeoTIFF SAR: {args.sar}')
    if args.chip_size < 16 or args.chip_size % 8 or not (0 < args.stride <= args.chip_size):
        p.error('chip-size debe ser divisible por 8 y >=16; stride entre 1 y chip-size')
    if not (0 < args.threshold < 1):
        p.error('threshold debe estar entre 0 y 1')
    if args.reference_mask and (args.gdb or args.flood_layer or args.coverage_layer):
        p.error('Utiliza --reference-mask O --gdb con capas, no ambas fuentes')
    if args.gdb and not args.flood_layer:
        p.error('Con --gdb debes indicar --flood-layer confirmado mediante inspección')
    if args.flood_layer and not args.gdb:
        p.error('--flood-layer necesita --gdb')
    if args.coverage_layer and not args.flood_layer:
        p.error('--coverage-layer necesita --flood-layer')
    if args.assume_outside_dry and not args.flood_layer:
        p.error('--assume-outside-dry necesita --gdb y --flood-layer')
    if args.assume_outside_dry and args.coverage_layer:
        p.error('No utilices --assume-outside-dry con --coverage-layer')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA solicitado pero no disponible')
    for model, group in CHECKPOINTS.items():
        run = args.gold_run if group == 'gold' else args.weak_run
        if not (run / model / 'best.pt').is_file():
            p.error(f'Falta checkpoint: {run / model / "best.pt"}')
    return args


def load_sar(path):
    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError(f'El SAR debe tener 2 bandas VV/VH; tiene {src.count}.')
        if src.crs is None:
            raise ValueError('El SAR no tiene CRS.')
        if src.res != (10.0, 10.0):
            print(f'ADVERTENCIA: resolución del SAR {src.res}, no exactamente 10 m.')
        raw = src.read((1, 2), out_dtype='float32', masked=True)
        nodata = np.ma.getmaskarray(raw).any(axis=0)
        raw = np.ma.asarray(raw).astype('float32').filled(np.nan)
        valid = (~nodata) & np.isfinite(raw).all(axis=0) & (raw > -1000).all(axis=0)
        # EXACTAMENTE la misma normalización que src/dataset.py
        x = np.nan_to_num(raw, nan=-50.0, neginf=-50.0, posinf=5.0)
        x = (np.clip(x, -50.0, 5.0) + 50.0) / 55.0
        x[:, ~valid] = 0.0
        meta = {
            'transform': src.transform, 'crs': src.crs, 'height': src.height,
            'width': src.width, 'profile': src.profile.copy(), 'nodata_original': src.nodata,
            'descriptions': src.descriptions, 'res': src.res,
        }
    print(f'SAR cargado {meta["width"]}x{meta["height"]} | validos={int(valid.sum()):,} ({100*valid.mean():.2f}%)')
    print('Banda 1:', meta['descriptions'][0], '| Banda 2:', meta['descriptions'][1])
    return raw, x.astype(np.float32, copy=False), valid, meta


def read_vector_shapes(gdb, layer, target_crs, filter_expr=None):
    """Reproyectar geometrías, sin adivinar campos o clases de EMSN038."""
    key, expected = (filter_expr.split('=', 1) if filter_expr else (None, None))
    geometries = []
    try:
        fiona = importlib.import_module('fiona')
    except ImportError:
        fiona = None
    if fiona is not None:
        with fiona.open(gdb, layer=layer) as ds:
            source_crs = ds.crs_wkt or ds.crs
            if key and key not in ds.schema.get('properties', {}):
                raise ValueError(f'No existe el campo {key} en {layer}. Revisa inspeccionar_emsn038.py')
            total = 0
            for ft in ds:
                total += 1
                if key and str(ft['properties'].get(key)) != expected:
                    continue
                geo = ft.get('geometry')
                if geo:
                    geometries.append(dict(geo))
    else:
        try:
            import pyogrio
            from shapely.geometry import mapping
        except ImportError as exc:
            raise ImportError('Para leer EMSN038 instala fiona o pyogrio y shapely') from exc
        info = pyogrio.read_info(gdb, layer=layer)
        if key and key not in info['fields']:
            raise ValueError(f'No existe el campo {key} en {layer}. Revisa inspeccionar_emsn038.py')
        ds = pyogrio.read_dataframe(gdb, layer=layer, columns=[key] if key else [])
        source_crs = ds.crs
        total = len(ds)
        for _, ft in ds.iterrows():
            if key and str(ft[key]) != expected:
                continue
            geo = ft.geometry
            if geo is not None and not geo.is_empty:
                geometries.append(mapping(geo))
    if not source_crs:
        raise ValueError(f'Falta CRS en la capa {layer}')
    geometries = [transform_geom(source_crs, str(target_crs), geo) for geo in geometries]
    for geom in geometries:
        if geom.get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError(f'La capa {layer} contiene geometrías de tipo {geom.get("type")}; se esperan polígonos')
    print(f'Capa {layer}: {len(geometries)}/{total} geometrías seleccionadas')
    if not geometries:
        raise ValueError(f'No hay polígonos utilizables en la capa {layer} con filtro {filter_expr}')
    return geometries


def load_reference(args, meta, sar_valid):
    """Devuelve mask int16 [-1,0,1], origen y si tiene etiquetas negativas fiables."""
    shape = (meta['height'], meta['width'])
    if args.reference_mask:
        with rasterio.open(args.reference_mask) as ref:
            if (ref.width, ref.height) != (meta['width'], meta['height']):
                raise ValueError('La referencia tiene diferente tamaño que el SAR; debes alinear grillas antes de evaluar.')
            if ref.crs != meta['crs'] or not ref.transform.almost_equals(meta['transform']):
                raise ValueError('La referencia no está alineada con SAR (CRS o transform).')
            if ref.nodata == 0:
                raise ValueError('La referencia declara nodata=0; se confundiría con no agua. Usa -1 o 255 para nodata.')
            arr = ref.read(1, masked=True)
            ref_valid = ~np.ma.getmaskarray(arr)
            labels = np.ma.asarray(arr).astype(np.int16).filled(-1)
            bad = set(np.unique(labels[ref_valid]).tolist()) - {0, 1, -1, 255}
            if bad:
                raise ValueError(f'La máscara de referencia debe usar 0=no agua,1=agua,-1/255=ignorar. Valores inesperados: {bad}')
            labels[(~ref_valid) | (labels == 255)] = -1
        labels[~sar_valid] = -1
        return labels, 'Máscara GeoTIFF binaria suministrada', True
    if args.gdb:
        flooding = read_vector_shapes(args.gdb, args.flood_layer, meta['crs'], args.flood_filter)
        flooded = rasterize([(g, 1) for g in flooding], out_shape=shape,
                            transform=meta['transform'], all_touched=args.all_touched,
                            fill=0, dtype='uint8').astype(bool)
        if not (flooded & sar_valid).any():
            raise ValueError('Los polígonos de inundación no intersectan píxeles válidos del SAR. Comprueba capa/CRS/recorte.')
        labels = np.full(shape, -1, dtype='int16')
        if args.coverage_layer:
            coverage = read_vector_shapes(args.gdb, args.coverage_layer, meta['crs'], args.coverage_filter)
            covmask = rasterize([(g, 1) for g in coverage], out_shape=shape,
                                transform=meta['transform'], all_touched=args.all_touched,
                                fill=0, dtype='uint8').astype(bool)
            # No asumir que polígonos positivos fuera de cobertura son evidencia negativa.
            outside = int((flooded & ~covmask & sar_valid).sum())
            if outside:
                raise ValueError(f'Hay {outside:,} píxeles de inundación fuera de la huella de cobertura. Comprueba capas.')
            labels[covmask & sar_valid] = 0
            labels[flooded & covmask & sar_valid] = 1
            source = f'EMSN038: inundación={args.flood_layer}; cobertura={args.coverage_layer}'
            reliable = True
        elif args.assume_outside_dry:
            labels[sar_valid] = 0
            labels[flooded & sar_valid] = 1
            source = 'EMSN038 con SUPOSICIÓN EXPLÍCITA: todo SAR válido fuera de polígonos es seco'
            reliable = True
            print('ADVERTENCIA METODOLÓGICA: assumption outside-dry exige justificar cobertura exhaustiva.')
        else:
            labels[flooded & sar_valid] = 1
            source = f'EMSN038: solo polígonos de agua ({args.flood_layer}); fondo SIN etiqueta'
            reliable = False
            print('SIN IoU/F1: no hay evidencia suficiente para etiquetar el resto del área como no agua.')
        return labels, source, reliable
    return None, 'Sin referencia', False


def offsets(length, size, stride):
    if length < size:
        raise ValueError(f'La dimensión SAR {length} es menor que chip-size {size}')
    values = list(range(0, length - size + 1, stride))
    if not values or values[-1] != length - size:
        values.append(length - size)
    return values


def run_inference(model, x, valid, chip_size, stride, device):
    h, w = valid.shape
    rows = offsets(h, chip_size, stride)
    cols = offsets(w, chip_size, stride)
    # Ventana suavizada: elimina discontinuidades al unir predicciones.
    wy = np.maximum(np.hanning(chip_size).astype('float32'), 0.1)
    wt = np.outer(wy, wy).astype('float32')
    acc = np.zeros((h, w), dtype='float32')
    weights = np.zeros((h, w), dtype='float32')
    model.eval()
    start = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for r in rows:
            for c in cols:
                area = valid[r:r+chip_size, c:c+chip_size]
                if not area.any():
                    continue
                tile = torch.from_numpy(np.ascontiguousarray(x[:,r:r+chip_size,c:c+chip_size]))[None].to(device)
                pred = torch.sigmoid(model(tile))[0, 0].cpu().numpy().astype('float32', copy=False)
                good_w = wt * area
                acc[r:r+chip_size, c:c+chip_size] += pred * good_w
                weights[r:r+chip_size, c:c+chip_size] += good_w
                processed += 1
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    probs = np.full((h, w), -9999, dtype='float32')
    covered = valid & (weights > 0)
    probs[covered] = acc[covered] / weights[covered]
    print(f'  {processed}/{len(rows)*len(cols)} ventanas inferidas en {time.perf_counter()-start:.2f}s')
    return probs, processed, round(time.perf_counter()-start, 2)


def save_geotiff(path, arr, meta, nodata, dtype):
    profile = dict(meta['profile'])
    # No heredar metadata de compresión/tiling accidental incompatible con dtype.
    profile.update(driver='GTiff', count=1, dtype=dtype, nodata=nodata,
                   compress='deflate', predictor=3 if dtype == 'float32' else 1)
    for k in ('photometric', 'alpha', 'colorinterp'):
        profile.pop(k, None)
    if profile.get('tiled'):
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile['tiled'] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype(dtype, copy=False), 1)


def calculate_metrics(pred, reference, evaluable, chip_size):
    p = pred[evaluable]
    y = reference[evaluable] == 1
    tp = int((p & y).sum()); fp = int((p & ~y).sum())
    fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    union = tp + fp + fn
    result = {
        'iou_global': tp / union if union else None,
        'f1_agua': 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) else None,
        'precision_agua': tp/(tp+fp) if tp+fp else None,
        'recall_agua': tp/(tp+fn) if tp+fn else None,
        'tp':tp,'fp':fp,'fn':fn,'tn':tn, 'pixeles_evaluados': int(evaluable.sum())
    }
    per_chip = []
    h, w = reference.shape
    # Chips disjuntos, con chips parciales en los bordes, sin contar un mismo píxel dos veces.
    for r in range(0, h, chip_size):
        for c in range(0, w, chip_size):
            v = evaluable[r:r+chip_size,c:c+chip_size]
            if not v.any():
                continue
            pp = pred[r:r+chip_size,c:c+chip_size][v]
            yy = reference[r:r+chip_size,c:c+chip_size][v] == 1
            a = int((pp & yy).sum())
            b = int((pp & ~yy).sum())
            d = int((~pp & yy).sum())
            if a+b+d:
                per_chip.append(a/(a+b+d))
    result['iou_media_chips'] = sum(per_chip)/len(per_chip) if per_chip else None
    result['chips_iou_definido'] = len(per_chip)
    return result


def contrast(band, valid):
    good = band[valid & np.isfinite(band)]
    if not good.size:
        return np.zeros(band.shape, dtype='float32')
    low, hi = np.percentile(good, (2,98))
    return np.clip((np.nan_to_num(band, nan=low)-low)/max(hi-low,1e-6), 0, 1)


def save_figure(path, raw, valid, probs, classified, reference, reference_evaluable, source, metric, max_px):
    h,w=valid.shape
    step = max(1, math.ceil(max(h,w)/max_px))
    sl = np.s_[::step, ::step]
    vv, vh = contrast(raw[0],valid)[sl], contrast(raw[1],valid)[sl]
    pr = np.ma.masked_where(~valid[sl], probs[sl])
    pred = np.where(valid[sl], classified[sl], 2)
    maskcmap = ListedColormap(['#e5e7eb','#0284c7','#111827'])
    fig, ax = plt.subplots(2,3,figsize=(15,9))
    ax[0,0].imshow(vv,cmap='gray',vmin=0,vmax=1); ax[0,0].set_title('Sentinel-1 VV (dB, contraste)')
    ax[0,1].imshow(vh,cmap='gray',vmin=0,vmax=1); ax[0,1].set_title('Sentinel-1 VH (dB, contraste)')
    pcm = ax[0,2].imshow(pr,cmap='viridis',vmin=0,vmax=1); ax[0,2].set_title('Probabilidad de agua (U-Net)')
    fig.colorbar(pcm,ax=ax[0,2],shrink=.75)
    ax[1,0].imshow(pred,cmap=maskcmap,vmin=0,vmax=2); ax[1,0].set_title('Predicción: agua azul')
    if reference is not None:
        refview=np.where(reference[sl]<0,2,reference[sl])
        ax[1,1].imshow(refview,cmap=maskcmap,vmin=0,vmax=2)
        ax[1,1].set_title('EMSN038: referencia (negro=sin etiqueta)')
    else:
        ax[1,1].text(.5,.5,'Referencia EMSN038 no preparada',ha='center',va='center',transform=ax[1,1].transAxes)
        ax[1,1].set_title('Referencia')
    if reference_evaluable is not None and reference_evaluable.any():
        v=reference_evaluable[sl]
        p=classified[sl]
        truth=reference[sl]==1
        e=np.zeros(v.shape,dtype='uint8')
        e[v & p & truth]=1
        e[v & p & ~truth]=2
        e[v & ~p & truth]=3
        e[~v]=4
        cm=ListedColormap(['#e5e7eb','#16a34a','#ef4444','#facc15','#111827'])
        ax[1,2].imshow(e,cmap=cm,vmin=0,vmax=4)
        ax[1,2].set_title(f'Errores | IoU={metric["iou_global"]:.3f}' if metric and metric['iou_global'] is not None else 'Errores')
        patches=[Patch(facecolor=c,label=n) for c,n in [('#16a34a','TP agua'),('#ef4444','FP falsa alarma'),('#facc15','FN agua omitida'),('#111827','Sin evaluar')]]
    else:
        ax[1,2].text(.5,.5,'No se calculan errores\nsin área evaluable de no agua',ha='center',va='center',transform=ax[1,2].transAxes)
        ax[1,2].set_title('Métricas no disponibles')
        patches=[Patch(facecolor='#0284c7',label='Agua'),Patch(facecolor='#e5e7eb',label='No agua'),Patch(facecolor='#111827',label='Sin evaluar/NoData')]
    for axis in ax.flat: axis.axis('off')
    fig.legend(handles=patches,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Lima, 17 marzo 2017 | ' + source[:90],fontsize=12)
    fig.tight_layout(rect=[0,.06,1,.96])
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=140,bbox_inches='tight')
    plt.close(fig)


def build_chip_rows(pred, reference, evaluable, valid, chip_size, meta):
    """Inventario de chips disjuntos para analizar errores desde un notebook."""
    rows=[]
    h,w=valid.shape
    for r in range(0,h,chip_size):
        for c in range(0,w,chip_size):
            r2=min(r+chip_size,h);c2=min(c+chip_size,w)
            v=valid[r:r2,c:c2]
            if not v.any():
                continue
            pp=pred[r:r2,c:c2]
            x_utm,y_utm=rasterio.transform.xy(meta['transform'],r,c,offset='ul')
            record={
                'fila_inicio':r,'col_inicio':c,'alto':r2-r,'ancho':c2-c,
                'x_utm18s_inicio':round(x_utm,2),'y_utm18s_inicio':round(y_utm,2),
                'pixeles_validos':int(v.sum()),
                'pixeles_agua_predichos':int((pp & v).sum()),
                'fraccion_agua_predicha':float((pp & v).sum()/v.sum()),
                'iou_chip':None,'tp':None,'fp':None,'fn':None,'tn':None,
                'pixeles_evaluados':None,
            }
            if evaluable is not None:
                use=evaluable[r:r2,c:c2]
                if use.any():
                    p=pp[use]
                    y=reference[r:r2,c:c2][use]==1
                    tp=int((p & y).sum());fp=int((p & ~y).sum())
                    fn=int((~p & y).sum());tn=int((~p & ~y).sum())
                    record.update(tp=tp,fp=fp,fn=fn,tn=tn,
                                  pixeles_evaluados=int(use.sum()),
                                  iou_chip=(tp/(tp+fp+fn) if tp+fp+fn else None))
            rows.append(record)
    return rows


def main():
    a=get_args()
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    raw,x,valid,meta=load_sar(a.sar)
    reference,ref_source,reliable=load_reference(a,meta,valid)
    evaluate=bool(reference is not None and reliable and np.any(reference==0) and np.any(reference==1))
    if reliable and not evaluate:
        print('AVISO: referencia no contiene ambas clases evaluables (0 y 1), se omiten IoU/F1.')
    if reference is not None:
        save_geotiff(a.output_dir/'referencia_lima.tif', reference,meta,-1,'int16')
    if not evaluate:
        print('MODO INFERENCIA: se guardarán mapas, pero NO métricas cuantitativas.')
        if reference is None:
            print('CAUSA: no se indicó --reference-mask ni --gdb con --flood-layer y --coverage-layer verificados.')
    if a.gdb and not a.coverage_layer and not a.assume_outside_dry:
        print('La referencia solo contiene positivos; fuera del polígono queda SIN ETIQUETA.')
    print('Dispositivo:',device)
    results=[]
    for name,kind in CHECKPOINTS.items():
        ckpt_path=(a.gold_run if kind=='gold' else a.weak_run)/name/'best.pt'
        print('\n'+('='*60)+'\nEVALUANDO '+name+' | '+str(ckpt_path),flush=True)
        checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=True)
        model=UNet(in_channels=2,out_channels=1,base=int(checkpoint.get('base',16)))
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model=model.to(device).eval()
        p,tiles,seconds=run_inference(model,x,valid,a.chip_size,a.stride,device)
        pred=(p>=a.threshold) & valid
        out=a.output_dir/name
        out.mkdir(parents=True,exist_ok=True)
        save_geotiff(out/'probabilidad_agua_lima.tif', p,meta,-9999,'float32')
        binary=np.where(valid,pred.astype('uint8'),255).astype('uint8')
        save_geotiff(out/'mascara_predicha_lima.tif', binary,meta,255,'uint8')
        mask_valid=valid & (reference>=0) if evaluate else None
        metric=calculate_metrics(pred,reference,mask_valid,a.chip_size) if evaluate else None
        row={'modelo':name,'checkpoint':str(ckpt_path),'segundos_inferencia':seconds,'ventanas':tiles,
             'pixeles_SAR_validos':int(valid.sum()),'pixeles_agua_predicha':int(pred.sum()),
             'referencia':ref_source,'evaluacion_cuantitativa':evaluate,
             'iou_global':None,'iou_media_chips':None,'f1_agua':None,'precision_agua':None,'recall_agua':None,'tp':None,'fp':None,'tn':None,'fn':None,'pixeles_evaluados':None}
        if metric:
            row.update(metric)
            # Si la huella de cobertura es muy limitada, reportar el tamaño del dominio evaluable.
            print('MÉTRICAS:', json.dumps(metric,ensure_ascii=False))
        else:
            print('Sin IoU/F1: no hay referencia positiva y negativa validada en el mismo dominio.')
        save_figure(out/'prediccion_lima.png',raw,valid,p,pred,reference,mask_valid,ref_source,metric,a.max_fig_px)
        chip_rows=build_chip_rows(pred,reference,mask_valid,valid,a.chip_size,meta)
        if chip_rows:
            with (out/'chips_lima.csv').open('w',encoding='utf-8-sig',newline='') as csvf:
                dw=csv.DictWriter(csvf,fieldnames=list(chip_rows[0]))
                dw.writeheader();dw.writerows(chip_rows)
            # Guardar 2 chips representativos sin necesidad de un notebook o ejecución extra.
            order=sorted(chip_rows,key=lambda rec: rec['fraccion_agua_predicha'],reverse=True)
            if mask_valid is not None:
                reference_chips=[d for d in order if d['pixeles_evaluados'] and d['tp'] is not None]
                # Si hay evaluación, preferir un chip difícil para el segundo ejemplo.
                difficult=sorted([d for d in reference_chips if d['iou_chip'] is not None],key=lambda d:d['iou_chip'])
                choices=[order[0]]+([difficult[0]] if difficult else order[1:2])
            else:
                choices=order[:2]
            selected=set()
            for chip in choices:
                r,c=chip['fila_inicio'],chip['col_inicio']
                if (r,c) in selected: continue
                selected.add((r,c))
                r2=min(r+a.chip_size,valid.shape[0]);c2=min(c+a.chip_size,valid.shape[1])
                box=(slice(r,r2),slice(c,c2))
                refchip=reference[box] if reference is not None else None
                valchip=mask_valid[box] if mask_valid is not None else None
                metricchip={'iou_global':chip['iou_chip']} if chip['iou_chip'] is not None else None
                save_figure(out/f'chip_lima_f{r}_c{c}.png',raw[:,r:r2,c:c2],valid[box],
                            p[box],pred[box],refchip,valchip,ref_source,metricchip,a.max_fig_px)
        (out/'metricas_lima.json').write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        results.append(row)
        del model,p
        if device.type=='cuda': torch.cuda.empty_cache()
    with (a.output_dir/'comparacion_lima.csv').open('w',encoding='utf-8-sig',newline='') as f:
        dw=csv.DictWriter(f,fieldnames=list(results[0]))
        dw.writeheader(); dw.writerows(results)
    metadata={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'input_sar':str(a.sar),'size':[meta['width'],meta['height']],
        'resolution_m':list(meta['res']), 'sar_crs':str(meta['crs']),
        'bands':['VV_dB','VH_dB'], 'normalization':'clip dB [-50,5] -> (dB+50)/55',
        'chip_size':a.chip_size,'stride':a.stride,'threshold':a.threshold,
        'padded':False,'reference':ref_source, 'metricas_habilitadas':evaluate,
        'advertencia':'EMSN038 óptico del 18/21 de marzo de 2017 puede no coincidir temporal ni semánticamente con SAR del 17/03/2017. No interpretar NoData como no agua.',
        'resultados':results,
    }
    (a.output_dir/'protocolo_lima.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if evaluate:
        fig,ax=plt.subplots(figsize=(8,4))
        ind=np.arange(len(results)); vals=[r['iou_global'] for r in results]
        ax.bar(ind,vals);ax.set_xticks(ind,[r['modelo'].replace('_','\n') for r in results]);ax.set_ylim(0,1)
        ax.set_ylabel('IoU global — Lima');ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(a.output_dir/'comparacion_iou_lima.png',dpi=150);plt.close(fig)
    print('\nEVALUACIÓN COMPLETA. Revisa:',a.output_dir)
    if not evaluate:
        print('PENDIENTE: proporcionar máscara 0/1/-1 alineada, o confirmar una capa de inundación observada y una huella de cobertura válida en EMSN038.')


if __name__=='__main__':
    main()
