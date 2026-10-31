# Evaluación Lima 2017 — Sentinel-1 U-Net M1–M4

> La carpeta ya está integrada al proyecto. Para ejecutar el flujo completo con
> una orden, usa `python -m src.run_all --mode full` desde la raíz; para reutilizar
> pesos, usa `python -m src.run_all --mode reuse`. Consulta el README principal.

## Instalación

Descomprime y COPIA `src/lima/` dentro de tu proyecto `D:\maestria\sentinel1-flood-unet\src\lima\`. No modifiques `model.py`, `dataset.py`, `train.py`, ni los checkpoints `.pt`. Ejecuta los comandos **desde la raíz del proyecto** con el `.venv` activo.

Dependencias ya utilizadas en el proyecto: `torch`, `numpy`, `rasterio`, `matplotlib`. Para leer geodatabase agrega `fiona` (`python -m pip install fiona` si falta). Para el primer pase sin máscaras, no hace falta Fiona.

## Paso 1 — Predicciones de los 4 modelos YA ENTRENADOS

```powershell
python -m src.lima.evaluar --sar data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif --gold-run outputs/experimentos_gold/run_20261029_013146_547224 --weak-run outputs/experimentos_weak/run_20261029_035455_467350 --output-dir outputs/lima_2017/evaluacion_20261030
```

Genera para cada modelo `mascara_predicha_lima.tif` (0=no agua, 1=agua, 255=NoData), `probabilidad_agua_lima.tif` (float32, -9999=NoData), `prediccion_lima.png`, `metricas_lima.json`. También `comparacion_lima.csv` y `protocolo_lima.json`.

Este primer paso **NO calcula IoU/F1**: sin etiquetas negativas fiables, no es una evaluación cuantitativa válida.

## Paso 2 — Identificar capas de la geodatabase

```powershell
python -m src.lima.inspeccionar_emsn038 --gdb data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb --sar data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif
```

Revisa qué capa contiene SOLO POLÍGONOS INUNDADOS y qué capa representa la **huella territorial cartografiada con posibilidad de agua y no agua**. Si la capa contiene clases mixtas, usa `--flood-filter "campo=valor"` para quedarte solo con los polígonos inundados. **La cantidad de 13 polígonos no prueba su tipo semántico ni cobertura.**

## Paso 3 — Con referencia válida y huella de cobertura confirmada

SUSTITUYE `NOMBRE_CAPA_INUNDADA` y `NOMBRE_CAPA_COBERTURA` por los nombres reales identificados en el paso 2. No ejecutes literalmente los marcadores.

```powershell
python -m src.lima.evaluar --sar data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif --gold-run outputs/experimentos_gold/run_20261029_013146_547224 --weak-run outputs/experimentos_weak/run_20261029_035455_467350 --gdb data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb --flood-layer "NOMBRE_CAPA_INUNDADA" --coverage-layer "NOMBRE_CAPA_COBERTURA" --output-dir outputs/lima_2017/evaluacion_con_referencia
```

Alternativamente, con una máscara raster ya alineada (0=no inundado, 1=inundado, nodata=-1 o 255):

```powershell
python -m src.lima.evaluar --sar data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif --gold-run outputs/experimentos_gold/run_20261029_013146_547224 --weak-run outputs/experimentos_weak/run_20261029_035455_467350 --reference-mask data/processed/lima_2017/final/mascara_referencia_verificada.tif --output-dir outputs/lima_2017/evaluacion_con_referencia
```

**No uses `--assume-outside-dry`** salvo que puedas justificar que toda la superficie SAR incluida fue efectivamente revisada y etiquetada por EMSN038. Una delineación de positivos (polígonos de agua) no es evidencia de ausencia de inundación fuera de ellos.

## Consideraciones metodológicas

- Se utiliza exactamente la normalización de M1–M4: VV/VH en dB, recorte a [-50, 5] y escala a [0, 1].
- La segmentación usa ventanas 512x512 solapadas cada 384 píxeles, con mezcla ponderada de probabilidades para evitar bordes visibles. No reentrena.
- Las probabilidades son continuas; la máscara predicha aplica umbral 0.5. `NoData` no se trata como no inundado.
- IoU/F1/precision/recall/confusión se calculan solo en el dominio etiquetado con ambas clases. El IoU por chip usa mosaicos **disjuntos** de 512, con bordes parciales.
- EMSN038 usa observaciones ópticas de 18/21 marzo de 2017, frente al SAR del 17 marzo; la diferencia temporal, geográfica y de resolución introduce incertidumbre. **Una evaluación de Lima no reemplaza los criterios de validación** que seleccionaron los modelos.
- `src/lima/` se puede subir a GitHub; no subas las escenas SAR, los archivos `.gdb`, ni los checkpoints `.pt` sin revisar licencias y tamaño.

## Para el notebook y el informe

Cada modelo añade `chips_lima.csv` (posición fila/columna, coordenadas UTM, agua predicha y, cuando exista referencia fiable, confusión e IoU por chip). Genera también `chip_lima_f*_c*.png` (uno o dos ejemplos), útiles para la sección de fallos del SAR. El archivo global `protocolo_lima.json` registra resolución, normalización, umbral, referencias, checkpoint y limitaciones.

Los nombres de columnas de `chips_lima.csv` son adecuados para construir el notebook sin volver a ejecutar inferencia. Si un chip no tiene área etiquetada, sus columnas de métricas quedan vacías y no debe considerarse un chip negativo.
