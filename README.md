# Segmentación de agua con Sentinel-1 y U-Net

Proyecto de Redes Neuronales y Aprendizaje Profundo, UNI, 2026-II, pregunta 2. U-Net propia en PyTorch con bandas SAR VV/VH. Se comparan etiquetas manuales frente a manuales + débiles, y pérdidas BCE frente a BCE + Dice.

## 1. Entorno e instalación inicial

Las dependencias están en [requirements.txt](requirements.txt). Se propone Python 3.12. El modo `full` requiere GPU CUDA e instalación de PyTorch compatible; `reuse` también admite CPU, con mayor tiempo de evaluación.

Desde la raíz del repositorio, en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En los comandos siguientes, `python` debe corresponder al entorno donde se instalaron las dependencias. En Windows puede sustituirse por `.\.venv\Scripts\python.exe` sin activar el entorno. Comprobar CUDA antes de entrenar:

```powershell
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

Las dependencias usan intervalos; no se dispone de un bloqueo exacto del entorno original. Una semilla fija no garantiza igualdad numérica entre hardware y versiones.

## 2. Insumos necesarios

Los datos pesados y checkpoints no se suben a GitHub. Los archivos `.gitkeep` solo conservan la estructura.

| Insumo | Ruta esperada | Preparación |
|---|---|---|
| Sen1Floods11 v1.1 | `data/raw/sen1floods11/` | Descargador incluido; ver comando y estructura siguientes. |
| SAR de Lima procesado | `data/processed/lima_2017/final/lima_20170317_VV_VH_dB.tif` | Escena del 17/03/2017, procesada en SNAP según el informe. |
| Geodatabase EMSN038 | `data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb/` | Descargar desde la activación Copernicus EMSN038. |
| Cuatro archivos `best.pt` | Carpetas de M1/M2 del run gold y M3/M4 del run weak | Solo se necesitan de antemano para `reuse`; `full` los genera. |

Fuentes: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), [Copernicus EMSN038](https://mapping.emergency.copernicus.eu/activations/EMSN038/), [Copernicus Data Space](https://dataspace.copernicus.eu/) y [ASF Vertex](https://search.asf.alaska.edu/). Ver también [data/README.md](data/README.md).

Descargar Sen1Floods11 como preparación inicial:

```powershell
python -m src.descargar_sen1floods11 --dest data/raw/sen1floods11 --mode all --optical
```

Requiere Internet y espacio para varios GB. `--optical` añade Sentinel-2 para los paneles visuales; la red solo recibe VV/VH. Conservar estas carpetas dentro de `data/raw/sen1floods11/`:

| Carpeta | Contenido |
|---|---|
| `data/flood_events/HandLabeled/S1Hand/` | SAR manual. |
| `data/flood_events/HandLabeled/LabelHand/` | Máscaras manuales. |
| `data/flood_events/HandLabeled/S2Hand/` | Óptica de apoyo visual. |
| `data/flood_events/WeaklyLabeled/S1Weak/` | SAR débil. |
| `data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/` | Máscaras débiles. |
| `splits/flood_handlabeled/` | CSV `flood_train_data.csv`, `flood_valid_data.csv`, `flood_test_data.csv` y `flood_bolivia_data.csv`. |

El GeoTIFF de Lima debe tener dos bandas en orden **VV/VH**, valores en **dB**, píxel de **10 m** y CRS **EPSG:32718** para la escena descrita. El archivo del informe tiene 3,404 columnas y 3,287 filas. El ejecutor comprueba dos bandas y CRS; esto no acredita por sí solo la escala radiométrica, el orden o la procedencia del archivo.

**Alcance:** la instalación, las descargas y la preparación del GRD en SNAP se realizan primero. El comando siguiente ejecuta el pipeline de aprendizaje y análisis desde esos insumos locales preparados. No automatiza SNAP ni reproduce directamente desde un GRD recién descargado.

## 3. Ejecución de extremo a extremo con un único comando

Desde la **raíz del repositorio**, con el entorno y los insumos preparados, ejecutar esta única línea:

```powershell
python -m src.run_all --mode full --weak-batch-size 4 --weak-limit 2048 --amp
```

Este es el comando principal de entrenamiento y evaluación de la entrega. **No requiere checkpoints previos.** Ejecuta automáticamente:

1. Comprobación de insumos y CUDA.
2. Entrenamiento de M1 (BCE) y M2 (BCE + Dice) con etiquetas manuales.
3. Entrenamiento de M3/M4 con 252 imágenes manuales y hasta 2,048 pares débiles admisibles.
4. Validación y obtención de cualquier archivo de métricas de validación faltante.
5. Evaluación M1–M4 en test interno y Bolivia.
6. Inspección EMSN038 e inferencia de los cuatro modelos en Lima.
7. Diagnóstico espacial y mapas comparativos.
8. Exportación del resumen CSV y JSON.

Una vez iniciado, no necesita comandos intermedios. Si una etapa falla, el flujo se detiene con un código de salida distinto de cero; los resultados parciales se conservan para diagnosticarlo.

| Configuración del comando | M1/M2 | M3/M4 |
|---|---|---|
| Arquitectura | U-Net, base 16 | U-Net, base 16 |
| Entrada y semilla nominal | VV/VH, 512 × 512, semilla 42 | VV/VH, 512 × 512, semilla 42 |
| Lote | 4 | 4, equilibrado entre manuales y débiles |
| Tasa de aprendizaje | 0.001 | 0.0003 |
| Épocas máximas | 15 | 12 |
| Parada temprana | No | Paciencia 5 |
| Pares débiles | 0 | Límite 2,048; comprobar número efectivo en la auditoría |

Antes del muestreo débil se excluyen Bolivia y solapamientos con particiones retenidas. AMP se solicita mediante `--amp`, pero el código lo desactiva si la capacidad CUDA es menor que 8.0. Normalización: `clip(dB, -50, 5)` y `(dB + 50) / 55`; umbral de predicción: 0.5.

`full` crea **nuevas ejecuciones** con esas condiciones y no sobrescribe los runs del informe. Los resultados numéricos pueden variar. Si hay menos pares débiles admisibles o cambia el dataset local, revisar `auditoria_weak.json` antes de compararlo con los 2,048 pares del informe.

## 4. Repetir la evaluación del informe sin entrenar

Con los cuatro `best.pt` originales disponibles en la computadora, ejecutar esta única línea:

```powershell
python -m src.run_all --mode reuse --gold-run outputs/experimentos_gold/run_20261029_013146_547224 --weak-run outputs/experimentos_weak/run_20261029_035455_467350
```

Se necesitan `M1_BCE/best.pt` y `M2_BCE_Dice/best.pt` dentro del run gold; `M3_Weak_BCE/best.pt` y `M4_Weak_BCE_Dice/best.pt` dentro del run weak. No se aporta una URL externa de esos pesos en el material recibido. Una clonación limpia sin checkpoints debe usar `full` o conseguir los pesos por otro medio.

`reuse` no entrena. Obtiene validación si falta, vuelve a evaluar test/Bolivia en las carpetas de los modelos y crea nuevos resultados de Lima y resumen. Los JSON y figuras de test/Bolivia se actualizan al reevaluar. Si se omiten las dos rutas de runs, selecciona las ejecuciones completas más recientes; para la entrega se recomienda fijarlas.

## 5. Comprobar insumos y plan antes de ejecutar

Para el entrenamiento:

```powershell
python -m src.run_all --mode full --weak-batch-size 4 --weak-limit 2048 --amp --dry-run
```

Para los pesos del informe:

```powershell
python -m src.run_all --mode reuse --gold-run outputs/experimentos_gold/run_20261029_013146_547224 --weak-run outputs/experimentos_weak/run_20261029_035455_467350 --dry-run
```

`--dry-run` valida los insumos y muestra las etapas sin entrenar ni evaluar. También requiere los datos locales; en `full` comprueba CUDA y en `reuse` comprueba checkpoints. Superarlo no demuestra una ejecución completa.

```powershell
python -m src.run_all --help
```

Las opciones `--data-root`, `--sar` y `--gdb` permiten otras ubicaciones. Las rutas relativas se resuelven respecto a la raíz del proyecto. Lanzar `python -m src.run_all` desde esa raíz asegura que Python encuentre `src`.

## 6. Salidas y comprobación de finalización

Cada ejecución crea un identificador `run_FECHA_HORA_MICROSEGUNDOS`.

| Ruta | Resultado esperado |
|---|---|
| `outputs/experimentos_gold/run_.../` | M1/M2, configuración y resultados; nuevos pesos solo en `full`. |
| `outputs/experimentos_weak/run_.../` | M3/M4, auditoría e historiales; nuevos pesos solo en `full`. |
| `RUN/MODELO/evaluacion/metricas_valid.json` | Métricas de validación; RUN es la carpeta gold o weak. |
| `RUN/MODELO/test/metricas_test.json` | Métricas y conteos de test. |
| `RUN/MODELO/bolivia/metricas_bolivia.json` | Métricas de Bolivia. |
| `outputs/lima_2017/run_.../MODELO/` | Probabilidades, predicción georreferenciada, figuras y `metricas_lima.json`. |
| `outputs/pipeline/run_.../comparacion_visual/` | Comparaciones M1–M4 con EMSN038. |
| `outputs/pipeline/run_.../diagnostico_emsn038/` | Diagnóstico espacial y solapamientos. |
| `outputs/pipeline/run_.../resumen_modelos.csv` | Cuatro filas, una por modelo, con métricas y brechas disponibles. |
| `outputs/pipeline/run_.../resumen_ejecucion.json` | Rutas, fecha, referencia utilizada y resultados de los cuatro modelos. |

Para considerar finalizada una ejecución real debe dejar **código de salida 0**, imprimir `Resumen final: .../resumen_modelos.csv` y producir ambos resúmenes. En PowerShell, inmediatamente después del pipeline, `echo $LASTEXITCODE` muestra el código. Un cero del instalador, de `--help` o de `--dry-run` no demuestra que el pipeline real haya terminado.

Para revisar las cuatro filas del resumen más reciente en PowerShell:

```powershell
$ultimaEjecucion = Get-ChildItem .\outputs\pipeline -Directory -Filter 'run_*' | Sort-Object Name | Select-Object -Last 1
Import-Csv (Join-Path $ultimaEjecucion.FullName 'resumen_modelos.csv') | Format-Table modelo,iou_test,iou_bolivia,iou_lima
```

La duración depende del hardware y de la lectura/escritura. `full` vuelve a entrenar las cuatro redes; `reuse` evalúa pesos existentes. No se garantiza un tiempo fijo.

## 7. Interpretación de Lima

Sin máscara binaria local validada, Lima produce predicciones y contraste espacial **exploratorio**: las métricas quedan `null` en JSON y vacías en CSV. Es el comportamiento previsto. No usar el exterior de `Potential_Maximum_flood` ni los footprints ópticos como negativos verificados.

Si existe una máscara revisada con `0=no agua`, `1=agua`, `-1/255=ignorar`, alineada exactamente con el SAR, añadir `--reference-mask RUTA_A_LA_MASCARA.tif`. La alineación no garantiza validez temporal o semántica. Bolivia es una prueba geográfica complementaria; no valida Perú.

## 8. Informe y evidencia publicada

- `reports/informe/Informe_Final_Sentinel1_UNet_NeurIPS_2026.pdf`: informe académico.
- `reports/informe/fuentes/`: LaTeX y figuras; compilar `main.tex` con pdfLaTeX.
- `reports/resultados/`: métricas e historiales que respaldan el informe.
- `data/README.md`: fuentes y organización de insumos.
- `PUBLICAR_GITHUB.md`: instrucciones para publicar desde la cuenta del grupo.

Los datos pesados, checkpoints y salidas locales se excluyen con `.gitignore`. Las evidencias pequeñas se publican en `reports/`. Completar los cinco autores y la URL real de GitHub antes de presentar.

## 9. Estado de verificación

Se comprobaron sintaxis, interfaz y coordinación de etapas mediante pruebas con salidas simuladas. Las pruebas verifican argumentos, resumen y detención ante fallos; **no ejecutan las redes ni acreditan rendimiento**.

```powershell
python -m unittest discover -s tests -v
```

La ejecución completa real no se repitió en el entorno donde se preparó este paquete por no disponer de los datos y pesos. Para cerrar la comprobación de entrega, ejecutar en el equipo con los insumos y conservar los resúmenes y el código de salida descritos arriba. Documentar un comando no sustituye demostrar que terminó.
