# Mapeo de inundaciones con Sentinel-1 y U-Net

**Examen parcial 2026-II — Pregunta 2**  
**Estado:** estructura inicial del proyecto; todavía no incluye entrenamiento ni código de inferencia.

## Objetivo

Implementar una U-Net **propia en PyTorch** para segmentación de agua a partir de imágenes SAR Sentinel-1 con bandas VV y VH. Comparar etiquetas doradas y débiles y evaluar la generalización en Lima (marzo de 2017) con referencia Copernicus EMSN038, o justificar Bolivia como respaldo si Lima no es viable.

## Estructura mínima

```text
sentinel1-flood-unet/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── experimentos.md
├── notebooks/
│   └── .gitkeep
├── src/
│   └── .gitkeep
├── data/
│   └── README.md
├── outputs/
│   └── README.md
└── reports/
    └── .gitkeep
```

El archivo `.gitkeep` únicamente conserva una carpeta vacía en Git; puede retirarse cuando se añadan archivos reales.

## Qué va en cada lugar

- `configs/`: registro de decisiones comunes y los cuatro experimentos M1–M4. Más adelante contendrá las configuraciones que leerá el entrenamiento.
- `notebooks/`: notebook de exploración y, si hace falta, notebook de inspección de Lima. En Colab se pueden abrir desde GitHub.
- `src/`: implementación original de dataset, U-Net, entrenamiento, evaluación y herramientas geográficas; no se han creado scripts vacíos innecesarios.
- `data/`: caché y datos de trabajo locales **excluidos de GitHub**; consultar `data/README.md`.
- `outputs/`: checkpoints y resultados pesados **excluidos de GitHub**; consultar `outputs/README.md`.
- `reports/`: informe en formato NeurIPS y figuras pequeñas seleccionadas para el informe.

## Datos externos (no descargar todo de entrada)

- Sen1Floods11: https://github.com/cloudtostreet/Sen1Floods11
- Carpeta del bucket: `gs://sen1floods11/v1.1/`
- Sentinel-1 GRD de Lima: https://search.asf.alaska.edu/ o https://browser.dataspace.copernicus.eu/
- Cartografía EMSN038: https://mapping.emergency.copernicus.eu/activations/EMSN038/

La preparación debe verificar valores `-1` (ignorar), `0` (no agua), `1` (agua), el orden de VV/VH y la alineación geográfica de Lima. No se deben tratar automáticamente las delineaciones de inundación EMS como equivalentes a etiquetas de agua/no agua.

## Uso desde PC y Google Colab

1. Mantener **una sola versión** del código en GitHub.
2. En PC y en Colab obtener la **misma revisión** del repositorio (mismo commit cuando se comparen experimentos).
3. Obtener los datos desde su fuente oficial hacia el almacenamiento de trabajo de cada entorno, sin subirlos a GitHub.
4. Ejecutar cada experimento con su configuración fija; registrar entorno, versión de PyTorch, semilla, particiones y preprocesamiento.
5. Guardar checkpoints y métricas en `outputs/`, con copia persistente en disco local o Drive. Las máquinas virtuales de Colab son temporales.
6. Evaluar todos los modelos con la misma lógica de métricas y las mismas particiones.

> **Pendiente:** el requisito final del examen es ejecutar todo el proyecto con un único comando documentado en este README, y fijar el entorno en `requirements.txt` o `environment.yml`. En esta etapa se entrega solo la estructura; el comando se incorporará cuando exista el pipeline.

## Entregables al completar el desarrollo

- U-Net original con entradas VV y VH.
- Estudio de ablación: etiquetas doradas vs. doradas + débiles, BCE vs. BCE + Dice (o diseño arquitectónico).
- IoU, F1, precisión, recall y matriz de confusión en evaluación interna.
- Prueba geográfica en Lima sin entrenamiento peruano, brecha de rendimiento y al menos dos hipótesis de su causa (o Bolivia con justificación).
- Tres ejemplos visuales de fallos típicos SAR.
- Informe PDF de 6–10 páginas, formato NeurIPS, mínimo cinco referencias.
- Repositorio reproducible, entorno especificado y un solo comando.

**Fecha de entrega indicada en el examen:** 31 de octubre de 2026.
