# Experimentos planificados — Pregunta 2

Se utilizará **una U-Net de implementación propia con entrada VV + VH**.

| Experimento | Etiquetas | Función de pérdida | Entorno inicial sugerido |
|---|---|---|---|
| M1 | Doradas | BCE | PC |
| M2 | Doradas | BCE + Dice | PC |
| M3 | Doradas + débiles Sentinel-1 | BCE | Colab |
| M4 | Doradas + débiles Sentinel-1 | BCE + Dice | Colab |

## Valores comunes que se fijarán antes de entrenar

- Particiones de train/val/test y exclusión de Bolivia si se usa como respaldo.
- Orden de canales VV, VH y escala en dB.
- Normalización usando estadísticas calculadas solo con entrenamiento.
- Tamaño de chip, recortes, batch size, número de épocas, semilla y optimizador.
- Tratamiento de `-1` como píxel ignorado en pérdida y métricas.
- Selección del checkpoint por validación y umbral de segmentación.
- Misma evaluación sobre chips dorados y sobre el conjunto Lima/Bolivia.

**Nota:** el experimento opcional VV vs. VH aislados se añadirá solo después de completar los experimentos obligatorios.

Las configuraciones ejecutables `.yaml` se crearán cuando implementemos el entrenamiento; no se anticipan archivos innecesarios.
