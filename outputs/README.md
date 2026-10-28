# Salidas de los experimentos (local / Drive; no se suben a GitHub)

En la etapa de entrenamiento se crearán cuando sean necesarias:

- `checkpoints/`: pesos `best` para predicción y `last` para reanudar, con metadatos de entrenamiento.
- `metrics/`: métricas, curvas y resúmenes de los experimentos.
- `predictions/`: máscaras georreferenciadas y resultados de inferencia.

Los archivos pesados de esta carpeta no se publicarán en GitHub. Conservaremos copias persistentes de los checkpoints de Colab, porque sus máquinas virtuales se eliminan al finalizar la sesión.

Las figuras finales pequeñas elegidas para el informe se copiarán a `reports/` cuando corresponda.
