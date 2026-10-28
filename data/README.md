# Datos de trabajo (local / Colab; no se suben a GitHub)

Esta carpeta contiene por ahora solo esta explicación.

Cuando carguemos los datos, crearemos **únicamente si se necesitan**:

- `raw/`: chips SAR y máscaras descargadas selectivamente, sin modificar.
- `processed/`: chips listos para entrenamiento y manifiestos locales.
- `lima/`: escenas de marzo de 2017 y referencia EMSN038 alineada.

No descargaremos todo Sen1Floods11 desde el comienzo. Los archivos `.tif/.tiff` y los datos de trabajo están excluidos por `.gitignore`.

Los manifiestos pequeños que permitan reproducir las particiones podrán versionarse más adelante fuera de esta carpeta, previa revisión.
