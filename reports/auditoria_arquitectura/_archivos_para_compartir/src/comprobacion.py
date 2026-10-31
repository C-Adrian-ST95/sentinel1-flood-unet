
@'
import pyogrio
from pathlib import Path

# Ruta de la geodatabase de Lima 2017
ruta = Path(
    "data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb"
)

# Capas que necesitamos inspeccionar
capas = [
    "Potential_Maximum_flood",
    "Riverbank",
    "EMSN038_AOI",
    "Footprints",
    "grid200",
]

for nombre in capas:
    print(f"\n{'=' * 55}")
    print(f"CAPA: {nombre}")
    print("=" * 55)

    info = pyogrio.read_info(ruta, layer=nombre)

    print("Geometría:", info["geometry_type"])
    print("CRS:", info["crs"])
    print("Número de elementos:", info["features"])
    print("Extensión geográfica:", info["total_bounds"])
    print("Campos:", list(info["fields"]))

    # Mostrar registros de ejemplo de las dos capas principales
    if nombre in ["Potential_Maximum_flood", "Riverbank"]:
        datos = pyogrio.read_dataframe(
            ruta,
            layer=nombre,
            max_features=3
        )

        print("\nPrimeros registros:")
        print(
            datos.drop(columns="geometry")
            .iloc[:, :12]
            .to_string(index=False)
        )

print("\nINSPECCIÓN COMPLETADA")
'@ | python -


@'
from pathlib import Path
import pyogrio
from shapely import union_all

# Ruta original de la geodatabase
ruta = Path(
    "data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb"
)

# Leer las dos capas principales
flood = pyogrio.read_dataframe(
    ruta, layer="Potential_Maximum_flood"
)

river = pyogrio.read_dataframe(
    ruta, layer="Riverbank"
)

# 1. Analizar los atributos de cada capa
for nombre, datos in [
    ("Potential_Maximum_flood", flood),
    ("Riverbank", river)
]:
    print(f"\n{'='*55}")
    print(f"CAPA: {nombre}")
    print("="*55)

    print("\nCantidad de polígonos:", len(datos))
    print("\nValores del campo source:")
    print(datos["source"].fillna("SIN VALOR").value_counts())

    print("\nÁrea de cada polígono (km²):")
    for i, geometria in enumerate(datos.geometry):
        print(f"  Polígono {i+1}: {geometria.area/1e6:.4f}")

    print("\nGeometrías inválidas:",
          (~datos.geometry.is_valid).sum())

# 2. Comparar la superposición de las capas
if flood.geometry.is_valid.all() and river.geometry.is_valid.all():

    area_flood = union_all(flood.geometry.values)
    area_river = union_all(river.geometry.values)

    interseccion = area_flood.intersection(area_river)

    print("\n" + "="*55)
    print("COMPARACIÓN ESPACIAL")
    print("="*55)

    print(f"Área única Flood: {area_flood.area/1e6:.4f} km²")
    print(f"Área única Riverbank: {area_river.area/1e6:.4f} km²")
    print(f"Superposición: {interseccion.area/1e6:.4f} km²")

else:
    print("\nHay geometrías inválidas que debemos revisar.")

# 3. Inspeccionar información de coberturas
footprints = pyogrio.read_dataframe(
    ruta, layer="Footprints"
)

print("\n" + "="*55)
print("FOOTPRINTS")
print("="*55)
print(footprints.drop(columns="geometry").to_string(index=False))

print("\nINSPECCIÓN FINALIZADA")
'@ | python -



@'
from pathlib import Path
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely import union_all

# 1. Rutas
ruta = Path(
    "data/raw/lima_2017/emsn038/EMSN038_UTM18S_v0.gdb"
)

salida = Path("outputs/figures")
salida.mkdir(parents=True, exist_ok=True)

# 2. Leer las capas
flood = pyogrio.read_dataframe(
    ruta, layer="Potential_Maximum_flood"
)
riverbank = pyogrio.read_dataframe(
    ruta, layer="Riverbank"
)
aoi = pyogrio.read_dataframe(
    ruta, layer="EMSN038_AOI"
)
footprints = pyogrio.read_dataframe(
    ruta, layer="Footprints"
)

# 3. Separar las categorías de inundación y río
categoria = flood["source"].fillna("").str.strip().str.lower()

inundacion = flood[categoria == "maximun height"].copy()
rios = flood[categoria == "river"].copy()

# 4. Seleccionar únicamente la cobertura WorldView-2
wv = footprints[
    footprints["SDP"] == "WorldView-2"
].copy()

# 5. Comprobar duplicidad de geometrías
geom_inundacion = union_all(inundacion.geometry.values)
geom_riverbank = union_all(riverbank.geometry.values)
geom_aoi = union_all(aoi.geometry.values)
geom_wv = union_all(wv.geometry.values)

diferencia = geom_inundacion.symmetric_difference(
    geom_riverbank
).area

# 6. Calcular cobertura óptica válida dentro del AOI
cobertura_valida = geom_wv.intersection(geom_aoi)

area_aoi = geom_aoi.area / 1e6
area_cobertura = cobertura_valida.area / 1e6

inundacion_cubierta = geom_inundacion.intersection(
    cobertura_valida
).area / 1e6

print("\n========== RESULTADOS ==========")
print(f"Diferencia geométrica: {diferencia:.4f} m²")
print(f"Área AOI: {area_aoi:.4f} km²")
print(f"Cobertura WorldView-2 dentro del AOI: {area_cobertura:.4f} km²")
print(f"Porcentaje de cobertura: {100*area_cobertura/area_aoi:.2f}%")
print(f"Extensión potencial cubierta por WorldView-2: {inundacion_cubierta:.4f} km²")

if diferencia < 0.01:
    print("CONFIRMADO: Riverbank y maximun height tienen la misma geometría.")
else:
    print("Las capas presentan diferencias geométricas.")

# 7. Generar un mapa general y un acercamiento
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax in axes:
    aoi.boundary.plot(ax=ax, color="gray", linewidth=1)
    wv.plot(
        ax=ax, color="gold", alpha=0.15,
        edgecolor="darkorange", linewidth=0.6
    )
    rios.plot(ax=ax, color="royalblue", alpha=0.7)
    inundacion.plot(ax=ax, color="crimson", alpha=0.65)

    ax.set_aspect("equal")
    ax.set_xlabel("Este UTM (m)")
    ax.set_ylabel("Norte UTM (m)")
    ax.grid(alpha=0.2)

axes[0].set_title("EMSN038 - Área general de estudio")
axes[1].set_title("EMSN038 - Zonas de inundación")

# Acercamiento a las zonas de inundación
xmin, ymin, xmax, ymax = flood.total_bounds
margen = 1500

axes[1].set_xlim(xmin - margen, xmax + margen)
axes[1].set_ylim(ymin - margen, ymax + margen)

leyenda = [
    Line2D([0], [0], color="gray", label="Límite AOI"),
    Patch(facecolor="gold", alpha=0.3, label="WorldView-2"),
    Patch(facecolor="royalblue", label="Ríos"),
    Patch(facecolor="crimson", label="Extensión potencial")
]

fig.legend(handles=leyenda, loc="lower center", ncol=4)
plt.tight_layout(rect=[0, 0.07, 1, 1])

archivo = salida / "emsn038_revision_espacial.png"
plt.savefig(archivo, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nMapa guardado en: {archivo}")
print("PROCESO FINALIZADO")
'@ | python -
