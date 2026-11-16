#%% Taller: Estadística descriptiva y análisis exploratorio de datos (EDA)
# Curso: Estadística III — Ingeniería de Sistemas
# Docente: Daniel Betancur Trujillo
# Universidad Católica de Oriente
#
# Integrante(s): ______________________
#
# NOTA METODOLÓGICA IMPORTANTE:
# El dataset "songs_normalize.csv" entregado para este taller NO incluye la
# columna `explicit` que describe la guía original (lenguaje explícito).
# Se verificó el archivo y sus columnas reales son:
# artist, song, duration_ms, genre, year, popularity, danceability, energy,
# key, loudness, mode, speechiness, acousticness, instrumentalness,
# liveness, valence, tempo  (17 columnas, sin `explicit`).
# En consecuencia:
#   - Ejercicio 5: se calculan los 5 géneros más frecuentes; se omite el
#     porcentaje de canciones explícitas por no existir esa variable.
#   - Ejercicio 7: se reemplaza la comparación "explícita vs no explícita"
#     por una comparación válida con otra variable categórica disponible:
#     `mode` (0 = modo menor, 1 = modo mayor), que sí existe en los datos.

#%% Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# ============================================================
# PARTE 1 — INGESTA Y LIMPIEZA INICIAL
# ============================================================

#%% Ejercicio 1: Cargar el dataset y mostrar las primeras 5 filas
df = pd.read_csv('songs_normalize.csv')
print(df.head())

#%% Ejercicio 2: Resumen del DataFrame (filas, columnas, tipos, nulos)
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")
print("Tipos de dato por columna:")
print(df.dtypes)
print("\nResumen general (info):")
df.info()
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nFilas duplicadas (registros exactamente repetidos):", df.duplicated().sum())

# Limpieza: se eliminan las filas exactamente duplicadas (mismo artista,
# canción, año y todos los atributos de audio). Se conserva la primera
# aparición de cada una.
filas_antes = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
filas_despues = df.shape[0]
print(f"\nFilas antes de eliminar duplicados: {filas_antes}")
print(f"Filas después de eliminar duplicados: {filas_despues}")
print(f"Filas eliminadas: {filas_antes - filas_despues}")

# ANÁLISIS EJERCICIO 2:
# El conjunto llega con 2000 filas y 17 columnas, sin valores nulos en
# ninguna columna, por lo que no se requiere imputación de datos
# faltantes. Sin embargo, sí requiere limpieza antes de continuar:
# (1) tiene 59 filas exactamente duplicadas (mismo artista, canción, año
# y todos los atributos de audio repetidos) — por ejemplo "Love Me
# Harder" de Ariana Grande o "Clocks" de Coldplay aparecen dos veces con
# valores idénticos — que se eliminan para no inflar artificialmente
# conteos y estadísticos, dejando 1941 filas útiles; (2) `duration_ms`
# está en milisegundos y conviene expresarla en minutos para que sea
# interpretable (ejercicio 3); (3) `key` y `mode` son enteros que en
# realidad codifican categorías (tonalidad y modo musical), no
# cantidades, así que no deben tratarse con estadísticos de variables
# continuas (media, desviación) sino como categóricas; y (4) `genre` es
# una variable de texto multivaluada (una celda puede traer varios
# géneros separados por "; "), lo que exige una decisión explícita de
# tratamiento antes de contar frecuencias (ver ejercicio 5). Además, como
# se documentó arriba, el archivo entregado no trae la columna `explicit`
# que pedía la guía original, lo que obliga a adaptar los ejercicios 5 y 7.
# De aquí en adelante todos los ejercicios se calculan sobre el DataFrame
# `df` ya sin duplicados.

# ============================================================
# PARTE 2 — ESTADÍSTICA DESCRIPTIVA UNIVARIADA
# ============================================================

#%% Ejercicio 3: duration_ms -> duration_min
df['duration_min'] = df['duration_ms'] / 60000
print(df[['duration_ms', 'duration_min']].head())

#%% Ejercicio 4: medidas descriptivas de duration_min, popularity y danceability
resumen = df[['duration_min', 'popularity', 'danceability']].agg(
    ['mean', 'median', 'std', 'min', 'max']
)
print(resumen)

# ANÁLISIS EJERCICIO 4:
# La media de `popularity` es 59.6 y la mediana es 65.0: la mediana supera
# claramente a la media, lo que indica una distribución asimétrica hacia
# la izquierda (cola hacia valores bajos). Esto ocurre porque hay un grupo
# de canciones con popularidad muy baja (mínimo 0) que "jalan" la media
# hacia abajo, mientras que la mayoría del catálogo —canciones que sí
# llegaron a las listas globales— se concentra en valores altos, cercanos
# a 65-73. Para describir la popularidad "típica" de una canción de este
# conjunto, la mediana (65.0) es más representativa que la media, porque
# no se deja arrastrar por esa minoría de valores extremos y refleja mejor
# dónde se ubica realmente la mayoría de las canciones exitosas.

#%% Ejercicio 5: 5 géneros más frecuentes y % de canciones explícitas
# La columna `genre` puede traer varios géneros en una misma celda,
# separados por "; " (ej. "hip hop; pop; R&B"). Hay dos formas válidas de
# contar frecuencias:
#   a) Tratar cada combinación completa como una única categoría.
#   b) Separar ("explotar") cada celda en sus géneros individuales y
#      contar cada uno por separado.
# Se opta aquí por la opción (b): separar en géneros individuales. La
# razón es que la pregunta de negocio típica ("¿qué géneros dominan el
# catálogo?") se responde mejor así, porque la opción (a) diluye el
# conteo en decenas de combinaciones casi únicas (una canción "pop; rock"
# y otra "rock; pop" quedarían en categorías distintas de tamaño pequeño)
# y no deja ver que, por ejemplo, el pop aparece en la inmensa mayoría de
# las canciones sin importar con qué otro género se combine. Se muestra
# también el resultado de la opción (a) como referencia.

# Opción (a): combinación completa como categoría única
top5_combinado = df['genre'].value_counts().head(5)
print("Top 5 - género como combinación completa:")
print(top5_combinado)

# Opción (b): géneros individuales (criterio elegido)
generos_individuales = df['genre'].str.split(';').explode().str.strip()
top5_individual = generos_individuales.value_counts().head(5)
print("\nTop 5 - géneros individuales (criterio elegido):")
print(top5_individual)

# Porcentaje de canciones explícitas: NO CALCULABLE.
# El archivo de datos entregado no contiene la columna `explicit` descrita
# en la guía (se verificó explícitamente la lista de columnas del CSV).
# Este punto queda documentado como limitación de los datos disponibles,
# no omitido por descuido.

# ANÁLISIS EJERCICIO 5:
# Separando por género individual, el pop domina de forma abrumadora: está
# presente en 1582 de las 1941 canciones (más del 80% del catálogo),
# seguido de lejos por hip hop (751), R&B (437), Dance/Electronic (380) y
# rock (225). Esto confirma que el pop no es solo "un género más" sino el
# eje sobre el cual gira casi toda la música que llega a las listas
# globales de éxitos entre 2000 y 2019; los demás géneros aparecen casi
# siempre combinados con él (fusiones como "hip hop; pop" o "pop; R&B"),
# más que como categorías puras independientes.

# ============================================================
# PARTE 3 — VISUALIZACIÓN DE DATOS
# ============================================================

#%% Ejercicio 6: histograma de tempo con línea de la media
media_tempo = df['tempo'].mean()

plt.figure(figsize=(9, 5))
plt.hist(df['tempo'], bins=30, color='#4C72B0', edgecolor='white')
plt.axvline(media_tempo, color='red', linewidth=2,
            label=f'Media = {media_tempo:.1f} BPM')
plt.title('Distribución del tempo de las canciones (2000-2019)')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Número de canciones')
plt.legend()
plt.tight_layout()
plt.savefig('fig_06_histograma_tempo.png', dpi=150)
plt.show()

# ANÁLISIS EJERCICIO 6:
# La media del tempo es de aproximadamente 120 BPM, pero el histograma
# muestra que el repertorio no se concentra en un único pico alrededor de
# ese valor: se observan dos zonas de mayor densidad, una entre 90 y 105
# BPM y otra entre 115 y 130 BPM, con una ligera caída entre ambas. Es
# decir, la distribución es más bien bimodal que unimodal, aunque los dos
# picos están cerca uno del otro y la media cae precisamente en la "zona
# de transición" entre ellos. Una explicación musical razonable es que el
# primer grupo corresponde a ritmos típicos de hip hop y R&B de tempo
# medio (alrededor de 90-100 BPM), mientras que el segundo agrupa canciones
# de pop y dance de tempo más marcado (120-128 BPM, rango clásico del
# "four-on-the-floor" bailable), dos estilos que dominan el catálogo según
# el ejercicio 5 y que imponen cada uno su propio rango de tempo preferido.

#%% Ejercicio 7: boxplot de popularity según explicit (no disponible) -> según mode
# Como el dataset no tiene `explicit`, se usa la variable categórica
# disponible más cercana en espíritu (categórica binaria, codificada como
# entero): `mode` (0 = modo menor, 1 = modo mayor).
plt.figure(figsize=(7, 5))
sns.boxplot(x='mode', y='popularity', data=df, hue='mode',
            palette=['#DD8452', '#4C72B0'], legend=False)
plt.title('Popularidad según el modo musical (0=menor, 1=mayor)')
plt.xlabel('Modo (0 = menor, 1 = mayor)')
plt.ylabel('Popularidad (0-100)')
plt.tight_layout()
plt.savefig('fig_07_boxplot_popularity_mode.png', dpi=150)
plt.show()

print(df['mode'].value_counts())
print(df.groupby('mode')['popularity'].describe())

# ANÁLISIS EJERCICIO 7:
# (Nota: se compara `popularity` según `mode` en lugar de `explicit`,
# variable no disponible en los datos entregados; ver nota metodológica
# al inicio del script.)
# Las medianas de popularidad son idénticas entre modo menor y modo mayor
# (65 en ambos grupos), y las cajas se solapan casi por completo, con
# dispersiones (rango intercuartílico) muy similares en ambos grupos. Es
# decir, no hay una diferencia visual apreciable en popularidad entre
# canciones en modo mayor y en modo menor: el modo musical, por sí solo,
# no distingue a las canciones más o menos exitosas de este catálogo. Los
# tamaños de los dos grupos son razonablemente comparables (867 canciones
# en modo menor frente a 1074 en modo mayor), por lo que la comparación no
# está sesgada por un grupo demasiado pequeño frente al otro.

# ============================================================
# PARTE 4 — ANÁLISIS BIVARIADO Y CORRELACIÓN
# ============================================================

#%% Ejercicio 8: matriz de correlación de Pearson
variables_audio = ['danceability', 'energy', 'valence', 'loudness']
corr = df[variables_audio].corr(method='pearson')
print(corr)

#%% Ejercicio 9: mapa de calor de la matriz de correlación
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            center=0, square=True, linewidths=0.5, cbar_kws={'label': 'Correlación de Pearson'})
plt.title('Correlación entre atributos de audio')
plt.tight_layout()
plt.savefig('fig_09_heatmap_correlacion.png', dpi=150)
plt.show()

#%% Ejercicio 10: par más fuertemente correlacionado -> dispersión
corr_sin_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
par_max = corr_sin_diagonal.abs().stack().idxmax()
valor_corr = corr.loc[par_max]
print(f"Par con correlación más fuerte: {par_max} -> r = {valor_corr:.3f}")

var_x, var_y = par_max
plt.figure(figsize=(7, 6))
plt.scatter(df[var_x], df[var_y], alpha=0.35, s=18, color='#4C72B0', edgecolor='none')
plt.title(f'Relación entre {var_x} y {var_y} (r = {valor_corr:.2f})')
plt.xlabel(var_x.capitalize() + (' (dB)' if var_x == 'loudness' else ' (0-1)'))
plt.ylabel(var_y.capitalize() + (' (dB)' if var_y == 'loudness' else ' (0-1)'))
plt.tight_layout()
plt.savefig('fig_10_dispersión_energy_loudness.png', dpi=150)
plt.show()

# ANÁLISIS EJERCICIO 10:
# El par con la correlación más fuerte en valor absoluto es energy-loudness
# (r ≈ 0.65): una correlación positiva y de fuerza moderada-alta. La nube
# de puntos muestra una tendencia claramente ascendente y razonablemente
# lineal —a mayor energía percibida, mayor volumen promedio de la
# canción—, aunque con dispersión considerable alrededor de la tendencia,
# especialmente en el rango de energía media (0.6-0.8), donde el volumen
# varía bastante de una canción a otra. No se observa curvatura marcada
# que sugiera una relación no lineal, ni un grupo aislado de valores
# atípicos evidentes que por sí solo esté inflando artificialmente el
# coeficiente; la relación se sostiene a lo largo de casi todo el rango de
# energía. Dicho esto, esta correlación describe una asociación estadística
# entre dos atributos de audio calculados sobre las mismas canciones, y
# NO permite concluir que una propiedad cause la otra (por ejemplo, no
# implica que subir el volumen de una grabación la haga sonar más
# "enérgica" en sentido musical, ni que ambas no estén siendo influidas
# por decisiones de producción o mezcla comunes a ambas variables).

# ============================================================
# PUNTO OPCIONAL (bonificación hasta 0.5)
# ============================================================

#%% Pregunta propia: ¿cómo ha evolucionado la duración promedio de los
# éxitos musicales a lo largo de los años (2000-2019)?
duracion_por_anio = df[(df['year'] >= 2000) & (df['year'] <= 2019)] \
    .groupby('year')['duration_min'].mean()

plt.figure(figsize=(9, 5))
plt.plot(duracion_por_anio.index, duracion_por_anio.values,
         marker='o', color='#4C72B0')
plt.title('Duración promedio de las canciones exitosas por año (2000-2019)')
plt.xlabel('Año')
plt.ylabel('Duración promedio (minutos)')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('fig_bonus_duracion_por_anio.png', dpi=150)
plt.show()

# ANÁLISIS PUNTO OPCIONAL:
# La duración promedio de las canciones exitosas muestra una tendencia
# decreciente sostenida a lo largo de las dos décadas analizadas: pasa de
# rondar los 4.2 minutos a comienzos de los 2000 a cerca de 3.3 minutos en
# 2019, es decir, una reducción de casi un minuto en promedio. La caída no
# es perfectamente monótona año a año, pero la tendencia general es clara
# y consistente con un fenómeno ampliamente documentado en la industria:
# el auge del streaming (donde una canción compite por retener la atención
# del oyente en los primeros segundos, y donde los ingresos dependen de la
# reproducción completa) ha empujado a los artistas a producir canciones
# más cortas y directas que las radiales o físicas de los años 2000.

#%% Fin del taller
print("Taller ejecutado completamente sin errores.")
