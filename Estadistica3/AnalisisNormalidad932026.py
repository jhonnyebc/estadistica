import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sns
from fontTools.t1Lib import PD_values
from scipy import stats


# ==========================================================
# 1. LEER EL ARCHIVO
# ==========================================================

df = pd.read_csv("Datos.csv", header=None)


# ==========================================================
# 2. SEPARAR MANUALMENTE LOS DATOS POR COMAS
# ==========================================================

df = df[0].str.split(",", expand=True)

df.columns = ["height", "weight", "age", "male"]


# ==========================================================
# 3. ELIMINAR LA PRIMERA FILA
# ==========================================================

df = df.iloc[1:].copy()


# ==========================================================
# 4. CONVERTIR DATOS A NUMÉRICOS
# ==========================================================

df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["male"] = pd.to_numeric(df["male"], errors="coerce")


# ==========================================================
# 5. CREAR VARIABLE PESO
# Mujeres mayores de 15 años
#
# male = 0 -> mujer
# male = 1 -> hombre
# ==========================================================

peso = df.loc[
    (df["age"] > 15) & (df["male"] == 0),
    "weight"
].dropna()


# ==========================================================
# 6. INFORMACIÓN GENERAL
# ==========================================================

print("\n==========================================")
print("DATOS ANALIZADOS")
print("==========================================")

print("\nPesos de mujeres mayores de 15 años:")
print(peso)

print("\nCantidad de datos:")
print(len(peso))


# ==========================================================
# 7. MEDIA Y DESVIACIÓN ESTÁNDAR
# ==========================================================

mu, sigma = stats.norm.fit(peso)

print("\n==========================================")
print("ESTADÍSTICOS DESCRIPTIVOS")
print("==========================================")

print(f"\nMedia: {mu:.4f}")
print(f"Desviación estándar: {sigma:.4f}")


# ==========================================================
# 8. ASIMETRÍA
# ==========================================================

asimetria = stats.skew(peso)

print(f"Asimetría: {asimetria:.4f}")

if asimetria > 0:
    print("Interpretación: asimetría positiva.")
elif asimetria < 0:
    print("Interpretación: asimetría negativa.")
else:
    print("Interpretación: distribución simétrica.")


# ==========================================================
# 9. CURTOSIS
# ==========================================================

curtosis = stats.kurtosis(peso)

print(f"Curtosis: {curtosis:.4f}")

if curtosis > 0:
    print("Interpretación: distribución leptocúrtica.")
elif curtosis < 0:
    print("Interpretación: distribución platicúrtica.")
else:
    print("Interpretación: distribución mesocúrtica.")


# ==========================================================
# 10. PRUEBAS DE HIPÓTESIS PARA NORMALIDAD
# ==========================================================

alpha = 0.05

print("\n==========================================")
print("PRUEBAS DE HIPÓTESIS DE NORMALIDAD")
print("==========================================")

print("\nHipótesis:")
print("H0: Los datos siguen una distribución normal.")
print("H1: Los datos NO siguen una distribución normal.")
print(f"Nivel de significancia: alpha = {alpha}")


# ==========================================================
# 10.1 SHAPIRO-WILK
# ==========================================================

estadistico_shapiro, p_shapiro = stats.shapiro(peso)

print("\n------------------------------------------")
print("PRUEBA SHAPIRO-WILK")
print("------------------------------------------")

print(f"Estadístico W: {estadistico_shapiro:.6f}")
print(f"p-valor: {p_shapiro:.6f}")

if p_shapiro > alpha:
    print("Decisión: No se rechaza H0.")
    print("Conclusión: Los datos son compatibles con una distribución normal.")
else:
    print("Decisión: Se rechaza H0.")
    print("Conclusión: Los datos NO siguen una distribución normal.")


# ==========================================================
# 10.2 D'AGOSTINO-PEARSON
# ==========================================================

estadistico_dagostino, p_dagostino = stats.normaltest(peso)

print("\n------------------------------------------")
print("PRUEBA D'AGOSTINO-PEARSON")
print("------------------------------------------")

print(f"Estadístico: {estadistico_dagostino:.6f}")
print(f"p-valor: {p_dagostino:.6f}")

if p_dagostino > alpha:
    print("Decisión: No se rechaza H0.")
    print("Conclusión: Los datos son compatibles con una distribución normal.")
else:
    print("Decisión: Se rechaza H0.")
    print("Conclusión: Los datos NO siguen una distribución normal.")


# ==========================================================
# 10.3 JARQUE-BERA
# ==========================================================

estadistico_jb, p_jb = stats.jarque_bera(peso)

print("\n------------------------------------------")
print("PRUEBA JARQUE-BERA")
print("------------------------------------------")

print(f"Estadístico JB: {estadistico_jb:.6f}")
print(f"p-valor: {p_jb:.6f}")

if p_jb > alpha:
    print("Decisión: No se rechaza H0.")
    print("Conclusión: Los datos son compatibles con una distribución normal.")
else:
    print("Decisión: Se rechaza H0.")
    print("Conclusión: Los datos NO siguen una distribución normal.")


# ==========================================================
# 10.4 ANDERSON-DARLING
# ==========================================================

resultado_anderson = stats.anderson(peso, dist="norm")

print("\n------------------------------------------")
print("PRUEBA ANDERSON-DARLING")
print("------------------------------------------")

print(f"Estadístico A²: {resultado_anderson.statistic:.6f}")

print("\nValores críticos:")

for nivel, valor_critico in zip(
        resultado_anderson.significance_level,
        resultado_anderson.critical_values
):
    print(
        f"Nivel de significancia {nivel}% "
        f"-> Valor crítico = {valor_critico:.4f}"
    )


# Buscar el valor crítico correspondiente al 5 %
indice_5 = np.where(
    resultado_anderson.significance_level == 5
)[0][0]

valor_critico_5 = resultado_anderson.critical_values[indice_5]

print(f"\nValor crítico para alpha = 0.05: {valor_critico_5:.4f}")

if resultado_anderson.statistic < valor_critico_5:
    print("Decisión: No se rechaza H0.")
    print("Conclusión: Los datos son compatibles con una distribución normal.")
else:
    print("Decisión: Se rechaza H0.")
    print("Conclusión: Los datos NO siguen una distribución normal.")


# ==========================================================
# 11. RESUMEN DE LAS PRUEBAS
# ==========================================================

print("\n==========================================")
print("RESUMEN")
print("==========================================")

print(f"Shapiro-Wilk p-valor:       {p_shapiro:.6f}")
print(f"D'Agostino p-valor:         {p_dagostino:.6f}")
print(f"Jarque-Bera p-valor:        {p_jb:.6f}")

print("\nRegla de decisión:")
print("Si p-valor < 0.05  -> Se rechaza H0")
print("Si p-valor >= 0.05 -> No se rechaza H0")


# ==========================================================
# 12. CURVA NORMAL TEÓRICA
# ==========================================================

x_hat = np.linspace(
    peso.min(),
    peso.max(),
    100
)

y_hat = stats.norm.pdf(
    x_hat,
    mu,
    sigma
)


# ==========================================================
# 13. HISTOGRAMA
# ==========================================================

fig, ax = plt.subplots(figsize=(16, 9))


# Curva normal teórica
ax.plot(
    x_hat,
    y_hat,
    linewidth=3,
    label="Normal teórica"
)


# Histograma
ax.hist(
    peso,
    density=True,
    bins=int(math.sqrt(len(peso))),
    alpha=0.5,
    label="Pesos"
)


# ==========================================================
# 14. CONFIGURACIÓN DEL GRÁFICO
# ==========================================================

plt.title(
    "Histograma peso mujeres > 15\n"
    f"Media = {mu:.2f} | "
    f"Desv. Est. = {sigma:.2f} | "
    f"Asimetría = {asimetria:.3f} | "
    f"Curtosis = {curtosis:.3f}"
)

plt.xlabel("Peso (kg)")
plt.ylabel("Densidad de probabilidad")
plt.legend()

plt.show()
# ==========================================================
# PRUEBA DE HIPÓTESIS SHAPIRO-WILK
# ==========================================================

alpha = 0.05

estadistico_shapiro, p_valor_shapiro = stats.shapiro(peso)

print("\n==========================================")
print("PRUEBA DE NORMALIDAD SHAPIRO-WILK")
print("==========================================")

print("\nHipótesis:")
print("H0: Los pesos siguen una distribución normal.")
print("H1: Los pesos NO siguen una distribución normal.")

print(f"\nNivel de significancia: {alpha}")
print(f"Estadístico W: {estadistico_shapiro:.6f}")
print(f"p-valor: {p_valor_shapiro:.6f}")

# Decisión
if p_valor_shapiro < alpha:
    print("\nDecisión: Se rechaza H0.")
    print("Conclusión: Los datos NO siguen una distribución normal.")
else:
    print("\nDecisión: No se rechaza H0.")
    print("Conclusión: No hay evidencia suficiente para afirmar que los datos no sean normales.")



#D agostino k-Squared
k2, PD_values = stats.normaltest(peso)

# ==========================================================
# PRUEBA DE HOMOCEDASTICIDAD
# LEVENE
# ==========================================================

# Crear grupos
peso_mujeres = df.loc[
    (df["age"] > 15) & (df["male"] == 0),
    "weight"
].dropna()

peso_hombres = df.loc[
    (df["age"] > 15) & (df["male"] == 1),
    "weight"
].dropna()


# Nivel de significancia
alpha = 0.05


# ----------------------------------------------------------
# Mostrar información de los grupos
# ----------------------------------------------------------

print("\n==========================================")
print("PRUEBA DE HOMOCEDASTICIDAD")
print("==========================================")

print("\nGrupos analizados:")
print("Grupo 1: Mujeres mayores de 15 años")
print("Grupo 2: Hombres mayores de 15 años")

print(f"\nCantidad mujeres: {len(peso_mujeres)}")
print(f"Cantidad hombres: {len(peso_hombres)}")


# ----------------------------------------------------------
# Varianzas
# ----------------------------------------------------------

varianza_mujeres = np.var(peso_mujeres, ddof=1)
varianza_hombres = np.var(peso_hombres, ddof=1)

print(f"\nVarianza mujeres: {varianza_mujeres:.4f}")
print(f"Varianza hombres: {varianza_hombres:.4f}")


# ----------------------------------------------------------
# PRUEBA DE LEVENE
# ----------------------------------------------------------

estadistico_levene, p_valor_levene = stats.levene(
    peso_mujeres,
    peso_hombres,
    center="median"
)

print("\n------------------------------------------")
print("PRUEBA DE LEVENE")
print("------------------------------------------")

print("\nHipótesis:")
print("H0: Las varianzas de los grupos son iguales.")
print("H1: Las varianzas de los grupos son diferentes.")

print(f"\nNivel de significancia: {alpha}")
print(f"Estadístico de Levene: {estadistico_levene:.6f}")
print(f"p-valor: {p_valor_levene:.6f}")


# ----------------------------------------------------------
# DECISIÓN
# ----------------------------------------------------------

if p_valor_levene >= alpha:

    print("\nDecisión: No se rechaza H0.")

    print(
        "Conclusión: No existe evidencia estadística suficiente "
        "para afirmar que las varianzas sean diferentes."
    )

    print("Se cumple el supuesto de HOMOCEDASTICIDAD.")

else:

    print("\nDecisión: Se rechaza H0.")

    print(
        "Conclusión: Existe evidencia estadística suficiente "
        "para afirmar que las varianzas son diferentes."
    )

    print("NO se cumple el supuesto de homocedasticidad.")
    print("Existe HETEROCEDASTICIDAD.")


# ==========================================================
# METODO GRÁFICO: VIOLIN PLOT
# ==========================================================

# Filtrar personas mayores de 15 años
df_mayores_15 = df[df["age"] > 15].copy()

# Crear una columna con el nombre del sexo
df_mayores_15["sexo"] = df_mayores_15["male"].map({
    0: "Mujer",
    1: "Hombre"
})

# Crear gráfico
plt.figure(figsize=(10, 6))

sns.violinplot(
    data=df_mayores_15,
    x="sexo",
    y="weight",
    inner="box"
)

plt.title("Distribución del peso por sexo - Mayores de 15 años")
plt.xlabel("Sexo")
plt.ylabel("Peso (kg)")
plt.show()