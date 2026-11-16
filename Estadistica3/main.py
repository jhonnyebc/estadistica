import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Paso 1. Importar los datos a mi area de trabajo
# =========================================================
estudiantes = pd.read_csv("04 - estudiantes.csv")

# Vistazo inicial
print(estudiantes.columns)
print(estudiantes.head())
print(estudiantes.info())
print(estudiantes.shape)
print(estudiantes.describe())

#tablas de frcuencia
estratos = pd.DataFrame()
estratos['F absoluta']=estudiantes.Estrato.value_counts().sort_index()
estratos['f relativa']= estudiantes.Estrato.value_counts(normalize=True)*100
estratos.reset_index(inplace=True)