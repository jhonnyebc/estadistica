# Google Trends: interés público en energías renovables

**Jhonny Benitez Caro** · Electiva 2 y 3

Caso de estudio 2 de *Python for Research: Data Processing Case Studies*.
Analiza el interés de búsqueda mundial de "solar energy", "wind energy" y "hydropower" en los últimos 10 años.

## Estructura
| Archivo | Contenido |
|---|---|
| `Google_Trends_Renewables.ipynb` | Notebook principal (limpieza, tendencias, estacionalidad, anomalías, regiones, conclusiones) |
| `download_data.py` | Descarga los datos con `pytrends` y los guarda en `data/` |
| `data/` | `interest_over_time.csv` (mensual) e `interest_by_region.csv` (por país) |
| `figures/` | Gráficos exportados en PNG |
| `requirements.txt` | Dependencias |

## Ejecución en PyCharm
1. Abrir la carpeta como proyecto y usar el intérprete `.venv` (Settings → Python Interpreter → Add → Existing → `.venv\Scripts\python.exe`).
2. Abrir `Google_Trends_Renewables.ipynb` y ejecutar *Run All*.
3. Opcional: `python download_data.py` para volver a descargar (Google puede limitar las peticiones con HTTP 429; los CSV incluidos permiten ejecutar sin conexión).

Nota: `pytrends` 4.9.2 es incompatible con `urllib3>=2` si se usan reintentos internos; por eso `download_data.py` usa `retries=0` y gestiona los reintentos por su cuenta.
