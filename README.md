# Ciencia De Datos Con Python

<p align="center">
  <img src="assets/readme/readme_banner.png" alt="Banner portfolio" width="100%" />
</p>

Portafolio tecnico para reclutadores y equipos de analitica: proyectos de machine learning, estadistica aplicada y storytelling con datos.

## Vista Rapida

<p align="center">
  <img src="assets/readme/model_snapshot.png" alt="Model snapshot" width="95%" />
</p>
<p align="center">
  <img src="assets/readme/miniproyecto_top5_trend.png" alt="MiniProyecto trend" width="95%" />
</p>

## Lo Que Demuestro

- construccion de pipelines de datos reproducibles
- comparacion de modelos con metricas correctas
- analisis estadistico con criterio de supuestos
- comunicacion ejecutiva orientada a decision

## Demos Principales

- Clasificacion Iris: `python scripts/project_iris_ai.py`
- Clasificacion Titanic/OpenML con fallback: `python scripts/project_titanic_openml_ai.py --offline`
- Regresion Wine Quality/UCI con fallback: `python scripts/project_wine_quality_ai.py --save-raw --offline`
- EDA MiniProyecto: `python scripts/analisis_datos.py --input MiniProyecto/coffee_db.parquet`

## Notebooks Showcase (Muy Vendedor)

- [notebooks/showcase/01_showcase_ejecutivo.ipynb](notebooks/showcase/01_showcase_ejecutivo.ipynb)
- [notebooks/showcase/02_miniproyecto_storytelling.ipynb](notebooks/showcase/02_miniproyecto_storytelling.ipynb)

Estos notebooks estan preparados para entrevista: contexto, grafica, lectura ejecutiva y cierre accionable.

## Comandos Make

- `make install`
- `make notebook`
- `make run-eda`
- `make churn-demo`
- `make iris-ai`
- `make titanic-ai`
- `make wine-ai`
- `make datasets-demo`
- `make visual-assets`

## Conexion De Datasets

Script: `scripts/dataset_connector.py`

- `sklearn`: datasets locales (iris, wine, breast_cancer)
- `openml`: titanic, adult, house_prices
- `url`: wine_quality_red, wine_quality_white
- `kaggle`: descarga por slug con CLI oficial

Ejemplos:

```bash
python scripts/dataset_connector.py --source sklearn --name iris
python scripts/dataset_connector.py --source openml --name titanic --fallback-sklearn iris --offline
python scripts/dataset_connector.py --source url --name wine_quality_red --fallback-sklearn wine --offline
python scripts/dataset_connector.py --source kaggle --kaggle-dataset yasserh/titanic-dataset --output-dir data/raw --unzip
```

## Estructura

```text
data/                      # datos por etapa
models/                    # artefactos de modelos
notebooks/                 # notebooks tecnicos + showcase
reports/figures/           # imagenes y resultados
scripts/                   # pipelines ejecutables
MiniProyecto/              # dataset y material complementario
```

## Referencias

- scikit-learn datasets: https://scikit-learn.org/stable/datasets.html
- OpenML Titanic: https://api.openml.org/d/40945
- UCI Wine Quality: https://archive.ics.uci.edu/ml/datasets/wine_quality
- Kaggle API: https://github.com/Kaggle/kaggle-api
