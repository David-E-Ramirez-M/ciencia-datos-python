# Ciencia De Datos Con Python

Portafolio tecnico orientado a reclutadores y equipos de analitica.
Este repositorio muestra habilidades practicas en:

- analisis exploratorio y estadistica aplicada
- machine learning supervisado y no supervisado
- visualizacion y comunicacion de hallazgos
- organizacion reproducible de proyectos de datos

## Perfil Tecnico Que Demuestra Este Repo

- `Python`: `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- `Visualizacion`: `matplotlib`, `seaborn`, `plotly`
- `Entorno`: notebooks + scripts + estructura reproducible
- `Buenas practicas`: `ruff`, `black`, `isort`, entorno virtual

## Proyectos Destacados

Revision rapida por notebook: ver [PORTFOLIO.md](PORTFOLIO.md).

### Machine Learning
- `Random Forest (Iris)`: clasificacion multiclase y evaluacion base.
- `MLP (Digits)`: red neuronal para reconocimiento de digitos.
- `Gradient Boosting (Regresion)`: modelado y comparacion en problemas continuos.

### Ciencia De Datos E IA
- `K-Means Clustering`: segmentacion y analisis de grupos.
- `Decision Tree Classifier`: modelo interpretable para clasificacion.
- `SVM Classification`: separacion de clases con margen maximo.
- `PCA`: reduccion de dimensionalidad y analisis de varianza explicada.

### Estadistica Aplicada
- `Descriptive Statistics`: distribucion y metricas clave.
- `Correlation Analysis`: relaciones entre variables.
- `Hypothesis Testing (t-test)`: contraste de medias.
- `Chi-Square Test`: independencia entre variables categoricas.
- `ANOVA`: comparacion de medias en multiples grupos.
- `Time Series Decomposition`: tendencia, estacionalidad y residuales.

## Ejecucion Rapida

### Opcion recomendada (Makefile)
```bash
make install
make notebook
```

### Opcion manual
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

## Comandos Utiles

- `make install`: crea `.venv` e instala dependencias.
- `make notebook`: registra kernel y abre JupyterLab.
- `make lint`: valida estilo con `ruff`.
- `make format`: aplica `isort` + `black`.
- `make clean`: limpia entorno virtual y caches.
- `make run-eda`: ejecuta EDA reproducible sobre `NTT/coffee_db.parquet`.
- `make normality-demo`: ejecuta diagnostico de normalidad con salida en `reports/`.
- `make churn-demo`: ejecuta pipeline de clasificacion binaria end-to-end.

## Estructura Del Repositorio

```text
data/                      # Datos por etapa (raw/interim/processed/external)
models/                    # Artefactos de modelos entrenados
notebooks/
  ciencia_datos_ia/        # IA clasica: clustering, PCA, clasificacion
  estadistica/             # Estadistica inferencial y descriptiva
  machine_learning/        # Modelos supervisados
reports/figures/           # Graficos y reportes generados
scripts/                   # Scripts ejecutables de apoyo
NTT/                       # Material complementario del proyecto NTT
requirements.txt
Makefile
README.md
```

## Que Valora Un Reclutador Aqui

- capacidad de ir de datos a conclusion analitica
- conocimiento de fundamentos estadisticos y de ML
- estructura ordenada y facil de reproducir
- codigo legible y mantenible

## Proximas Mejoras Recomendadas

- anadir metricas y resultados clave en cada notebook (`accuracy`, `F1`, `RMSE`)
- exportar graficos finales a `reports/figures/`
- incluir 1 caso end-to-end con problema de negocio y conclusiones ejecutivas

## Roadmap De Portafolio

- ver plan de 10 proyectos estrategicos en [ROADMAP_PORTFOLIO.md](ROADMAP_PORTFOLIO.md)
- ver estandar de entregables en [PROJECT_STANDARDS.md](PROJECT_STANDARDS.md)

## Referencias Tecnicas (Internet)

- SciPy (Shapiro / Anderson / KS): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
- SciPy Anderson-Darling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
- Statsmodels QQ-plot: https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
- scikit-learn ROC-AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- scikit-learn Average Precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- scikit-learn StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- scikit-learn fetch_openml: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html
- TensorFlow text/IMDB: https://www.tensorflow.org/text/guide/word_embeddings
- Docker overview: https://docs.docker.com/get-started/docker-overview/

## Contribuciones

Se aceptan pull requests con mejoras, nuevos notebooks o refactorizaciones.
