# Portfolio De Proyectos

Guia breve para revisar evidencias tecnicas del repositorio.

## Machine Learning

| Notebook | Habilidad demostrada | Aplicacion de negocio |
| --- | --- | --- |
| `notebooks/machine_learning/random_forest_iris.ipynb` | Clasificacion multiclase, feature importance, evaluacion | Segmentacion de clientes o clasificacion de tickets |
| `notebooks/machine_learning/mlp_digits_classification.ipynb` | Redes neuronales basicas con `scikit-learn` | OCR simple, automatizacion de captura de datos |
| `notebooks/machine_learning/gradient_boosting_regression.ipynb` | Regresion no lineal y ajuste de hiperparametros | Prediccion de demanda, ventas o tiempos de entrega |

## Ciencia De Datos E IA

| Notebook | Habilidad demostrada | Aplicacion de negocio |
| --- | --- | --- |
| `notebooks/ciencia_datos_ia/kmeans_clustering.ipynb` | Clustering no supervisado | Segmentacion de usuarios, canastas de compra |
| `notebooks/ciencia_datos_ia/decision_tree_classifier.ipynb` | Modelos interpretables basados en reglas | Scoring basico y soporte de decisiones |
| `notebooks/ciencia_datos_ia/svm_classification.ipynb` | Clasificacion con margenes maximos | Deteccion de patrones en datos de alta dimension |
| `notebooks/ciencia_datos_ia/pca_dimensionality_reduction.ipynb` | Reduccion de dimensionalidad y visualizacion | Compresion de variables y preprocesamiento para ML |

## Estadistica Aplicada

| Notebook | Habilidad demostrada | Aplicacion de negocio |
| --- | --- | --- |
| `notebooks/estadistica/descriptive_statistics.ipynb` | Resumen exploratorio y distribuciones | Reportes ejecutivos de estado de datos |
| `notebooks/estadistica/correlation_analysis.ipynb` | Relaciones entre variables | Identificacion de drivers de negocio |
| `notebooks/estadistica/hypothesis_testing_ttest.ipynb` | Contraste de medias | Validacion de cambios en experimentos A/B |
| `notebooks/estadistica/chi_square_test.ipynb` | Dependencia entre variables categoricas | Analisis de comportamiento por segmentos |
| `notebooks/estadistica/anova_analysis.ipynb` | Comparacion de grupos multiples | Evaluacion de variaciones entre regiones/canales |
| `notebooks/estadistica/time_series_decomposition.ipynb` | Tendencia y estacionalidad en series de tiempo | Planeacion de inventario y demanda |

## Scripts De Apoyo

- `scripts/analisis_datos.py`: EDA reproducible con resumen exportado a `reports/eda_summary.md`.
- `scripts/visualizacion_matplotlib.py`: visualizacion reproducible con exporte de figura.
- `scripts/normality_diagnostics.py`: Shapiro + KS + Anderson + QQ-plot para validar supuestos.
- `scripts/churn_classification_pipeline.py`: baseline end-to-end con CV estratificada y metricas ROC-AUC/F1/AP.

## Navegacion Recomendada

1. Ver [PROJECT_STANDARDS.md](PROJECT_STANDARDS.md) para entender el criterio metodologico.
2. Revisar [ROADMAP_PORTFOLIO.md](ROADMAP_PORTFOLIO.md) para el plan de expansion.
3. Ejecutar demos por script para ver reproducibilidad tecnica.
