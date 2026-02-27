# Roadmap De 10 Proyectos

Plan para evolucionar este repo hacia un portafolio de nivel reclutador.

## Estado Actual

1. EDA: `notebooks/estadistica/descriptive_statistics.ipynb` (hecho, mejorar narrativa de negocio).
2. Clasificacion end-to-end (churn): `scripts/churn_classification_pipeline.py` (base hecha).
3. Regresion aplicada: `notebooks/machine_learning/gradient_boosting_regression.ipynb` (hecho, falta analisis de residuos).
4. Series temporales: `notebooks/estadistica/time_series_decomposition.ipynb` (hecho, falta forecasting comparativo).
5. NLP sentimiento: pendiente.
6. Dashboard ejecutivo (Power BI/Tableau): pendiente.
7. Segmentacion de clientes: `notebooks/ciencia_datos_ia/kmeans_clustering.ipynb` (hecho, falta DBSCAN).
8. Deteccion de fraude desbalanceado: pendiente.
9. Big Data (Spark + SQL): pendiente.
10. Deploy (FastAPI + Docker + cloud): pendiente.

## Prioridad Recomendada

1. Completar proyecto churn con dataset real y conclusion ejecutiva.
2. Agregar fraude desbalanceado con precision-recall y costo de error.
3. Agregar deploy de 1 modelo en API.
4. Agregar 1 dashboard conectado a resultados del modelo.

## Datasets Sugeridos

- Tabular clasificacion/regresion: `fetch_openml` o Kaggle.
- Series de tiempo: datasets de ventas publicos.
- NLP: IMDB (TensorFlow datasets / Kaggle).

## Criterio De Cierre Por Proyecto

- contexto de negocio explicitado
- limpieza y calidad de datos
- baseline + comparacion de modelos
- validacion rigurosa y metricas correctas
- interpretacion y recomendacion accionable
