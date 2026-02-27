# Estandar De Proyecto Data Science

Checklist base para que cada notebook/proyecto del repo se vea profesional.

## 1. Contexto

- problema de negocio (1 parrafo)
- objetivo analitico (que variable o decision impacta)
- criterio de exito (metrica y umbral objetivo)

## 2. Datos

- origen del dataset y fecha de corte
- definicion de variable objetivo
- supuestos y limitaciones
- calidad de datos: nulos, duplicados, outliers

## 3. Metodologia

- baseline simple primero
- feature engineering justificado
- esquema de validacion (train/test + CV estratificada cuando aplique)
- seleccion de metricas alineadas al costo del error

## 4. Inferencia Y Supuestos

- pruebas de normalidad cuando aplique a pruebas parametricas
- QQ-plot e inspeccion visual, no solo p-value
- aclarar cuando normalidad no es critica (arboles/boosting/no parametrico)

## 5. Resultados

- tabla de metricas comparativas
- interpretacion de variables importantes
- errores relevantes y posibles sesgos

## 6. Cierre Ejecutivo

- recomendacion accionable
- riesgo de implementacion
- proximos experimentos priorizados

## Plantilla Minima De Notebook

1. Resumen ejecutivo.
2. Contexto y objetivo.
3. Preparacion de datos.
4. EDA.
5. Modelado y validacion.
6. Interpretacion.
7. Conclusiones.
8. Proximos pasos.
