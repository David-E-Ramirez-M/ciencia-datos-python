# ciencia-datos-python

Este repositorio recopila cuadernos y scripts de Python relacionados con ciencia de datos, estadística y aprendizaje automático. La meta es contar con ejemplos reutilizables listos para ejecutar o adaptar a nuevos proyectos.

## Requisitos previos

- Python 3.10 o superior
- `make` (opcional pero recomendado para simplificar los comandos)

## Configuración rápida del entorno

1. Clona el repositorio y accede a la carpeta del proyecto.
2. Crea un entorno virtual e instala las dependencias:

   ```bash
   make install
   # o, manualmente:
   python -m venv .venv
   source .venv/bin/activate  # En Windows usa .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Registra el kernel de Jupyter (el comando `make notebook` lo hará automáticamente):

   ```bash
   make notebook
   # abre JupyterLab en el navegador con el kernel ciencia-datos-python
   ```

El archivo `requirements.txt` incluye las bibliotecas esenciales para análisis exploratorio, manipulación de datos, visualización y modelado (pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, entre otras). También se añaden herramientas de calidad de código (`black`, `isort`, `ruff`).

### Comandos adicionales

- `make lint`: ejecuta `ruff` sobre `notebooks/`, `scripts/` y `src/`.
- `make format`: aplica `isort` y `black` a los módulos y cuadernos del proyecto.
- `make clean`: elimina el entorno virtual y carpetas de caché de Python.

## Estructura del repositorio

```
├── data/
│   ├── external/         # Datos de fuentes externas (APIs, CSVs públicos, etc.)
│   ├── interim/          # Datos intermedios generados durante el procesamiento
│   ├── processed/        # Datos listos para análisis/modelado
│   └── raw/              # Datos crudos originales (no modificados)
├── models/               # Modelos entrenados y artefactos serializados
├── notebooks/
│   ├── ciencia_datos_ia/ # Técnicas de IA clásica y reducción de dimensionalidad
│   ├── estadistica/      # Análisis estadístico, pruebas de hipótesis, series de tiempo
│   └── machine_learning/ # Experimentación con algoritmos de ML
├── reports/
│   └── figures/          # Gráficos y reportes generados
├── scripts/              # Scripts reutilizables para análisis y utilidades
├── src/                  # Módulos de soporte para notebooks o scripts
├── NTT/                  # Material complementario del proyecto NTT
├── requirements.txt      # Dependencias del entorno científico de datos
├── Makefile              # Atajos para instalar dependencias y lanzar JupyterLab
├── LICENSE
└── README.md
```

## Cómo contribuir

Se aceptan mejoras, nuevos ejemplos y correcciones mediante _pull requests_. ¡Gracias por colaborar!
