PYTHON ?= python
VENV ?= .venv

ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	RM_RF := powershell -NoProfile -Command "if (Test-Path '$(VENV)') { Remove-Item -Recurse -Force '$(VENV)' }; Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force"
else
	VENV_BIN := $(VENV)/bin
	RM_RF := rm -rf $(VENV) && find . -name '__pycache__' -type d -prune -exec rm -rf {} +
endif

PIP := $(VENV_BIN)/pip
PYTHON_BIN := $(VENV_BIN)/python
JUPYTER := $(VENV_BIN)/jupyter

.DEFAULT_GOAL := help

.PHONY: help install notebook lint format clean run-eda normality-demo churn-demo iris-ai titanic-ai wine-ai datasets-demo visual-assets

help:
	@echo "Comandos disponibles:"
	@echo "  make install       Crea un entorno virtual e instala requirements.txt"
	@echo "  make notebook      Lanza JupyterLab usando el entorno virtual"
	@echo "  make lint          Ejecuta ruff para analisis estatico"
	@echo "  make format        Aplica black e isort"
	@echo "  make clean         Elimina artefactos temporales"
	@echo "  make run-eda       Ejecuta EDA sobre MiniProyecto/coffee_db.parquet"
	@echo "  make normality-demo Ejecuta diagnostico de normalidad (demo)"
	@echo "  make churn-demo    Ejecuta clasificacion binaria end-to-end (demo)"
	@echo "  make iris-ai       Ejecuta proyecto IA de clasificacion Iris"
	@echo "  make titanic-ai    Ejecuta proyecto IA de Titanic (OpenML)"
	@echo "  make wine-ai       Ejecuta proyecto IA de regresion Wine Quality (UCI)"
	@echo "  make datasets-demo Descarga datasets de ejemplo en data/raw/"
	@echo "  make visual-assets Genera imagenes para README/portfolio"

$(VENV): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -r requirements.txt

install: $(VENV)
	@echo "Entorno instalado en $(VENV)"

notebook: $(VENV)
	$(PYTHON_BIN) -m ipykernel install --user --name=ciencia-datos-python --display-name "ciencia-datos-python" || true
	$(JUPYTER) lab

lint: $(VENV)
	$(PYTHON_BIN) -m ruff check notebooks scripts src

format: $(VENV)
	$(PYTHON_BIN) -m isort notebooks scripts src
	$(PYTHON_BIN) -m black notebooks scripts src

clean:
	$(RM_RF)

run-eda: $(VENV)
	$(PYTHON_BIN) scripts/analisis_datos.py --top-n 10

normality-demo: $(VENV)
	$(PYTHON_BIN) scripts/normality_diagnostics.py --column Total_domestic_consumption

churn-demo: $(VENV)
	$(PYTHON_BIN) scripts/churn_classification_pipeline.py

iris-ai: $(VENV)
	$(PYTHON_BIN) scripts/project_iris_ai.py

titanic-ai: $(VENV)
	$(PYTHON_BIN) scripts/project_titanic_openml_ai.py

wine-ai: $(VENV)
	$(PYTHON_BIN) scripts/project_wine_quality_ai.py --save-raw

datasets-demo: $(VENV)
	$(PYTHON_BIN) scripts/dataset_connector.py --source sklearn --name iris
	$(PYTHON_BIN) scripts/dataset_connector.py --source openml --name titanic --fallback-sklearn iris
	$(PYTHON_BIN) scripts/dataset_connector.py --source url --name wine_quality_red --fallback-sklearn wine

visual-assets: $(VENV)
	$(PYTHON_BIN) scripts/build_readme_assets.py

