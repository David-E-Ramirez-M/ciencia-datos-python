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

.PHONY: help install notebook lint format clean run-eda normality-demo churn-demo

help:
	@echo "Comandos disponibles:"
	@echo "  make install       Crea un entorno virtual e instala requirements.txt"
	@echo "  make notebook      Lanza JupyterLab usando el entorno virtual"
	@echo "  make lint          Ejecuta ruff para analisis estatico"
	@echo "  make format        Aplica black e isort"
	@echo "  make clean         Elimina artefactos temporales"
	@echo "  make run-eda       Ejecuta EDA sobre NTT/coffee_db.parquet"
	@echo "  make normality-demo Ejecuta diagnostico de normalidad (demo)"
	@echo "  make churn-demo    Ejecuta clasificacion binaria end-to-end (demo)"

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
	$(PYTHON_BIN) scripts/analisis_datos.py --input NTT/coffee_db.parquet --top-n 10

normality-demo: $(VENV)
	$(PYTHON_BIN) scripts/normality_diagnostics.py --input NTT/coffee_db.parquet --column Total_domestic_consumption

churn-demo: $(VENV)
	$(PYTHON_BIN) scripts/churn_classification_pipeline.py

