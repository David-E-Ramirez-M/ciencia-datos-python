PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter

.DEFAULT_GOAL := help

.PHONY: help install notebook lint format clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install       Crea un entorno virtual e instala requirements.txt"
	@echo "  make notebook      Lanza JupyterLab usando el entorno virtual"
	@echo "  make lint          Ejecuta ruff para análisis estático"
	@echo "  make format        Aplica black e isort"
	@echo "  make clean         Elimina artefactos temporales"

$(VENV): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: $(VENV)
	@echo "Entorno instalado en $(VENV)"

notebook: $(VENV)
	$(PYTHON_BIN) -m ipykernel install --user --name=ciencia-datos-python --display-name "ciencia-datos-python" || true
	$(JUPYTER) lab

lint: $(VENV)
	$(VENV)/bin/ruff check notebooks scripts src

format: $(VENV)
	$(VENV)/bin/isort notebooks scripts src
	$(VENV)/bin/black notebooks scripts src

clean:
	rm -rf $(VENV)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
