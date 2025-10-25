#!/bin/bash

# --- Script Principal para Ejecutar el Pipeline de Churn ---
# Este script activa el entorno virtual de Python y luego ejecuta
# el pipeline completo de machine learning.

echo "--- Iniciando el Proceso de Predicción de Churn ---"

# 1. Define la ruta a tu entorno virtual
# Asegúrate de que esta ruta sea correcta para tu sistema.
VENV_PATH="$HOME/.venv/.venv-py311/bin/activate"

# 2. Verifica si el entorno virtual existe y actívalo
if [ -f "$VENV_PATH" ]; then
    echo "Activando el entorno virtual en: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "Error: Entorno virtual no encontrado en '$VENV_PATH'."
    echo "Por favor, ajusta la variable VENV_PATH en este script."
    exit 1
fi

# 3. Ejecuta el pipeline principal de Python
# Este script se encargará de todas las etapas:
# - Creación del target y features (si es necesario)
# - Selección de características
# - Optimización de hiperparámetros con Optuna
# - Entrenamiento de modelos
# - Generación de predicciones y archivos de envío
echo "Ejecutando el ingenieria de datos..."
python3 feature_engineering.py

echo "Ejecutando el pipeline principal (pipeline.py)..."
python3 pipeline_final.py

echo "--- Proceso completado ---"
