"""
API de inferencia para el dataset Palmer Penguins.

Objetivo:
Exponer modelos previamente entrenados mediante una API REST usando FastAPI.
La API permite:
- Verificar que está activa
- Listar modelos disponibles
- Cambiar el modelo activo
- Realizar predicciones de especie

Los modelos ya fueron entrenados offline y guardados como archivos .joblib.
Aquí solo se cargan y se usan para inferencia.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import joblib
import pandas as pd
from pathlib import Path



# Inicialización de la aplicación
app = FastAPI()


# Carpeta donde están los modelos serializados
MODELS_DIR = Path("models")

# Archivo que contiene el registro de modelos disponibles
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Nombre del modelo actualmente activo
ACTIVE_MODEL_NAME = None

# Pipeline completo cargado en memoria (preprocesamiento + clasificador)
ACTIVE_MODEL_PIPE = None

# Diccionario con la información del registry.json
REGISTRY = None



# Modelos de datos (Pydantic)
class PenguinFeatures(BaseModel):
    """
    Modelo de entrada para predicción.

    Representa las características físicas de un pingüino.
    Algunos campos son opcionales porque el pipeline incluye imputación.
    """

    island: str
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[str] = None
    year: int


class SelectModelRequest(BaseModel):
    """
    Modelo de entrada para cambiar el modelo activo.
    """
    model_name: str

# Funciones auxiliares

def load_registry():
    """
    Carga el archivo registry.json.

    Salida:
        dict con:
            - default_model
            - available_models
    """
    if not REGISTRY_PATH.exists():
        raise RuntimeError("registry.json no existe en /models")

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def load_model(model_name: str):
    """
    Carga un modelo específico desde la carpeta models.

    Entrada:
        model_name (str): nombre del modelo a cargar

    Salida:
        Pipeline entrenado

    Lanza error 404 si el modelo no existe.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado: {model_name}"
        )

    return joblib.load(model_path)


def set_active_model(model_name: str):
    """
    Cambia el modelo activo en memoria.

    Entrada:
        model_name (str)

    Efecto:
        Actualiza ACTIVE_MODEL_PIPE y ACTIVE_MODEL_NAME
    """
    global ACTIVE_MODEL_NAME, ACTIVE_MODEL_PIPE

    ACTIVE_MODEL_PIPE = load_model(model_name)
    ACTIVE_MODEL_NAME = model_name


# Evento de inicio

@app.on_event("startup")
def startup_event():
    """
    Se ejecuta automáticamente cuando la API arranca.

    Proceso:
        1. Carga el registry.json
        2. Identifica el modelo por defecto
        3. Carga ese modelo en memoria
    """
    global REGISTRY

    REGISTRY = load_registry()

    default_model = REGISTRY.get("default_model")

    if default_model is None:
        raise RuntimeError("registry.json no tiene 'default_model'")

    set_active_model(default_model)

# Endpoints

@app.get("/")
def home():
    """
    Endpoint de verificación.

    Salida:
        Mensaje indicando que la API está activa
        Nombre del modelo activo
    """
    return {
        "message": "API Penguins funcionando",
        "active_model": ACTIVE_MODEL_NAME
    }


@app.get("/models")
def list_models():
    """
    Devuelve información sobre los modelos disponibles.

    Salida:
        - Modelo por defecto
        - Lista de modelos disponibles
        - Modelo actualmente activo
    """
    return {
        "default_model": REGISTRY.get("default_model"),
        "available_models": REGISTRY.get("available_models", []),
        "active_model": ACTIVE_MODEL_NAME
    }


@app.post("/select_model")
def select_model(req: SelectModelRequest):
    """
    Permite cambiar el modelo activo.

    Entrada:
        JSON con:
            {
                "model_name": "rf"
            }

    Proceso:
        Verifica que el modelo exista en available_models
        Si existe, lo carga en memoria
        Si no, devuelve error 404
    """
    available = REGISTRY.get("available_models", [])

    if req.model_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no disponible: {req.model_name}"
        )

    set_active_model(req.model_name)

    return {
        "message": "Modelo activo actualizado",
        "active_model": ACTIVE_MODEL_NAME
    }


@app.post("/predict")
def predict(features: PenguinFeatures):
    """
    Realiza una predicción de especie.

    Entrada:
        JSON con las características del pingüino.

    Proceso:
        1. Convierte el input en DataFrame
        2. El pipeline aplica automáticamente:
           - imputación
           - encoding
           - transformación
           - predicción
        3. Devuelve la especie predicha

    Salida:
        {
            "prediction": "Adelie",
            "model_used": "rf"
        }
    """

    if ACTIVE_MODEL_PIPE is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado"
        )

    # Convertimos el JSON en un DataFrame de una sola fila
    df = pd.DataFrame([features.dict()])

    # El pipeline hace todo: preprocesamiento + predicción
    pred = ACTIVE_MODEL_PIPE.predict(df)[0]

    return {
        "prediction": pred,
        "model_used": ACTIVE_MODEL_NAME
    }
