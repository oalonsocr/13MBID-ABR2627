
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import joblib


# ==========================================
# CONFIGURACION DE LA API
# ==========================================

app = FastAPI(
    title="Modelo de clasificación de clientes bancarios",
    description="API para predecir si un cliente bancario suscribirá un depósito a plazo fijo",
    version="1.0.0"
)


# ==========================================
# CARGA DEL MODELO
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado correctamente: {MODEL_PATH}")

except Exception as e:
    raise RuntimeError(
        f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}"
    )


# ==========================================
# ESQUEMA DE ENTRADA
# ==========================================

class PredictionRequest(BaseModel):

    age: int = Field(ge=17, le=100)
    job: str
    marital: str
    education: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str

    duration: int = Field(ge=0)
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    poutcome: str

    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


# ==========================================
# ESQUEMA DE RESPUESTA
# ==========================================

class PredictionResponse(BaseModel):

    prediction: str
    probability: Dict[str, float]
    model_info: Dict[str, Any]


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Crear grupo etario
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"]
    )

    # 2. Crear indicador de contacto previo
    df["previously_contacted"] = np.where(
        df["pdays"] == 999,
        0,
        1
    )

    # 3. Agrupar cantidad de contactos de campaña
    df["campaign_group"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 3, 5, float("inf")],
        labels=["1", "2-3", "4-5", "6+"]
    )

    return df


# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================

@app.get("/")
def read_root():

    return {
        "message": "API del modelo de clasificación de clientes bancarios en funcionamiento",
        "model": type(model.named_steps["classifier"]).__name__
    }


# ==========================================
# ENDPOINT DE PREDICCION
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:

        # Convertir solicitud a DataFrame
        input_data = pd.DataFrame(
            [request.model_dump()]
        )

        # Aplicar ingeniería de características
        input_data = create_features(input_data)

        # Predecir clase
        prediction = model.predict(input_data)[0]

        # Obtener probabilidades
        probabilities = model.predict_proba(input_data)[0]

        # Obtener etiquetas de las clases
        class_labels = model.classes_

        probability_dict = {
            str(class_labels[i]): float(probabilities[i])
            for i in range(len(class_labels))
        }

        # Información del modelo
        model_info = {
            "model_type": type(model.named_steps["classifier"]).__name__,
            "model_file": MODEL_PATH.name
        }

        return PredictionResponse(
            prediction=str(prediction),
            probability=probability_dict,
            model_info=model_info
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la solicitud: {str(e)}"
        )
