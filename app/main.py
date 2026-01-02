import os
import logging
import io
from contextlib import asynccontextmanager
from typing import Literal, Optional, List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# =========================================================
# LOGGING
# =========================================================
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

artifacts: dict = {}

# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")
        
    loaded = joblib.load(MODEL_PATH)
    artifacts.update(loaded)
    yield

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(
    title="ChurnInsight API", 
    version="1.3.0", 
    lifespan=lifespan
)

# =========================================================
# UTILS
# =========================================================
def gerar_recomendacao(nivel_risco: str) -> str:
    recomendas = {
        "ALTO": "Ação imediata recomendada: contato ativo e oferta personalizada",
        "MÉDIO": "Monitoramento recomendado e campanhas de retenção",
        "BAIXO": "Cliente estável - manutenção padrão"
    }
    return recomendas.get(nivel_risco, "Manutenção padrão")

def calcular_explicabilidade_local(model, X, feature_names, baseline_proba, input_data):
    mapeamento_contrato = {
        "CreditScore": "CreditScore", "Age": "Age", "Tenure": "Tenure",
        "Balance": "Balance", "EstimatedSalary": "EstimatedSalary",
        "Geography_Germany": "Geography", "Geography_Spain": "Geography",
        "Gender_Male": "Gender", "Balance_Salary_Ratio": "Balance",
        "Age_Tenure": "Age", "High_Value_Customer": "Balance"
    }
    impactos = []
    for i, feature in enumerate(feature_names):
        X_mod = X.copy()
        X_mod[0, i] = 0 
        proba_mod = model.predict_proba(X_mod)[0, 1]
        impactos.append((feature, baseline_proba - proba_mod))
    
    impactos_ordenados = sorted(impactos, key=lambda x: x[1], reverse=True)
    features_finais = []
    for feat_interna, _ in impactos_ordenados:
        nome_amigavel = mapeamento_contrato.get(feat_interna)
        if nome_amigavel and nome_amigavel not in features_finais:
            features_finais.append(nome_amigavel)
        if len(features_finais) >= 3: break
    return features_finais

# =========================================================
# MODEL LOGIC 
# =========================================================
def executar_logica_modelo(input_dict: dict) -> dict:
    if not artifacts:
        raise ValueError("Modelo não carregado")

    df = pd.DataFrame([input_dict])

    df["Geography_Germany"] = 1 if str(input_dict.get("Geography")) == "Germany" else 0
    df["Geography_Spain"] = 1 if str(input_dict.get("Geography")) == "Spain" else 0
    df["Gender_Male"] = 1 if str(input_dict.get("Gender")) == "Male" else 0

    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]
    df["High_Value_Customer"] = (
        (df["Balance"] > artifacts.get("balance_median", 0)) &
        (df["EstimatedSalary"] > artifacts.get("salary_median", 0))
    ).astype(int)

    df_final = df[artifacts["columns"]]
    X_scaled = artifacts["scaler"].transform(df_final)

    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    threshold = artifacts.get("threshold", 0.35)

   
    if proba >= threshold:
        risco = "ALTO"
    elif proba >= (threshold * 0.7):
        risco = "MÉDIO"
    else:
        risco = "BAIXO"

    previsao = "Vai cancelar" if proba >= threshold else "Vai continuar"

    explicabilidade = None
    if previsao == "Vai cancelar":
        explicabilidade = calcular_explicabilidade_local(
            artifacts["model"], X_scaled, artifacts["columns"], proba, input_dict
        )

    return {
        "previsao": previsao,
        "probabilidade": round(proba, 4),
        "nivel_risco": risco,
        "recomendacao": gerar_recomendacao(risco),
        "explicabilidade": explicabilidade
    }

# =========================================================
# SCHEMAS
# =========================================================
class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=0, le=1000)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=92)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    EstimatedSalary: float = Field(..., ge=0)

class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str
    explicabilidade: Optional[List[str]] = None

# =========================================================
# ENDPOINTS
# =========================================================
@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    try:
        return executar_logica_modelo(data.model_dump())
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/previsao-lote")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Envie um arquivo .CSV válido")
    try:
        content = await file.read()
        df_input = pd.read_csv(io.BytesIO(content)).head(100)
        resultados = [executar_logica_modelo(row) for row in df_input.to_dict('records')]
        return {"arquivo": file.filename, "resultados": resultados}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health_check():
    return {"status": "UP", "model_loaded": "model" in artifacts}
