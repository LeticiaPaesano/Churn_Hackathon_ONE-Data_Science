import os
import logging
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================================================
# LOGGING — REDUÇÃO DE RUÍDO EM PRODUÇÃO
# =========================================================

logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

# =========================================================
# CONFIGURAÇÃO DE CAMINHOS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS_TO_TRY = [
    os.path.join(BASE_DIR, "..", "models", "model.joblib"),
    os.path.join(BASE_DIR, "..", "model", "model.joblib"),
    os.path.join(BASE_DIR, "model.joblib"),
]

artifacts: dict = {}

# =========================================================
# LIFESPAN — CARREGAMENTO DO MODELO
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = None

    for path in MODEL_PATHS_TO_TRY:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        raise RuntimeError(
            f"Modelo não encontrado. Caminhos verificados: {MODEL_PATHS_TO_TRY}"
        )

    try:
        loaded = joblib.load(model_path)

        artifacts["model"] = loaded["model"]
        artifacts["scaler"] = loaded["scaler"]
        artifacts["columns"] = loaded["columns"]
        artifacts["threshold"] = loaded.get("threshold", 0.35)
        artifacts["balance_median"] = loaded.get("balance_median", 0.0)
        artifacts["salary_median"] = loaded.get("salary_median", 0.0)

        print(f"✅ Modelo carregado com sucesso: {model_path}")
        yield

    except Exception as exc:
        raise RuntimeError(f"Falha crítica ao carregar o modelo: {exc}")

# =========================================================
# APLICAÇÃO FASTAPI
# =========================================================

app = FastAPI(
    title="ChurnInsight API",
    description="API de predição de churn com recomendação orientada a negócio",
    version="1.0.0",
    lifespan=lifespan,
)

# =========================================================
# CONTRATOS DA API
# =========================================================

class CustomerInput(BaseModel):
    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    EstimatedSalary: float


class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str

# =========================================================
# REGRAS DE NEGÓCIO
# =========================================================

def gerar_recomendacao(nivel_risco: str) -> str:
    if nivel_risco == "ALTO":
        return "Ação imediata recomendada: contato ativo e oferta personalizada"
    if nivel_risco == "MÉDIO":
        return "Monitoramento recomendado e campanhas de retenção"
    return "Cliente estável - manutenção padrão"

# =========================================================
# ENDPOINT PRINCIPAL
# =========================================================

@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput) -> PredictionOutput:
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        # 1. Entrada → DataFrame (Pydantic v2)
        df = pd.DataFrame([data.model_dump()])

        # 2. Feature Engineering
        df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
        df["Age_Tenure"] = df["Age"] * df["Tenure"]

        df["High_Value_Customer"] = (
            (df["Balance"] > artifacts["balance_median"])
            & (df["EstimatedSalary"] > artifacts["salary_median"])
        ).astype(int)

        # 3. One-Hot Encoding manual
        for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
            df[col] = 0

        if data.Geography == "Germany":
            df["Geography_Germany"] = 1
        elif data.Geography == "Spain":
            df["Geography_Spain"] = 1

        if data.Gender == "Male":
            df["Gender_Male"] = 1

        # 4. Validação de features
        missing_cols = set(artifacts["columns"]) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=500,
                detail=f"Features ausentes no input: {missing_cols}",
            )

        df_final = df[artifacts["columns"]]

        # 5. Escalonamento
        X_input = artifacts["scaler"].transform(df_final)

        # 6. Predição
        proba = float(artifacts["model"].predict_proba(X_input)[0, 1])
        threshold = artifacts["threshold"]

        if proba >= threshold:
            nivel_risco = "ALTO"
        elif proba >= threshold * 0.7:
            nivel_risco = "MÉDIO"
        else:
            nivel_risco = "BAIXO"

        previsao = "Vai cancelar" if proba >= threshold else "Vai continuar"

        return PredictionOutput(
            previsao=previsao,
            probabilidade=round(proba, 4),
            nivel_risco=nivel_risco,
            recomendacao=gerar_recomendacao(nivel_risco),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na predição: {exc}",
        )

# =========================================================
# ENDPOINT DE DEBUG CONTROLADO
# =========================================================

@app.get("/debug/model")
def debug_model():
    return {
        "status_carregamento": "OK" if "model" in artifacts else "ERRO",
        "model_type": str(type(artifacts.get("model"))),
        "scaler_presente": "scaler" in artifacts,
        "colunas_esperadas": artifacts.get("columns", []),
        "threshold_atual": artifacts.get("threshold"),
        "medianas": {
            "balance": artifacts.get("balance_median"),
            "salary": artifacts.get("salary_median"),
        },
    }
